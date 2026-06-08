//! The per-engine generation loop. Adapted from vLLM's in-tree `vllm-mock-engine`
//! (`rust/src/mock-engine/src/engine.rs`), with the prefill/decode data-plane hooks
//! added at the two points where real KV bytes would move.
//!
//! Everything wire-facing comes from the `vllm-engine-core-client` crate, so this
//! stays correct as the protocol evolves upstream.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use rmpv::Value as MsgpackValue;
use serde::Serialize;
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use tokio::task::yield_now;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core_client::protocol::utility::{
    EngineCoreUtilityRequest, UtilityOutput, UtilityResultEnvelope,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
};

use crate::Opt;
use crate::dataplane::{KvDataPlane, NixlConfig, PdRole, RequestKv, make_data_plane};

/// Derive a stable per-request seed from the CLI seed, engine, and request id.
fn request_seed(base_seed: u64, engine_index: u32, request_id: &str) -> u64 {
    use std::hash::{Hash as _, Hasher as _};
    let mut hasher = std::hash::DefaultHasher::new();
    base_seed.hash(&mut hasher);
    engine_index.hash(&mut hasher);
    request_id.hash(&mut hasher);
    hasher.finish()
}

/// Current UNIX timestamp in seconds for engine-core output envelopes.
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

/// Build one request output with only token IDs and terminal status populated.
fn request_output(
    request_id: String,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id,
        new_token_ids,
        finish_reason,
        ..Default::default()
    }
}

/// Produce an empty output with a terminal finish reason for an invalid request.
fn empty_finish_outputs(
    engine_index: u32,
    request_id: String,
    finish_reason: EngineCoreFinishReason,
) -> EngineCoreOutputs {
    let output = request_output(request_id, Vec::new(), Some(finish_reason));
    let finished_requests = BTreeSet::from([output.request_id.clone()]);

    EngineCoreOutputs {
        engine_index,
        outputs: vec![output],
        timestamp: now_secs(),
        finished_requests: Some(finished_requests),
        ..Default::default()
    }
}

/// Encode a utility result into the protocol's msgpack value envelope.
fn utility_envelope<T>(value: T) -> Result<UtilityResultEnvelope>
where
    T: Serialize,
{
    Ok(UtilityResultEnvelope::without_type_info(
        rmpv::ext::to_value(value)?,
    ))
}

/// Produce the minimal utility responses needed by a real frontend.
fn utility_response(
    engine_index: u32,
    request: EngineCoreUtilityRequest,
) -> Result<EngineCoreOutputs> {
    let result = match request.method_name.as_str() {
        "get_supported_tasks" => utility_envelope(vec!["generate"]),
        "is_sleeping" => utility_envelope(false),
        "reset_prefix_cache" => utility_envelope(true),
        "reset_mm_cache"
        | "reset_encoder_cache"
        | "profile"
        | "sleep"
        | "wake_up"
        | "execute_dummy_batch" => utility_envelope(()),
        _ => utility_envelope(MsgpackValue::Nil),
    }?;

    Ok(EngineCoreOutputs {
        engine_index,
        utility_output: Some(UtilityOutput {
            call_id: request.call_id,
            failure_message: None,
            result: Some(result),
        }),
        timestamp: now_secs(),
        ..Default::default()
    })
}

/// The `kv_transfer_params` the frontend ferries down from the OpenAI request. The
/// server merges them into `sampling_params.extra_args["kv_transfer_params"]`
/// (mirroring Python vLLM), so that is where the P/D intent (`do_remote_prefill` /
/// `do_remote_decode` / `remote_*`) arrives. In real vLLM the produce/consume logic
/// lives in the NixlConnector inside the engine; here our data plane plays that role.
fn extract_kv_params(request: &EngineCoreRequest) -> Option<JsonValue> {
    request
        .sampling_params
        .as_ref()?
        .extra_args
        .as_ref()?
        .get("kv_transfer_params")
        .cloned()
}

/// Phase 0 probe: a prefill-style request (`do_remote_decode: true`) gets a stub
/// remote descriptor echoed back so we can confirm `kv_transfer_params` survives the
/// full round trip through the real frontend. The real values land in the wire-compat
/// rework (mirroring `nixl_connector.py`); this only proves the channel.
fn phase0_probe_response(kv: &JsonValue) -> Option<JsonValue> {
    let is_prefill = kv
        .get("do_remote_decode")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    is_prefill.then(|| {
        serde_json::json!({
            "remote_block_ids": [0],
            "remote_engine_id": "mock-engine-nixl",
            "remote_host": "127.0.0.1",
            "remote_port": 0,
            "phase0_probe": true,
        })
    })
}

/// Message sent from the IO loop to the engine task to drive the engine loop.
pub(crate) enum EngineInput {
    Request(Box<EngineCoreRequest>),
    Abort(Vec<String>),
    Utility(EngineCoreUtilityRequest),
    StartDpWave,
}

/// Message sent from the engine task to the IO loop for one engine output batch.
pub(crate) struct EngineOutput {
    pub client_index: u32,
    pub outputs: EngineCoreOutputs,
}

/// Per-request decode state owned by one engine.
#[derive(Debug)]
struct ActiveRequest {
    request_id: String,
    client_index: u32,
    prompt_len: usize,
    max_tokens: usize,
    generated: usize,
    rng: StdRng,
    /// `kv_transfer_params` to stamp on this request's finishing output (Phase 0 probe
    /// today; the prefill engine's real remote descriptor after the wire-compat rework).
    advertise_on_finish: Option<JsonValue>,
}

impl ActiveRequest {
    /// Create a new active request, or return an immediate finish reason if invalid.
    fn new(
        engine_index: u32,
        request: Box<EngineCoreRequest>,
        opt: &Opt,
    ) -> Result<Self, EngineCoreFinishReason> {
        let incoming_kv = extract_kv_params(&request);
        let advertise_on_finish = incoming_kv.as_ref().and_then(phase0_probe_response);
        let request_id = request.request_id;
        let client_index = request.client_index;
        let prompt_len = request
            .prompt_token_ids
            .as_ref()
            .map(Vec::len)
            .unwrap_or_default();

        let Some(sampling_params) = request.sampling_params else {
            warn!(
                request_id,
                "request has no sampling params; returning engine error"
            );
            return Err(EngineCoreFinishReason::Error);
        };
        let max_tokens = sampling_params.max_tokens as usize;

        if let Some(kv) = &incoming_kv {
            info!(request_id, %kv, "received kv_transfer_params from frontend");
        }

        if opt.log_requests {
            info!(
                request_id,
                prompt_len,
                max_tokens,
                chunk_size = opt.output_token_chunk_size,
                "mock request started"
            );
        }

        if max_tokens == 0 {
            return Err(EngineCoreFinishReason::Length);
        }

        Ok(ActiveRequest {
            rng: StdRng::seed_from_u64(request_seed(opt.seed, engine_index, &request_id)),
            request_id,
            client_index,
            prompt_len,
            max_tokens,
            generated: 0,
            advertise_on_finish,
        })
    }

    /// Advance this request by one engine step.
    fn step(&mut self, opt: &Opt) -> EngineCoreOutput {
        let remaining = self.max_tokens - self.generated;
        let chunk_len = remaining.min(opt.output_token_chunk_size);
        let mut new_token_ids = Vec::with_capacity(chunk_len);
        for _ in 0..chunk_len {
            new_token_ids.push(self.rng.random_range(0..opt.vocab_size));
        }
        self.generated += chunk_len;

        let finished = self.generated >= self.max_tokens;
        let mut output = request_output(
            self.request_id.clone(),
            new_token_ids,
            finished.then_some(EngineCoreFinishReason::Length),
        );
        if finished {
            output.kv_transfer_params = self.advertise_on_finish.take();
        }
        output
    }
}

/// Internal state for one engine instance, owned by the engine loop task.
struct Engine {
    engine_index: u32,
    opt: Opt,
    role: PdRole,
    data_plane: Box<dyn KvDataPlane>,
    active_requests: HashMap<String, ActiveRequest>,
}

impl Engine {
    /// Whether this engine advertises prefilled KV (prefill side of a P/D pair).
    fn advertises(&self) -> bool {
        matches!(self.role, PdRole::Prefill)
    }

    /// Drain one frontend request message.
    fn handle_input(&mut self, input: EngineInput) -> Result<Vec<EngineOutput>> {
        let mut outputs = Vec::new();

        match input {
            EngineInput::Request(request) => {
                let request_id = request.request_id.clone();
                let client_index = request.client_index;

                if self.active_requests.contains_key(&request_id) {
                    warn!(
                        engine_index = self.engine_index,
                        request_id, "duplicate request id"
                    );
                    return Ok(vec![EngineOutput {
                        client_index,
                        outputs: empty_finish_outputs(
                            self.engine_index,
                            request_id,
                            EngineCoreFinishReason::Error,
                        ),
                    }]);
                }

                // === DATA PLANE: decode-side pull (bird two) ===
                // A decode engine pulls remote KV before it starts generating.
                if matches!(self.role, PdRole::Decode)
                    && let Some(params) = extract_kv_params(&request)
                {
                    let kv = RequestKv {
                        request_id: &request_id,
                        num_tokens: 0,
                    };
                    match self.data_plane.pull_prefilled(kv, &params) {
                        Ok(bytes) => {
                            debug!(request_id, bytes, "pulled remote KV before decode")
                        }
                        Err(error) => warn!(request_id, %error, "remote KV pull failed"),
                    }
                }

                match ActiveRequest::new(self.engine_index, request, &self.opt) {
                    Ok(request) => {
                        self.active_requests.insert(request_id, request);
                    }
                    Err(finish_reason) => {
                        return Ok(vec![EngineOutput {
                            client_index,
                            outputs: empty_finish_outputs(
                                self.engine_index,
                                request_id,
                                finish_reason,
                            ),
                        }]);
                    }
                }
            }

            EngineInput::Abort(request_ids) => {
                let mut by_client =
                    BTreeMap::<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>::new();
                for request_id in request_ids {
                    self.data_plane.release(&request_id);
                    if let Some(request) = self.active_requests.remove(&request_id) {
                        let output = request_output(
                            request_id.clone(),
                            Vec::new(),
                            Some(EngineCoreFinishReason::Abort),
                        );
                        let (outs, finished) = by_client
                            .entry(request.client_index)
                            .or_insert_with(|| (Vec::new(), BTreeSet::new()));
                        outs.push(output);
                        finished.insert(request_id.clone());
                        if self.opt.log_requests {
                            info!(request_id, finish_reason = "abort", "request aborted");
                        }
                    }
                }
                for (client_index, (client_outputs, finished_requests)) in by_client {
                    outputs.push(EngineOutput {
                        client_index,
                        outputs: EngineCoreOutputs {
                            engine_index: self.engine_index,
                            outputs: client_outputs,
                            timestamp: now_secs(),
                            finished_requests: Some(finished_requests),
                            ..Default::default()
                        },
                    });
                }
            }

            EngineInput::Utility(request) => {
                debug!(
                    engine_index = self.engine_index,
                    call_id = %request.call_id,
                    method = request.method_name,
                    "utility request"
                );
                let client_index = request.client_index;
                outputs.push(EngineOutput {
                    client_index,
                    outputs: utility_response(self.engine_index, request)?,
                });
            }

            EngineInput::StartDpWave => {
                debug!(engine_index = self.engine_index, "ignoring START_DP_WAVE");
            }
        }

        Ok(outputs)
    }

    /// Advance active requests once and return one batched engine output per client.
    fn step(&mut self) -> Vec<EngineOutput> {
        if self.active_requests.is_empty() {
            return Vec::new();
        }

        let advertises = self.advertises();
        let mut by_client = BTreeMap::<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>::new();
        let mut finished_ids = BTreeSet::new();

        // Collect (request_id, num_tokens) for finished requests so we can advertise
        // their KV after the borrow on active_requests ends.
        let mut to_advertise: Vec<(u32, String, usize)> = Vec::new();

        for request in self.active_requests.values_mut() {
            let client_index = request.client_index;
            let mut output = request.step(&self.opt);
            let request_id = request.request_id.clone();
            let finished = output.finished();
            if finished {
                finished_ids.insert(request_id.clone());
                if advertises {
                    to_advertise.push((client_index, request_id.clone(), request.prompt_len));
                }
                if self.opt.log_requests {
                    info!(
                        request_id,
                        prompt_len = request.prompt_len,
                        output_tokens = request.generated,
                        finish_reason = "length",
                        "request finished"
                    );
                }
            }
            let (outs, finished_set) = by_client
                .entry(client_index)
                .or_insert_with(|| (Vec::new(), BTreeSet::new()));
            if finished {
                finished_set.insert(request_id);
            }
            // Hold the output so we can attach kv_transfer_params below if advertised.
            outs.push(std::mem::take(&mut output));
        }

        for request_id in &finished_ids {
            self.active_requests.remove(request_id);
        }

        // === DATA PLANE: prefill-side advertise (bird two) ===
        // Once prefill finishes, register the fake KV and stamp the descriptors onto
        // the finishing output's kv_transfer_params for the decode engine to pull.
        for (client_index, request_id, num_tokens) in to_advertise {
            let kv = RequestKv {
                request_id: &request_id,
                num_tokens,
            };
            if let Some(params) = self.data_plane.advertise_prefilled(kv)
                && let Some((outs, _)) = by_client.get_mut(&client_index)
                && let Some(out) = outs.iter_mut().find(|o| o.request_id == request_id)
            {
                out.kv_transfer_params = Some(params);
            }
        }

        by_client
            .into_iter()
            .filter_map(|(client_index, (outputs, finished_requests))| {
                (!outputs.is_empty()).then(|| EngineOutput {
                    client_index,
                    outputs: EngineCoreOutputs {
                        engine_index: self.engine_index,
                        outputs,
                        timestamp: now_secs(),
                        finished_requests: (!finished_requests.is_empty())
                            .then_some(finished_requests),
                        ..Default::default()
                    },
                })
            })
            .collect()
    }
}

/// Run the main loop for one engine, receiving `EngineInput` and sending `EngineOutput`
/// until `shutdown` is cancelled.
pub(crate) async fn run_engine_loop(
    engine_index: u32,
    opt: Opt,
    mut input_rx: mpsc::UnboundedReceiver<EngineInput>,
    output_tx: mpsc::Sender<EngineOutput>,
    shutdown: CancellationToken,
) -> Result<()> {
    let role = opt.pd_role;
    let cfg = NixlConfig {
        kv_block_bytes: opt.kv_block_bytes,
        tokens_per_block: opt.tokens_per_block,
    };
    let mut engine = Engine {
        engine_index,
        opt,
        role,
        data_plane: make_data_plane(role, cfg),
        active_requests: HashMap::new(),
    };

    loop {
        let outputs = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,

            input = input_rx.recv() => {
                let input = input.ok_or_else(|| anyhow!("engine input channel closed"))?;
                engine.handle_input(input)?
            }

            _ = yield_now(), if !engine.active_requests.is_empty() => {
                engine.step()
            }
        };

        for output in outputs {
            output_tx
                .send(output)
                .await
                .map_err(|_| anyhow!("engine IO task shut down"))?;
        }
    }

    Ok(())
}
