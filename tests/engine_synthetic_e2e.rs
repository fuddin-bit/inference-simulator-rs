//! End-to-end integration tests for the engine using synthetic traces.
//!
//! Three focused tests over real ZMQ transport and `EngineCoreClient`:
//!   - Two `--latency-trace` tests (timing replay; no recorded token ids required)
//!   - One `--replay-tokens` test (hand-built trace with `arrival_ms` + `output_token_ids`)

use std::time::Duration;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_vcr::{Opt, run};

mod synthetic_trace_generator;
mod test_helpers;

use synthetic_trace_generator::*;
use test_helpers::*;

const TIMEOUT: Duration = Duration::from_secs(30);

/// Spin up the simulator with the given CLI flags, connect a real client.
/// Returns `(client, guard)`. The guard cancels the sim on drop.
async fn harness(test_name: &str, extra_flags: &[&str]) -> (EngineCoreClient, SimGuard) {
    let addr = unique_ipc_endpoint(test_name);

    let mut args: Vec<&str> = vec!["play", "--handshake-address", &addr];
    args.extend_from_slice(extra_flags);

    let opt = Opt::parse_from(&args);
    let token = CancellationToken::new();
    let guard = SimGuard {
        token: token.clone(),
    };

    // Spawn the simulator task.
    let sim_opt = opt.clone();
    let sim_token = token.clone();
    tokio::spawn(async move {
        let _ = run(sim_opt, sim_token).await;
    });

    // Connect the real client.
    let config = EngineCoreClientConfig::new_single(&addr);
    let client = tokio::time::timeout(Duration::from_secs(30), EngineCoreClient::connect(config))
        .await
        .expect("client connect timed out")
        .expect("client connect failed");

    (client, guard)
}

/// Build a simple request with repeated token IDs.
fn make_request(id: &str, prompt_len: usize, max_tokens: u32) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: id.to_string(),
        prompt_token_ids: Some(vec![42u32; prompt_len]),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens,
            ..EngineCoreSamplingParams::for_test()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn test_basic_trace_latency_replay() {
    let (meta, records) = generate_basic_trace(10, 54321);
    let trace_file =
        create_temp_trace("basic_latency_replay", &meta, &records).expect("create trace file");
    let trace_path = trace_file.path().to_str().expect("path to UTF-8");

    let (client, _guard) = harness("basic_latency_replay", &["--latency-trace", trace_path]).await;

    for (i, record) in records.iter().enumerate().take(3) {
        let req = make_request(
            &format!("latency-{}", i),
            record.prompt_tokens,
            record.output_tokens as u32,
        );

        let stream = client.call(req).await.expect("call failed");
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream collect timed out");

        let total_tokens: usize = outputs
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|o| o.new_token_ids.len())
            .sum();
        assert_eq!(total_tokens, record.output_tokens);
    }
}

#[tokio::test]
async fn test_batch_context_latency_replay() {
    let (meta, records) = generate_batch_context_trace(15, 11111);
    let trace_file =
        create_temp_trace("batch_context_latency", &meta, &records).expect("create trace file");
    let trace_path = trace_file.path().to_str().expect("path to UTF-8");

    let (client, _guard) = harness("batch_context_latency", &["--latency-trace", trace_path]).await;

    for (i, record) in records.iter().enumerate().take(3) {
        let req = make_request(
            &format!("latency-{}", i),
            record.prompt_tokens,
            record.output_tokens as u32,
        );

        let stream = client.call(req).await.expect("call failed");
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream collect timed out");

        assert!(!outputs.is_empty(), "should have outputs for record {}", i);
        let total_tokens: usize = outputs
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|o| o.new_token_ids.len())
            .sum();
        assert_eq!(total_tokens, record.output_tokens);
    }
}

#[tokio::test]
async fn test_replay_tokens_serves_recorded_ids() {
    let (meta, records) = generate_token_replay_trace();
    let (expected_early, expected_late) = token_replay_expected_ids();
    let trace_file = create_temp_trace("token_replay", &meta, &records).expect("create trace file");
    let trace_path = trace_file.path().to_str().expect("path to UTF-8");

    let (client, _guard) = harness("token_replay", &["--replay-tokens", trace_path]).await;

    // replay-0 = arrival_ms 0.0 (early record).
    let early = &records[1];
    let req = make_request("replay-0", early.prompt_tokens, early.output_tokens as u32);
    let stream = client.call(req).await.expect("call failed");
    let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream collect timed out");
    let tokens: Vec<u32> = outputs
        .iter()
        .flat_map(|r| r.as_ref().expect("stream item error").new_token_ids.clone())
        .collect();
    assert_eq!(tokens, expected_early, "replay-0 serves the early arrival");
    assert_eq!(
        outputs.last().unwrap().as_ref().unwrap().finish_reason,
        Some(EngineCoreFinishReason::Stop),
        "replay-0 ends with recorded finish reason"
    );

    // replay-1 = arrival_ms 100.0 (late record).
    let late = &records[0];
    let req = make_request("replay-1", late.prompt_tokens, late.output_tokens as u32);
    let stream = client.call(req).await.expect("call failed");
    let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream collect timed out");
    let tokens: Vec<u32> = outputs
        .iter()
        .flat_map(|r| r.as_ref().expect("stream item error").new_token_ids.clone())
        .collect();
    assert_eq!(tokens, expected_late, "replay-1 serves the late arrival");
    assert_eq!(
        outputs.last().unwrap().as_ref().unwrap().finish_reason,
        Some(EngineCoreFinishReason::Length),
        "replay-1 ends with recorded finish reason"
    );
}
