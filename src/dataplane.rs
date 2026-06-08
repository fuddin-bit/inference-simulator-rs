//! The KV-block data plane: the connector boundary where prefill/decode moves bytes.
//!
//! In real vLLM the produce/consume of `kv_transfer_params` and the NIXL transfer live
//! in the NixlConnector inside the engine/worker. Our mock engine *is* the engine, so
//! this module plays that connector role, wire-compatibly with the llm-d routing
//! sidecar and a real vLLM peer:
//!
//!   - PREFILL registers a (fake) KV region with NIXL, runs a NIXL listener thread, and
//!     advertises how to reach it as [`RemoteKv`] (`remote_engine_id` / `remote_host` /
//!     `remote_port` / `remote_block_ids`). The engine wraps that into the real
//!     `kv_transfer_params` dict the sidecar relays.
//!   - DECODE fetches the prefill agent's NIXL metadata from its listener (host:port),
//!     polls until it loads, then posts a NIXL READ to pull the bytes before generating.
//!
//! ```text
//!   prefill engine                            decode engine
//!   ┌────────────────────┐   kv_transfer_     ┌────────────────────┐
//!   │ register KV w/ NIXL │   params dict      │ fetch_remote_md     │
//!   │ NIXL listener :port │   {remote_host,    │ (host:port), poll,  │
//!   │ advertise RemoteKv ─┼─  port,engine_id, ─┼▶ NIXL READ, verify  │
//!   └────────────────────┘    block_ids}       └────────────────────┘
//! ```
//!
//! The default build ships [`NoopDataPlane`] (control plane only: produces/consumes the
//! real dict but moves no bytes), so the sidecar contract is exercisable with no NIXL.
//! The real transfer lives behind the `nixl` feature.

/// The role this engine plays in a disaggregated prefill/decode deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PdRole {
    /// Registers KV and advertises it for a remote puller.
    Prefill,
    /// Pulls remote KV before generating tokens.
    Decode,
    /// Monolithic: no handoff, behaves like a normal single engine.
    Both,
}

/// Sizing + identity knobs for the data plane.
#[derive(Debug, Clone)]
pub struct NixlConfig {
    /// Bytes per fabricated KV block.
    pub kv_block_bytes: usize,
    /// Prompt tokens that map to one KV block.
    pub tokens_per_block: usize,
    /// This engine's id, advertised as `remote_engine_id` (a real decode peer matches
    /// it against the NIXL agent metadata's engine id).
    pub engine_id: String,
    /// Host a decode peer connects to for the NIXL metadata side channel.
    pub side_channel_host: String,
    /// Port the NIXL metadata side channel listens on.
    pub side_channel_port: u32,
}

/// How a decode peer reaches a prefilled request's KV. `engine_id`/`host`/`port`/
/// `block_ids` are the wire-faithful `remote_*` fields of vLLM's `kv_transfer_params`;
/// `addr`/`len`/`pattern` are a mock extension (real vLLM carries the base address in the
/// NixlAgentMetadata over its own channel, we piggyback it so two mock peers interoperate
/// through the real sidecar without reimplementing that channel).
#[derive(Debug, Clone)]
pub struct RemoteKv {
    pub engine_id: String,
    pub host: String,
    pub port: u32,
    /// Logical KV block ids on the prefill side to read.
    pub block_ids: Vec<i64>,
    /// Base address of the prefill's registered KV buffer.
    pub addr: u64,
    /// Length of that buffer in bytes.
    pub len: u64,
    /// Fill byte, for the decode side to verify the pull moved the right bytes.
    pub pattern: u8,
}

/// The minimal view of a request the data plane needs, decoupled from the wire
/// `EngineCoreRequest` so the protocol crate can evolve without touching this trait.
#[derive(Debug, Clone, Copy)]
pub struct RequestKv<'a> {
    pub request_id: &'a str,
    /// Number of prompt tokens; drives how many KV blocks we fabricate.
    pub num_tokens: usize,
}

/// The connector boundary: where simulated KV cache bytes move between engines.
pub trait KvDataPlane: Send {
    /// Prefill side: register/stage this request's KV and return how a decode peer
    /// reaches it (becomes the `remote_*` fields of `kv_transfer_params`).
    fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> anyhow::Result<RemoteKv>;

    /// Decode side: pull the remote KV described by `remote` before generation.
    /// Returns the number of bytes moved.
    fn pull_prefilled(&mut self, kv: RequestKv<'_>, remote: &RemoteKv) -> anyhow::Result<u64>;

    /// Release any resources staged for a request (prefill dropping its KV buffer).
    fn release(&mut self, _request_id: &str) {}
}

/// Control-plane-only data plane: produces/consumes the real `kv_transfer_params`
/// addressing but moves zero bytes. This is what the default (no-NIXL) binary uses, so
/// the routing-sidecar contract is fully exercisable without libnixl/UCX.
pub struct NoopDataPlane {
    cfg: NixlConfig,
}

impl NoopDataPlane {
    pub fn new(cfg: NixlConfig) -> Self {
        Self { cfg }
    }

    fn block_ids(&self, num_tokens: usize) -> Vec<i64> {
        let n = num_tokens.div_ceil(self.cfg.tokens_per_block).max(1);
        (0..n as i64).collect()
    }
}

impl KvDataPlane for NoopDataPlane {
    fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> anyhow::Result<RemoteKv> {
        Ok(RemoteKv {
            engine_id: self.cfg.engine_id.clone(),
            host: self.cfg.side_channel_host.clone(),
            port: self.cfg.side_channel_port,
            block_ids: self.block_ids(kv.num_tokens),
            addr: 0,
            len: 0,
            pattern: 0,
        })
    }

    fn pull_prefilled(&mut self, _kv: RequestKv<'_>, _remote: &RemoteKv) -> anyhow::Result<u64> {
        Ok(0)
    }
}

/// Build the data plane for a given role. Without the `nixl` feature there is only the
/// no-op plane. With `nixl`, prefill/decode roles get a real NIXL-backed plane; if NIXL
/// init fails (no libnixl/UCX) we degrade to the no-op plane rather than crash, so the
/// same binary still runs as a pure protocol emulator.
pub fn make_data_plane(role: PdRole, cfg: NixlConfig) -> Box<dyn KvDataPlane> {
    #[cfg(feature = "nixl")]
    {
        if !matches!(role, PdRole::Both) {
            match nixl::NixlDataPlane::new(role, cfg.clone()) {
                Ok(plane) => return Box::new(plane),
                Err(error) => {
                    tracing::warn!(%error, "NIXL init failed; using no-op data plane");
                }
            }
        }
    }
    let _ = role;
    Box::new(NoopDataPlane::new(cfg))
}

/// Whether the NIXL bindings are the no-op stubs (true when built without real libnixl,
/// e.g. via `--features nixl-stub`, or when the `nixl` feature is off). Tests use this
/// to skip the real transfer on machines that cannot move bytes.
pub fn nixl_is_stub() -> bool {
    #[cfg(feature = "nixl")]
    {
        nixl_sys::is_stub()
    }
    #[cfg(not(feature = "nixl"))]
    {
        true
    }
}

/// The real NIXL-backed KV data plane.
#[cfg(feature = "nixl")]
mod nixl {
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    use anyhow::{Result, anyhow, bail};
    use nixl_sys::{
        Agent, AgentConfig, Backend, MemType, MemoryRegion as _, NixlError, NixlRegistration as _,
        OptArgs, SystemStorage, XferDescList, XferOp, is_stub,
    };
    use tracing::debug;

    use super::{KvDataPlane, NixlConfig, PdRole, RemoteKv, RequestKv};

    /// How long to poll a NIXL transfer before giving up.
    const XFER_TIMEOUT: Duration = Duration::from_secs(5);

    /// Map a `NixlError` into an `anyhow::Error` (it is not `std::error::Error`).
    fn ne(error: NixlError) -> anyhow::Error {
        anyhow!("nixl: {error:?}")
    }

    pub struct NixlDataPlane {
        role: PdRole,
        cfg: NixlConfig,
        /// NIXL agent name == engine id, so a peer's `remote_engine_id` is the agent name.
        agent: Agent,
        backend: Backend,
        /// Prefill side: registered KV buffers kept alive (and registered) at their
        /// advertised address until `release`d, keyed by request id.
        staged: HashMap<String, SystemStorage>,
    }

    impl NixlDataPlane {
        pub fn new(role: PdRole, cfg: NixlConfig) -> Result<Self> {
            // Both agents run the listener + prog threads (as in NIXL's basic_two_peers):
            // prefill serves its metadata, decode fetches it asynchronously and needs the
            // prog thread to process the response. Each listens on its own side-channel port.
            let acfg = AgentConfig {
                enable_prog_thread: true,
                enable_listen_thread: true,
                listen_port: cfg.side_channel_port as i32,
                ..AgentConfig::default()
            };
            let agent = Agent::new_configured(&cfg.engine_id, &acfg).map_err(ne)?;
            let (_, params) = agent.get_plugin_params("UCX").map_err(ne)?;
            let backend = agent.create_backend("UCX", &params).map_err(ne)?;
            debug!(
                engine_id = cfg.engine_id,
                ?role,
                listen_port = cfg.side_channel_port,
                "NIXL data plane ready (UCX backend)"
            );
            Ok(Self {
                role,
                cfg,
                agent,
                backend,
                staged: HashMap::new(),
            })
        }

        /// Fresh opt args carrying our backend (required for register and transfer calls).
        fn opt_args(&self) -> Result<OptArgs> {
            let mut opt = OptArgs::new().map_err(ne)?;
            opt.add_backend(&self.backend).map_err(ne)?;
            Ok(opt)
        }

        fn do_advertise(&mut self, kv: RequestKv<'_>) -> Result<RemoteKv> {
            let n_blocks = kv.num_tokens.div_ceil(self.cfg.tokens_per_block).max(1);
            let total = n_blocks * self.cfg.kv_block_bytes;

            let mut storage = SystemStorage::new(total).map_err(ne)?;
            let pattern = pattern_for(kv.request_id);
            storage.memset(pattern);
            let opt = self.opt_args()?;
            storage.register(&self.agent, Some(&opt)).map_err(ne)?;
            // SAFETY: the buffer is a heap `Vec<u8>`; its address is stable across the
            // move into `staged` (moving the Vec header does not reallocate).
            let addr = unsafe { storage.as_ptr() as u64 };

            self.staged.insert(kv.request_id.to_string(), storage);
            debug!(
                request_id = kv.request_id,
                blocks = n_blocks,
                bytes = total,
                "advertised KV"
            );

            Ok(RemoteKv {
                engine_id: self.cfg.engine_id.clone(),
                host: self.cfg.side_channel_host.clone(),
                port: self.cfg.side_channel_port,
                block_ids: (0..n_blocks as i64).collect(),
                addr,
                len: total as u64,
                pattern,
            })
        }

        /// Pull a remote prefill's KV: fetch its NIXL metadata from its listener at
        /// host:port, then post a NIXL READ from the advertised buffer into a fresh local
        /// landing buffer, and verify the pattern. Decode and prefill are distinct agents.
        fn do_pull(&mut self, kv: RequestKv<'_>, remote: &RemoteKv) -> Result<u64> {
            if remote.len == 0 {
                // Peer advertised no real buffer (e.g. a no-op prefill plane); nothing to move.
                return Ok(0);
            }

            // The remote source descriptor (the prefill's registered KV buffer).
            let mut remote_descs = XferDescList::new(MemType::Dram).map_err(ne)?;
            remote_descs.add_desc(remote.addr as usize, remote.len as usize, 0);

            // Fetch the prefill agent's metadata over its listener socket (ip+port in the
            // opt args). The fetch is async, so poll check_remote_metadata (for these descs)
            // until the metadata has actually loaded before creating the transfer.
            let mut fetch_opt = self.opt_args()?;
            fetch_opt.set_ip_addr(&remote.host).map_err(ne)?;
            fetch_opt.set_port(remote.port as u16).map_err(ne)?;
            self.agent
                .fetch_remote_md(&remote.engine_id, Some(&fetch_opt))
                .map_err(ne)?;
            let start = Instant::now();
            while !self
                .agent
                .check_remote_metadata(&remote.engine_id, Some(&remote_descs))
            {
                if start.elapsed() > XFER_TIMEOUT {
                    bail!("timed out fetching NIXL metadata for {}", remote.engine_id);
                }
                std::thread::sleep(Duration::from_millis(5));
            }

            // Land the bytes in a fresh, registered local buffer.
            let opt = self.opt_args()?;
            let mut dst = SystemStorage::new(remote.len as usize).map_err(ne)?;
            dst.register(&self.agent, Some(&opt)).map_err(ne)?;
            // SAFETY: stable heap address of the Vec backing the storage.
            let dst_base = unsafe { dst.as_ptr() as usize };

            let mut local = XferDescList::new(MemType::Dram).map_err(ne)?;
            local.add_desc(dst_base, remote.len as usize, 0);

            let req = self
                .agent
                .create_xfer_req(
                    XferOp::Read,
                    &local,
                    &remote_descs,
                    &remote.engine_id,
                    Some(&opt),
                )
                .map_err(ne)?;
            self.agent.post_xfer_req(&req, Some(&opt)).map_err(ne)?;

            let start = Instant::now();
            loop {
                if self.agent.get_xfer_status(&req).map_err(ne)?.is_success() {
                    break;
                }
                if start.elapsed() > XFER_TIMEOUT {
                    bail!(
                        "NIXL READ from {} timed out after {XFER_TIMEOUT:?}",
                        remote.engine_id
                    );
                }
                std::thread::sleep(Duration::from_micros(200));
            }

            let got = dst.as_slice().first().copied();
            if got != Some(remote.pattern) {
                bail!(
                    "KV verify failed: expected 0x{:02x}, got {got:?}",
                    remote.pattern
                );
            }
            debug!(
                request_id = kv.request_id,
                bytes = remote.len,
                engine_id = remote.engine_id,
                "pulled + verified KV over NIXL"
            );
            Ok(remote.len)
        }
    }

    impl KvDataPlane for NixlDataPlane {
        fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> Result<RemoteKv> {
            // Even under stub we can return addressing (control plane); only the transfer
            // is a no-op. But stub register/get_local_md may error, so short-circuit.
            if is_stub() {
                return Ok(RemoteKv {
                    engine_id: self.cfg.engine_id.clone(),
                    host: self.cfg.side_channel_host.clone(),
                    port: self.cfg.side_channel_port,
                    block_ids: vec![0],
                    addr: 0,
                    len: 0,
                    pattern: 0,
                });
            }
            self.do_advertise(kv)
        }

        fn pull_prefilled(&mut self, kv: RequestKv<'_>, remote: &RemoteKv) -> Result<u64> {
            if is_stub() {
                return Ok(0);
            }
            self.do_pull(kv, remote)
        }

        fn release(&mut self, request_id: &str) {
            if self.staged.remove(request_id).is_some() {
                debug!(request_id, role = ?self.role, "released staged KV");
            }
        }
    }

    /// Stable per-request fill byte, so a decode pull can verify it got the right bytes.
    fn pattern_for(request_id: &str) -> u8 {
        request_id
            .bytes()
            .fold(0xa5u8, |acc, b| acc.wrapping_add(b))
            | 1
    }
}
