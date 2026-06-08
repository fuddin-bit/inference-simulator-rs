//! The KV-block data plane: the boundary where simulated prefill/decode actually
//! moves bytes between engines.
//!
//! Today's `llm-d-inference-sim` fakes P/D purely in the control plane: it changes
//! the latency model and tags a finish reason, but no KV cache bytes ever move.
//! This module is where that changes. A prefill engine registers fake KV-block
//! buffers and advertises them; a decode engine pulls those bytes over NIXL
//! (UCX/DRAM, or real RDMA NICs) before it "decodes". No GPU required.
//!
//! ```text
//!   prefill engine                         decode engine
//!   ┌────────────────────┐                 ┌────────────────────┐
//!   │ generate fake KV    │  kv_transfer_   │ pull blocks over    │
//!   │ register w/ NIXL    │  params (JSON)  │ NIXL, then decode   │
//!   │ advertise_prefilled ├────────────────▶│ pull_prefilled      │
//!   └────────────────────┘   (side channel  └────────────────────┘
//!                             via frontend)
//! ```
//!
//! The default build ships [`NoopDataPlane`] (control-plane only, byte-for-byte the
//! current sim behavior). The real transfer lives behind the `nixl` feature.

use serde_json::Value;

/// The role this engine plays in a disaggregated prefill/decode deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PdRole {
    /// Computes (simulated) KV and hands it off. Advertises blocks for a remote puller.
    Prefill,
    /// Pulls remote KV before generating tokens.
    Decode,
    /// Monolithic: no handoff, behaves like a normal single engine.
    Both,
}

/// The minimal view of a request the data plane needs, decoupled from the wire
/// `EngineCoreRequest` so the protocol crate can evolve without touching this trait.
#[derive(Debug, Clone, Copy)]
pub struct RequestKv<'a> {
    pub request_id: &'a str,
    /// Number of prompt tokens; drives how many KV blocks we fabricate.
    pub num_tokens: usize,
}

/// Where simulated KV cache bytes actually move between a prefill and a decode engine.
pub trait KvDataPlane: Send {
    /// Called on a prefill engine once a request's simulated KV cache is "ready".
    /// Returns the `kv_transfer_params` to embed in the engine output so the decode
    /// engine can locate and pull these blocks. `None` advertises nothing.
    fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> Option<Value>;

    /// Called on a decode engine before generation when a request carries
    /// `kv_transfer_params`. Pulls the remote KV blocks. Returns bytes moved.
    fn pull_prefilled(&mut self, kv: RequestKv<'_>, params: &Value) -> anyhow::Result<u64>;
}

/// Control-plane-only data plane: models P/D handoff but moves zero bytes. This is
/// exactly what the simulator does today, kept as the default so the protocol spike
/// (bird one) needs no NIXL/UCX install.
pub struct NoopDataPlane;

impl KvDataPlane for NoopDataPlane {
    fn advertise_prefilled(&mut self, _kv: RequestKv<'_>) -> Option<Value> {
        None
    }

    fn pull_prefilled(&mut self, _kv: RequestKv<'_>, _params: &Value) -> anyhow::Result<u64> {
        Ok(0)
    }
}

/// Build the data plane for a given role. Without the `nixl` feature there is only
/// the no-op plane, so the default binary is a faithful protocol emulator and nothing
/// more. With `nixl`, prefill/decode roles get a real NIXL-backed plane (bird two).
pub fn make_data_plane(role: PdRole) -> Box<dyn KvDataPlane> {
    #[cfg(feature = "nixl")]
    {
        if !matches!(role, PdRole::Both) {
            return Box::new(nixl::NixlDataPlane::new(role));
        }
    }
    let _ = role;
    Box::new(NoopDataPlane)
}

/// The real NIXL-backed KV data plane. Stubbed for now: the integration point is
/// in place and typechecks against the bindings, but the register/transfer dance
/// (bird two) is the next step, intentionally not faked here.
#[cfg(feature = "nixl")]
mod nixl {
    use super::{KvDataPlane, PdRole, RequestKv, Value};
    use tracing::warn;

    pub struct NixlDataPlane {
        role: PdRole,
        // TODO(bird-two): own a `nixl_sys::Agent`, the registered KV buffer pool,
        // and the per-peer metadata once the sim-to-sim handshake is designed.
    }

    impl NixlDataPlane {
        pub fn new(role: PdRole) -> Self {
            warn!(
                ?role,
                "NIXL data plane selected but transfer path is not implemented yet"
            );
            Self { role }
        }
    }

    impl KvDataPlane for NixlDataPlane {
        fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> Option<Value> {
            // TODO(bird-two): register fake KV blocks with NIXL, serialize the
            // descriptors + agent metadata into kv_transfer_params.
            let _ = (self.role, kv);
            None
        }

        fn pull_prefilled(&mut self, kv: RequestKv<'_>, _params: &Value) -> anyhow::Result<u64> {
            // TODO(bird-two): deserialize remote descriptors, post a NIXL READ over
            // UCX, poll to completion, return bytes moved.
            let _ = (self.role, kv);
            Ok(0)
        }
    }
}
