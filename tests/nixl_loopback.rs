//! The first real NIXL transfer: a prefill agent registers + advertises a patterned KV
//! buffer and serves its metadata over a listener; a separate decode agent fetches that
//! metadata by host:port, posts a NIXL READ, and verifies the bytes. Two distinct agents
//! in one process (same shape as the cross-pod path, just over loopback).
//!
//! Needs real libnixl + UCX, so it runs on Linux and skips under the stub bindings.
//! Run it: `cargo test --features nixl` on a box with NIXL installed.

#![cfg(feature = "nixl")]

use inference_simulator_rs::dataplane::{
    NixlConfig, PdRole, RequestKv, make_data_plane, nixl_is_stub,
};

fn cfg(engine_id: &str, port: u32) -> NixlConfig {
    NixlConfig {
        kv_block_bytes: 4096,
        tokens_per_block: 16,
        engine_id: engine_id.to_string(),
        side_channel_host: "127.0.0.1".to_string(),
        side_channel_port: port,
    }
}

#[test]
fn loopback_dram_transfer() {
    if nixl_is_stub() {
        eprintln!("skipping NIXL loopback: built against stub bindings (no real libnixl)");
        return;
    }

    // Prefill: listens on 5600 and advertises KV. Decode: a distinct agent that pulls.
    let mut prefill = make_data_plane(PdRole::Prefill, cfg("mock-prefill", 5600));
    let mut decode = make_data_plane(PdRole::Decode, cfg("mock-decode", 5601));

    let kv = RequestKv {
        request_id: "req-loopback-1",
        num_tokens: 40,
    };

    let remote = prefill
        .advertise_prefilled(kv)
        .expect("prefill should register + advertise KV");
    assert_eq!(remote.engine_id, "mock-prefill");
    assert_eq!(remote.block_ids, vec![0, 1, 2]); // 40 tokens / 16 -> 3 blocks
    assert_eq!(remote.len, 3 * 4096);
    assert_ne!(remote.addr, 0);

    let bytes = decode
        .pull_prefilled(kv, &remote)
        .expect("decode should fetch md, READ over NIXL, and verify the pattern");
    assert_eq!(
        bytes,
        3 * 4096,
        "expected the full fabricated KV to transfer"
    );
}
