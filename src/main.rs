use anyhow::{Context as _, Result};
use clap::Parser as _;
use mock_engine_nixl::Opt;
use tokio_util::sync::CancellationToken;
use tracing::{Level, info};

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(Level::INFO.to_string())),
        )
        .init();
}

/// A cancellation token triggered by Ctrl-C.
fn shutdown_signal() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            info!("received Ctrl-C, shutting down");
            shutdown.cancel();
        }
    });
    token
}

fn main() -> Result<()> {
    init_tracing();
    let opt = Opt::parse();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime")?;

    runtime.block_on(async move {
        let shutdown = shutdown_signal();
        mock_engine_nixl::run(opt, shutdown).await
    })
}
