//! CLI for converting benchmark reports to the inference-sim trace format
//! and summarizing existing traces.

use std::fs;
use std::io::{self, BufReader, BufWriter};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use inference_simulator_rs::trace_convert::{
    ConvertOptions, convert_guidellm, summarize_trace, write_conversion, write_summary,
};

#[derive(Parser)]
#[command(
    name = "inference-sim-trace",
    about = "Convert benchmark reports to inference-sim trace format and summarize traces."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert a guidellm benchmark report (JSON) to an inference-sim trace (JSONL).
    FromGuidellm {
        /// Path to the guidellm report JSON file.
        input: PathBuf,
        /// Output JSONL trace file path.
        #[arg(short, long)]
        output: PathBuf,
        /// Override the model name in trace metadata.
        #[arg(long)]
        model: Option<String>,
        /// GPU identifier for trace metadata.
        #[arg(long)]
        gpu: Option<String>,
        /// Tensor-parallel degree for trace metadata.
        #[arg(long)]
        tp: Option<u32>,
    },
    /// Print summary statistics from an existing trace file.
    Summarize {
        /// Path to the JSONL trace file.
        input: PathBuf,
    },
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::FromGuidellm {
            input,
            output,
            model,
            gpu,
            tp,
        } => {
            let report_json = fs::read_to_string(&input)
                .with_context(|| format!("reading {}", input.display()))?;

            let opts = ConvertOptions { model, gpu, tp };
            let (meta, records) = convert_guidellm(&report_json, &opts)?;

            let file = fs::File::create(&output)
                .with_context(|| format!("creating {}", output.display()))?;
            let mut writer = BufWriter::new(file);
            write_conversion(&mut writer, &meta, &records)?;

            eprintln!("wrote {} records to {}", records.len(), output.display());
        }
        Command::Summarize { input } => {
            let file =
                fs::File::open(&input).with_context(|| format!("opening {}", input.display()))?;
            let reader = BufReader::new(file);
            let (meta, stats) = summarize_trace(reader)?;

            let stdout = io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            write_summary(&mut writer, &meta, &stats)?;
        }
    }

    Ok(())
}

fn main() -> ExitCode {
    // Initialize tracing for warnings emitted during conversion.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(io::stderr)
        .init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
