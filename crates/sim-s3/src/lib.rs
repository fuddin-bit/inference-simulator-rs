//! Remote trace I/O for vllm-vcr: a [`TraceUri`] is a local path, an `s3://`
//! object, or an `hf://` HuggingFace dataset, fetched via the AWS SDK or HF Hub API.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use aws_config::BehaviorVersion;
use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use tracing::{debug, info};
use url::Url;

/// Whether a raw path string is a remote URI (`s3://` or `hf://`) rather than a local path.
pub fn is_remote(uri: &str) -> bool {
    is_s3(uri) || is_hf(uri)
}

fn is_s3(uri: &str) -> bool {
    uri.len() >= 5 && uri[..5].eq_ignore_ascii_case("s3://")
}

fn is_hf(uri: &str) -> bool {
    uri.len() >= 5 && uri[..5].eq_ignore_ascii_case("hf://")
}

/// Sampling strategy for dataset conversion (CSV/Parquet/JSON → trace).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Pick random row for each request.
    Random,
    /// Iterate through rows sequentially.
    Sequential,
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        SamplingStrategy::Random
    }
}

/// A trace location, parsed (and validated) at the CLI boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceUri {
    Local(PathBuf),
    S3 {
        bucket: String,
        key: String,
    },
    HuggingFace {
        repo_id: String,
        filename: String,
        revision: Option<String>,
        sampling: SamplingStrategy,
    },
}

impl FromStr for TraceUri {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        if is_s3(s) {
            let (bucket, key) = parse_s3_uri(s).map_err(|e| format!("{e:#}"))?;
            Ok(TraceUri::S3 { bucket, key })
        } else if is_hf(s) {
            let (repo_id, filename, revision, sampling) =
                parse_hf_uri(s).map_err(|e| format!("{e:#}"))?;
            Ok(TraceUri::HuggingFace {
                repo_id,
                filename,
                revision,
                sampling,
            })
        } else {
            Ok(TraceUri::Local(PathBuf::from(s)))
        }
    }
}

impl fmt::Display for TraceUri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceUri::Local(path) => write!(f, "{}", path.display()),
            TraceUri::S3 { bucket, key } => write!(f, "s3://{bucket}/{key}"),
            TraceUri::HuggingFace {
                repo_id,
                filename,
                revision,
                sampling,
            } => {
                if let Some(rev) = revision {
                    write!(f, "hf://{repo_id}@{rev}/{filename}")?;
                } else {
                    write!(f, "hf://{repo_id}/{filename}")?;
                }
                if !matches!(sampling, SamplingStrategy::Random) {
                    write!(f, "?sampling=sequential")?;
                }
                Ok(())
            }
        }
    }
}

impl TraceUri {
    pub fn is_remote(&self) -> bool {
        matches!(self, TraceUri::S3 { .. } | TraceUri::HuggingFace { .. })
    }

    /// The local path, when this is a local target (`None` for remote).
    pub fn local_path(&self) -> Option<&Path> {
        match self {
            TraceUri::Local(path) => Some(path),
            TraceUri::S3 { .. } | TraceUri::HuggingFace { .. } => None,
        }
    }

    /// A local path holding this trace's bytes: the path itself when local, or a
    /// scratch file fetched from S3/HuggingFace.
    pub async fn materialize(&self, scratch_dir: &Path) -> Result<PathBuf> {
        match self {
            TraceUri::Local(path) => Ok(path.clone()),
            TraceUri::S3 { bucket, key } => self.fetch_s3(bucket, key, scratch_dir).await,
            TraceUri::HuggingFace {
                repo_id,
                filename,
                revision,
                sampling,
            } => {
                self.fetch_hf(repo_id, filename, revision.as_deref(), *sampling, scratch_dir)
                    .await
            }
        }
    }

    /// Where to write this trace locally before upload: its own path when local,
    /// else a scratch path under `scratch_dir`. HuggingFace URIs are read-only (no upload).
    pub fn write_path(&self, scratch_dir: &Path) -> PathBuf {
        match self {
            TraceUri::Local(path) => path.clone(),
            TraceUri::S3 { key, .. } => scratch_path(&self.to_string(), key, scratch_dir),
            TraceUri::HuggingFace { filename, .. } => {
                scratch_path(&self.to_string(), filename, scratch_dir)
            }
        }
    }

    /// Upload a finalized local file to this target; a no-op when local or HuggingFace (read-only).
    pub async fn upload(&self, local: &Path) -> Result<()> {
        let TraceUri::S3 { bucket, key } = self else {
            return Ok(());
        };
        let size = std::fs::metadata(local).map(|m| m.len()).ok();
        info!(local = %local.display(), uri = %self, bucket, key, bytes = size, "S3 PUT: uploading trace");
        let started = Instant::now();
        let body = ByteStream::from_path(local)
            .await
            .with_context(|| format!("opening {} for upload", local.display()))?;
        s3_client()
            .await
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(body)
            .send()
            .await
            .with_context(|| format!("S3 PUT {self}"))?;
        info!(uri = %self, bytes = size, elapsed_ms = started.elapsed().as_millis(), "S3 PUT: trace uploaded");
        Ok(())
    }

    async fn fetch_s3(&self, bucket: &str, key: &str, scratch_dir: &Path) -> Result<PathBuf> {
        let dest = scratch_path(&self.to_string(), key, scratch_dir);
        info!(uri = %self, bucket, key, dest = %dest.display(), "S3 GET: fetching trace to scratch");
        let started = Instant::now();
        let response = s3_client()
            .await
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .with_context(|| format!("S3 GET {self}"))?;
        let content_length = response.content_length();
        let bytes = response
            .body
            .collect()
            .await
            .with_context(|| format!("reading S3 object body: {self}"))?
            .into_bytes();
        std::fs::write(&dest, &bytes)
            .with_context(|| format!("writing scratch {} for {self}", dest.display()))?;
        info!(uri = %self, bytes = bytes.len(), content_length, dest = %dest.display(), elapsed_ms = started.elapsed().as_millis(), "S3 GET: trace materialized");
        Ok(dest)
    }

    async fn fetch_hf(
        &self,
        repo_id: &str,
        filename: &str,
        revision: Option<&str>,
        sampling: SamplingStrategy,
        scratch_dir: &Path,
    ) -> Result<PathBuf> {
        use hf_hub::api::tokio::Api;

        info!(
            uri = %self,
            repo_id,
            filename,
            revision,
            "HuggingFace: fetching dataset file"
        );
        let started = Instant::now();

        // Build HF API client (respects HF_TOKEN env var for private repos)
        let api = Api::new().with_context(|| "failed to initialize HuggingFace API client")?;

        // Create repo reference
        // Note: hf-hub 0.3 doesn't have a separate revision() method - revision is set during dataset() call
        let mut repo = api.dataset(repo_id.to_string());

        // If a specific revision is needed, we use the repo_with_revision method
        if let Some(rev) = revision {
            repo = api.repo(hf_hub::Repo::with_revision(
                repo_id.to_string(),
                hf_hub::RepoType::Dataset,
                rev.to_string(),
            ));
        }

        // Download file to HF cache (~/.cache/huggingface/hub/)
        let cached_path = repo
            .get(filename)
            .await
            .with_context(|| format!("HuggingFace GET {self}"))?;

        info!(
            uri = %self,
            cached = %cached_path.display(),
            elapsed_ms = started.elapsed().as_millis(),
            "HuggingFace: dataset file cached"
        );

        // Check if file needs conversion (CSV/JSON/Parquet → JSONL trace format)
        let needs_conversion = !cached_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.ends_with("jsonl") || e.ends_with("gz"))
            .unwrap_or(false);

        if needs_conversion {
            // Convert dataset → trace JSONL
            let converted = self
                .convert_to_trace(&cached_path, sampling, scratch_dir)
                .await?;
            Ok(converted)
        } else {
            // Already in trace format
            Ok(cached_path)
        }
    }

    async fn convert_to_trace(
        &self,
        input_path: &Path,
        sampling: SamplingStrategy,
        scratch_dir: &Path,
    ) -> Result<PathBuf> {
        use sha2::{Digest, Sha256};

        // Generate cache key from conversion parameters
        let mut hasher = Sha256::new();
        hasher.update(input_path.to_string_lossy().as_bytes());
        hasher.update(format!("{:?}", sampling).as_bytes());
        let hash = format!("{:x}", hasher.finalize());
        let cache_key = &hash[..16];

        let output_path = scratch_dir.join(format!("converted-{}.jsonl", cache_key));

        // Check if already converted
        if output_path.exists() {
            info!(
                cached = %output_path.display(),
                "Using cached converted trace"
            );
            return Ok(output_path);
        }

        info!(
            input = %input_path.display(),
            output = %output_path.display(),
            "Converting dataset to trace format"
        );

        // Delegate to conversion module in sim-trace
        // Note: vocab_size hardcoded to 32000 for now, could be made configurable
        let trace_sampling = match sampling {
            SamplingStrategy::Random => sim_trace::trace_convert::SamplingStrategy::Random,
            SamplingStrategy::Sequential => sim_trace::trace_convert::SamplingStrategy::Sequential,
        };
        sim_trace::trace_convert::convert_dataset_to_trace(
            input_path,
            &output_path,
            trace_sampling,
            32_000, // default vocab_size
        )?;

        Ok(output_path)
    }
}

fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    let url = Url::parse(uri).with_context(|| format!("parsing S3 URI: {uri}"))?;
    if url.scheme() != "s3" {
        bail!(
            "expected an s3:// URI, got scheme {:?}: {uri}",
            url.scheme()
        );
    }
    let bucket = url
        .host_str()
        .filter(|host| !host.is_empty())
        .with_context(|| format!("S3 URI has no bucket: {uri}"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    if key.is_empty() {
        bail!("S3 URI has no object key: {uri}");
    }
    Ok((bucket, key))
}

fn parse_hf_uri(uri: &str) -> Result<(String, String, Option<String>, SamplingStrategy)> {
    let url = Url::parse(uri).with_context(|| format!("parsing HF URI: {uri}"))?;

    if url.scheme() != "hf" {
        bail!("expected hf:// scheme, got: {}", url.scheme());
    }

    // URL treats hf://org/repo/file.txt as:
    //   - host: org
    //   - path: /repo/file.txt
    // So we need to combine host + path to get org/repo/file.txt
    let host = url
        .host_str()
        .filter(|h| !h.is_empty())
        .with_context(|| format!("HF URI has no host (org): {uri}"))?;

    let path = url.path().trim_start_matches('/');

    // Combine to get full path
    let full_path = if path.is_empty() {
        host.to_string()
    } else {
        format!("{}/{}", host, path)
    };

    // Split into components
    let parts: Vec<&str> = full_path.split('/').collect();
    if parts.len() < 3 {
        bail!("HF URI must have format hf://org/repo[@revision]/filename: {uri}");
    }

    // First part is org, second is repo (maybe with @revision), rest is filename
    let org = parts[0];
    let repo_with_maybe_rev = parts[1];
    let filename = parts[2..].join("/");

    if org.is_empty() || repo_with_maybe_rev.is_empty() || filename.is_empty() {
        bail!("HF URI has empty org, repo, or filename: {uri}");
    }

    // Check for @revision in repo part
    let (repo, revision) = match repo_with_maybe_rev.split_once('@') {
        Some((r, rev)) => (r, Some(rev.to_string())),
        None => (repo_with_maybe_rev, None),
    };

    let repo_id = format!("{}/{}", org, repo);

    // Parse query parameters
    let mut sampling = SamplingStrategy::default();
    for (key, value) in url.query_pairs() {
        if key == "sampling" {
            sampling = match value.as_ref() {
                "random" => SamplingStrategy::Random,
                "sequential" => SamplingStrategy::Sequential,
                _ => bail!("invalid sampling strategy: {} (expected 'random' or 'sequential')", value),
            };
        }
    }

    Ok((repo_id, filename, revision, sampling))
}

fn key_basename(key: &str) -> &str {
    key.rsplit('/')
        .find(|segment| !segment.is_empty())
        .unwrap_or("trace.jsonl")
}

/// Scratch path for a remote object: basename (keeping its suffix for gzip
/// detection) tagged with a hash of the URI so distinct objects don't collide.
fn scratch_path(uri: &str, key: &str, scratch_dir: &Path) -> PathBuf {
    use std::hash::{Hash as _, Hasher as _};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    uri.hash(&mut hasher);
    scratch_dir.join(format!(
        "sim-s3-{:016x}-{}",
        hasher.finish(),
        key_basename(key)
    ))
}

async fn s3_client() -> Client {
    let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
    // S3-compatible endpoints (MinIO/LocalStack) only serve path-style; real AWS
    // (no endpoint override) uses virtual-host style.
    let force_path_style = config.endpoint_url().is_some();
    debug!(
        region = config.region().map(|r| r.as_ref()),
        endpoint = config.endpoint_url(),
        force_path_style,
        "built S3 client from default credential chain"
    );
    Client::from_conf(
        aws_sdk_s3::config::Builder::from(&config)
            .force_path_style(force_path_style)
            .build(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_remote_only_matches_s3_scheme() {
        assert!(is_remote("s3://bucket/key"));
        assert!(is_remote("S3://Bucket/Key"));
        assert!(!is_remote("/tmp/trace.jsonl.gz"));
        assert!(!is_remote("trace.jsonl"));
        assert!(!is_remote("file:///tmp/trace.jsonl"));
        assert!(!is_remote(""));
        assert!(!is_remote("s3:"));
    }

    #[test]
    fn parses_s3_uri_into_typed_variant() {
        let uri: TraceUri = "s3://my-bucket/traces/abc/tap-trace.jsonl.gz"
            .parse()
            .unwrap();
        assert_eq!(
            uri,
            TraceUri::S3 {
                bucket: "my-bucket".to_string(),
                key: "traces/abc/tap-trace.jsonl.gz".to_string(),
            }
        );
        assert!(uri.is_remote());
        assert!(uri.local_path().is_none());
        assert_eq!(
            uri.to_string(),
            "s3://my-bucket/traces/abc/tap-trace.jsonl.gz"
        );
    }

    #[test]
    fn parses_bare_path_as_local() {
        let uri: TraceUri = "/tmp/trace.jsonl".parse().unwrap();
        assert_eq!(uri, TraceUri::Local(PathBuf::from("/tmp/trace.jsonl")));
        assert!(!uri.is_remote());
        assert_eq!(uri.local_path(), Some(Path::new("/tmp/trace.jsonl")));
    }

    #[test]
    fn rejects_malformed_s3_uri() {
        assert!("s3://bucket".parse::<TraceUri>().is_err()); // no key
        assert!("s3://bucket/".parse::<TraceUri>().is_err()); // empty key
        assert!("s3:///key".parse::<TraceUri>().is_err()); // no bucket
    }

    #[test]
    fn key_basename_keeps_gz_suffix() {
        assert_eq!(
            key_basename("traces/abc/tap-trace.jsonl.gz"),
            "tap-trace.jsonl.gz"
        );
        assert_eq!(key_basename("flat.jsonl"), "flat.jsonl");
        assert_eq!(key_basename("trailing/"), "trailing");
    }

    #[test]
    fn write_path_is_stable_per_uri_and_collision_free() {
        let dir = Path::new("/tmp/scratch");
        let a1: TraceUri = "s3://b/traces/a/tap-trace.jsonl.gz".parse().unwrap();
        let a2: TraceUri = "s3://b/traces/a/tap-trace.jsonl.gz".parse().unwrap();
        let b: TraceUri = "s3://b/traces/b/tap-trace.jsonl.gz".parse().unwrap();

        assert_eq!(a1.write_path(dir), a2.write_path(dir));
        assert_ne!(a1.write_path(dir), b.write_path(dir));
        assert!(
            a1.write_path(dir)
                .to_string_lossy()
                .ends_with("-tap-trace.jsonl.gz")
        );

        let local: TraceUri = "/tmp/x.jsonl".parse().unwrap();
        assert_eq!(local.write_path(dir), PathBuf::from("/tmp/x.jsonl"));
    }

    #[test]
    fn parses_hf_uri_basic() {
        let uri: TraceUri = "hf://neuralmagic/vllm-traces/trace.jsonl.gz"
            .parse()
            .unwrap();
        assert!(uri.is_remote());
        assert!(uri.local_path().is_none());
        match uri {
            TraceUri::HuggingFace {
                repo_id,
                filename,
                revision,
                sampling,
            } => {
                assert_eq!(repo_id, "neuralmagic/vllm-traces");
                assert_eq!(filename, "trace.jsonl.gz");
                assert_eq!(revision, None);
                assert_eq!(sampling, SamplingStrategy::Random);
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn parses_hf_uri_with_revision() {
        let uri: TraceUri = "hf://neuralmagic/vllm-traces@v1.2/trace.jsonl.gz"
            .parse()
            .unwrap();
        match &uri {
            TraceUri::HuggingFace {
                repo_id,
                filename,
                revision,
                ..
            } => {
                assert_eq!(repo_id, "neuralmagic/vllm-traces");
                assert_eq!(filename, "trace.jsonl.gz");
                assert_eq!(revision, &Some("v1.2".to_string()));
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn parses_hf_uri_with_query_params() {
        let uri: TraceUri = "hf://org/repo/file.csv?sampling=sequential"
            .parse()
            .unwrap();
        match &uri {
            TraceUri::HuggingFace { sampling, .. } => {
                assert_eq!(*sampling, SamplingStrategy::Sequential);
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn parses_hf_uri_nested_path() {
        let uri: TraceUri = "hf://org/repo/traces/subdir/file.jsonl.gz"
            .parse()
            .unwrap();
        match &uri {
            TraceUri::HuggingFace {
                repo_id, filename, ..
            } => {
                assert_eq!(repo_id, "org/repo");
                assert_eq!(filename, "traces/subdir/file.jsonl.gz");
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn rejects_malformed_hf_uri() {
        assert!("hf://no-filename".parse::<TraceUri>().is_err());
        assert!("hf:///no-repo/file".parse::<TraceUri>().is_err());
        assert!("hf://repo".parse::<TraceUri>().is_err()); // no filename
    }

    #[test]
    fn hf_uri_display_roundtrip() {
        let uri: TraceUri = "hf://org/repo@rev/file.jsonl.gz".parse().unwrap();
        assert_eq!(uri.to_string(), "hf://org/repo@rev/file.jsonl.gz");

        let uri2: TraceUri = "hf://org/repo/file.jsonl.gz".parse().unwrap();
        assert_eq!(uri2.to_string(), "hf://org/repo/file.jsonl.gz");

        let uri3: TraceUri = "hf://org/repo/file.csv?sampling=sequential"
            .parse()
            .unwrap();
        assert_eq!(uri3.to_string(), "hf://org/repo/file.csv?sampling=sequential");
    }
}
