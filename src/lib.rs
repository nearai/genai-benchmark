//! GenAI Benchmark - Load testing library for OpenAI-compatible LLM APIs

use anyhow::{anyhow, Context, Result};
use async_channel::{bounded, Receiver, Sender};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

fn build_http_client(concurrency: usize) -> Result<Client> {
    Client::builder()
        .pool_max_idle_per_host(concurrency * 2)
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_nodelay(true)
        .tcp_keepalive(Duration::from_secs(60))
        .build()
        .map_err(|e| anyhow!("Failed to build HTTP client: {}", e))
}

// ============================================================================
// Public Types
// ============================================================================

/// Configuration for a single benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Name/identifier for this benchmark
    #[serde(default)]
    pub name: Option<String>,
    /// Base URL of the OpenAI-compatible API
    pub base_url: String,
    /// API key for authentication
    #[serde(default)]
    pub api_key: String,
    /// Model name to use
    pub model: String,
    /// Maximum tokens to generate per request
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Maximum concurrent requests
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Target requests per second (0 = unlimited)
    #[serde(default = "default_rps")]
    pub rps: f64,
    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Disable connection pre-warming
    #[serde(default)]
    pub disable_prewarm: bool,
}

fn default_max_tokens() -> u32 {
    256
}
fn default_concurrency() -> usize {
    5
}
fn default_rps() -> f64 {
    100.0
}
fn default_timeout() -> u64 {
    300
}

impl BenchmarkConfig {
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_secs)
    }
}

/// A provider configuration for multi-provider scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Name of the provider
    pub name: String,
    /// Base URL of the API
    pub base_url: String,
    /// API key
    #[serde(default)]
    pub api_key: String,
    /// Optional model override (if different from scenario's model)
    #[serde(default)]
    pub model: Option<String>,
}

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DatasetConfig {
    /// ShareGPT format dataset
    Sharegpt {
        /// Path or URL to the dataset
        #[serde(default)]
        path: Option<String>,
        /// Hugging Face dataset name
        #[serde(default = "default_hf_dataset")]
        hf_dataset: String,
        /// Skip first N entries
        #[serde(default)]
        skip: usize,
    },
    /// Simple prompts file (one per line)
    Prompts {
        /// Path to the prompts file
        path: String,
        /// Skip first N entries
        #[serde(default)]
        skip: usize,
    },
    /// Synthetic generated prompts
    Synthetic {
        /// Random seed
        #[serde(default)]
        seed: Option<u64>,
    },
}

fn default_hf_dataset() -> String {
    "anon8231489123/ShareGPT_Vicuna_unfiltered".to_string()
}

impl Default for DatasetConfig {
    fn default() -> Self {
        DatasetConfig::Synthetic { seed: None }
    }
}

/// A complete scenario configuration for YAML-based testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    /// Name of the scenario
    pub name: String,
    /// Description
    #[serde(default)]
    pub description: Option<String>,
    /// Model to test
    pub model: String,
    /// List of providers to test against
    pub providers: Vec<ProviderConfig>,
    /// Dataset configuration
    #[serde(default)]
    pub dataset: DatasetConfig,
    /// Number of requests per provider
    #[serde(default = "default_num_requests")]
    pub num_requests: usize,
    /// Maximum concurrent requests
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Target requests per second
    #[serde(default = "default_rps")]
    pub rps: f64,
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_num_requests() -> usize {
    100
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Metrics for a single request
#[derive(Debug, Clone)]
pub struct RequestMetrics {
    pub success: bool,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub ttft_ms: f64,
    pub total_time_ms: f64,
    pub inter_chunk_latencies: Vec<f64>, // Time between SSE chunks
    pub chunk_count: u32,
    pub output_chars: usize,
    pub got_usage: bool,
}

impl RequestMetrics {
    /// Time per output token (excluding first token)
    /// Note: This is approximate since we measure inter-chunk, not inter-token
    pub fn tpot_ms(&self) -> Option<f64> {
        if self.output_tokens <= 1 || self.inter_chunk_latencies.is_empty() {
            return None;
        }
        let total_after_first: f64 = self.inter_chunk_latencies.iter().sum();
        Some(total_after_first / (self.output_tokens - 1) as f64)
    }
}

/// Results from a benchmark run
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    /// Provider/benchmark name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub total_requests: usize,
    pub requests_with_usage: usize,
    pub concurrency: usize,
    pub rps_configured: f64,
    pub duration_secs: f64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_chunks: u64,
    #[serde(skip)]
    pub ttft_values: Vec<f64>,
    #[serde(skip)]
    pub tpot_values: Vec<f64>,
    #[serde(skip)]
    pub itl_values: Vec<f64>,
    #[serde(skip)]
    pub request_duration_values: Vec<f64>,
    #[serde(skip)]
    pub tokens_per_request: Vec<u32>,
}

impl BenchmarkResult {
    pub fn request_throughput(&self) -> f64 {
        if self.duration_secs == 0.0 {
            return 0.0;
        }
        self.successful_requests as f64 / self.duration_secs
    }

    pub fn output_token_throughput(&self) -> f64 {
        if self.duration_secs == 0.0 {
            return 0.0;
        }
        self.total_output_tokens as f64 / self.duration_secs
    }

    pub fn total_token_throughput(&self) -> f64 {
        if self.duration_secs == 0.0 {
            return 0.0;
        }
        (self.total_input_tokens + self.total_output_tokens) as f64 / self.duration_secs
    }

    pub fn chunks_per_sec(&self) -> f64 {
        if self.duration_secs == 0.0 {
            return 0.0;
        }
        self.total_chunks as f64 / self.duration_secs
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.successful_requests as f64 / self.total_requests as f64 * 100.0
    }

    pub fn avg_tokens_per_request(&self) -> f64 {
        if self.tokens_per_request.is_empty() {
            return 0.0;
        }
        self.tokens_per_request
            .iter()
            .map(|&x| x as f64)
            .sum::<f64>()
            / self.tokens_per_request.len() as f64
    }

    pub fn avg_request_duration(&self) -> f64 {
        mean(&self.request_duration_values)
    }

    /// Tokens per chunk ratio (1.0 = one token per SSE chunk, ideal for accurate ITL)
    pub fn tokens_per_chunk(&self) -> f64 {
        if self.total_chunks == 0 {
            return 0.0;
        }
        self.total_output_tokens as f64 / self.total_chunks as f64
    }
}

// ============================================================================
// Internal Types
// ============================================================================

#[derive(Debug, Clone, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    // Request usage stats in final chunk (OpenAI/Bedrock style)
    // NEAR AI includes usage by default, no flag needed
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ShareGPTConversation {
    #[serde(alias = "conversation", alias = "conversations")]
    conversations: Vec<ShareGPTMessage>,
}

#[derive(Debug, Deserialize)]
struct ShareGPTMessage {
    from: String,
    value: String,
}

// ============================================================================
// Statistics
// ============================================================================

pub fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted_values.len() - 1) as f64).round() as usize;
    sorted_values[idx.min(sorted_values.len() - 1)]
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn median(sorted_values: &[f64]) -> f64 {
    percentile(sorted_values, 50.0)
}

// ============================================================================
// Dataset Loading
// ============================================================================

pub async fn load_sharegpt_dataset(
    path: &str,
    skip: usize,
    limit: usize,
) -> Result<Vec<Vec<Message>>> {
    info!("Loading ShareGPT dataset from: {}", path);

    let content = if path.starts_with("http://") || path.starts_with("https://") {
        let client = Client::new();
        client.get(path).send().await?.text().await?
    } else {
        tokio::fs::read_to_string(path).await?
    };

    let conversations: Vec<ShareGPTConversation> = if path.ends_with(".jsonl") {
        content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    } else {
        serde_json::from_str(&content)?
    };

    let prompts: Vec<Vec<Message>> = conversations
        .into_iter()
        .skip(skip)
        .take(limit)
        .filter_map(|conv| {
            let messages: Vec<Message> = conv
                .conversations
                .iter()
                .map(|m| {
                    let role = match m.from.to_lowercase().as_str() {
                        "human" | "user" => "user",
                        "gpt" | "assistant" | "chatgpt" => "assistant",
                        "system" => "system",
                        _ => "user",
                    };
                    Message {
                        role: role.to_string(),
                        content: m.value.clone(),
                    }
                })
                .collect();

            if messages.iter().any(|m| m.role == "user") {
                let mut prompt_messages = Vec::new();
                for msg in messages {
                    prompt_messages.push(msg.clone());
                    if msg.role == "user" {
                        break;
                    }
                }
                Some(prompt_messages)
            } else {
                None
            }
        })
        .collect();

    info!("Loaded {} prompts from dataset", prompts.len());
    Ok(prompts)
}

pub async fn load_prompts_file(path: &str, skip: usize, limit: usize) -> Result<Vec<Vec<Message>>> {
    info!("Loading prompts from: {}", path);
    let content = tokio::fs::read_to_string(path).await?;

    let prompts: Vec<Vec<Message>> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .skip(skip)
        .take(limit)
        .map(|line| {
            vec![Message {
                role: "user".to_string(),
                content: line.to_string(),
            }]
        })
        .collect();

    info!("Loaded {} prompts", prompts.len());
    Ok(prompts)
}

pub fn generate_synthetic_prompts(count: usize, seed: Option<u64>) -> Vec<Vec<Message>> {
    use rand::{Rng, SeedableRng};

    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

    let topics = [
        "Explain quantum computing to a beginner.",
        "Write a short story about a robot learning to paint.",
        "What are the main differences between Python and Rust?",
        "Describe the process of photosynthesis in detail.",
        "How does machine learning work?",
        "Write a poem about the ocean.",
        "Explain the theory of relativity.",
        "What are the benefits of meditation?",
        "Describe the history of the internet.",
        "How do neural networks learn?",
    ];

    (0..count)
        .map(|_| {
            let topic = topics[rng.gen_range(0..topics.len())];
            vec![Message {
                role: "user".to_string(),
                content: topic.to_string(),
            }]
        })
        .collect()
}

pub async fn download_hf_dataset(dataset_name: &str, _split: &str) -> Result<String> {
    let url = if dataset_name.contains("ShareGPT") {
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    } else {
        return Err(anyhow!(
            "Unknown dataset: {}. Please provide a direct URL or local path.",
            dataset_name
        ));
    };

    // Check if cached file exists
    let cache_path = format!(
        "/tmp/genai_benchmark_{}.json",
        dataset_name.replace("/", "_")
    );

    if tokio::fs::metadata(&cache_path).await.is_ok() {
        info!("Using cached dataset: {}", cache_path);
        return Ok(cache_path);
    }

    info!("Downloading dataset from Hugging Face: {}", url);

    let client = Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;

    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to download dataset")?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Failed to download dataset: HTTP {}",
            response.status()
        ));
    }

    let content = response.text().await?;

    tokio::fs::write(&cache_path, &content).await?;
    info!("Dataset cached at: {}", cache_path);

    Ok(cache_path)
}

/// Load prompts based on dataset configuration
pub async fn load_dataset(
    config: &DatasetConfig,
    num_requests: usize,
) -> Result<Vec<Vec<Message>>> {
    match config {
        DatasetConfig::Sharegpt {
            path,
            hf_dataset,
            skip,
        } => {
            let dataset_path = match path {
                Some(p) => p.clone(),
                None => download_hf_dataset(hf_dataset, "train").await?,
            };
            load_sharegpt_dataset(&dataset_path, *skip, num_requests * 2).await
        }
        DatasetConfig::Prompts { path, skip } => {
            load_prompts_file(path, *skip, num_requests * 2).await
        }
        DatasetConfig::Synthetic { seed } => Ok(generate_synthetic_prompts(num_requests, *seed)),
    }
}

// ============================================================================
// API Client
// ============================================================================

async fn send_streaming_request(
    client: &Client,
    base_url: &str,
    api_key: &str,
    request: ChatCompletionRequest,
    timeout: Duration,
) -> Result<RequestMetrics> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let start_time = Instant::now();

    let mut req_builder = client
        .post(&url)
        .header("Content-Type", "application/json")
        .timeout(timeout);

    if !api_key.is_empty() {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
    }

    let req_builder = req_builder.json(&request);

    let mut es = EventSource::new(req_builder)?;

    let mut first_token_time: Option<Instant> = None;
    let mut last_chunk_time = start_time;
    let mut output_tokens: u32 = 0;
    let mut output_chars: usize = 0;
    let mut inter_chunk_latencies: Vec<f64> = Vec::new();
    let mut input_tokens: u32 = 0;
    let mut chunk_count: u32 = 0;
    let mut got_usage: bool = false;

    while let Some(event) = es.next().await {
        match event {
            Ok(Event::Open) => {
                debug!("SSE connection opened");
            }
            Ok(Event::Message(msg)) => {
                let now = Instant::now();
                chunk_count += 1;

                if msg.data == "[DONE]" {
                    break;
                }

                match serde_json::from_str::<ChatCompletionChunk>(&msg.data) {
                    Ok(chunk) => {
                        // Check for usage stats first (may come in final chunk with empty choices)
                        if let Some(usage) = &chunk.usage {
                            input_tokens = usage.prompt_tokens;
                            output_tokens = usage.completion_tokens;
                            got_usage = true;
                            debug!(
                                "Got usage stats: input={}, output={}",
                                input_tokens, output_tokens
                            );
                        }

                        // Process content from choices
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                if !content.is_empty() {
                                    output_chars += content.len();

                                    if first_token_time.is_none() {
                                        first_token_time = Some(now);
                                    } else {
                                        let icl = now.duration_since(last_chunk_time).as_secs_f64()
                                            * 1000.0;
                                        inter_chunk_latencies.push(icl);
                                    }
                                    last_chunk_time = now;

                                    // Count chunks with content as fallback
                                    // (will be overridden by usage if available)
                                    if !got_usage {
                                        output_tokens += 1;
                                    }
                                }
                            }

                            // Only break on finish_reason if we already got usage
                            // Some APIs send usage in a separate final chunk
                            if choice.finish_reason.is_some() && got_usage {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        debug!("Failed to parse chunk: {} - {}", e, msg.data);
                    }
                }
            }
            Err(e) => {
                es.close();
                return Err(anyhow!("SSE error: {:?}", e));
            }
        }
    }

    // Log warning if we didn't get usage stats from the API
    if !got_usage {
        warn!(
            "No usage stats received from API for model '{}'. \
             Token count ({}) is based on chunk count, not actual tokens. \
             Ensure the API supports 'stream_options.include_usage'.",
            request.model, output_tokens
        );
    }

    let total_time = start_time.elapsed();

    let ttft = first_token_time
        .map(|t| t.duration_since(start_time).as_secs_f64() * 1000.0)
        .unwrap_or(total_time.as_secs_f64() * 1000.0);

    Ok(RequestMetrics {
        success: true,
        input_tokens,
        output_tokens,
        output_chars,
        chunk_count,
        got_usage,
        ttft_ms: ttft,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        inter_chunk_latencies,
    })
}

// ============================================================================
// Benchmark Runner
// ============================================================================

async fn prewarm_connections(client: &Client, base_url: &str, api_key: &str, count: usize) {
    info!("Pre-warming {} connections to {}...", count, base_url);
    let url = format!("{}/models", base_url.trim_end_matches('/'));

    let mut handles = Vec::new();
    for _ in 0..count {
        let client = client.clone();
        let url = url.clone();
        let api_key = api_key.to_string();

        handles.push(tokio::spawn(async move {
            let mut req = client.get(&url);
            if !api_key.is_empty() {
                req = req.header("Authorization", format!("Bearer {}", api_key));
            }
            let _ = req.send().await;
        }));
    }

    for handle in handles {
        let _ = handle.await;
    }
    info!("Connection pre-warming complete");
}

/// Run a benchmark with the given configuration
pub async fn run_benchmark(
    config: &BenchmarkConfig,
    prompts: Vec<Vec<Message>>,
    num_requests: usize,
) -> Result<BenchmarkResult> {
    let client = build_http_client(config.concurrency)?;

    if !config.disable_prewarm {
        prewarm_connections(
            &client,
            &config.base_url,
            &config.api_key,
            config.concurrency.min(10),
        )
        .await;
    }

    let semaphore = Arc::new(Semaphore::new(config.concurrency));
    let (tx, rx): (Sender<RequestMetrics>, Receiver<RequestMetrics>) = bounded(num_requests);

    let progress = ProgressBar::new(num_requests as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})",
            )?
            .progress_chars("#>-"),
    );

    let active_requests = Arc::new(AtomicUsize::new(0));
    let start_time = Instant::now();

    let request_delay = if config.rps > 0.0 {
        Duration::from_secs_f64(1.0 / config.rps)
    } else {
        Duration::ZERO
    };

    let mut handles = Vec::new();

    for i in 0..num_requests {
        let permit = semaphore.clone().acquire_owned().await?;
        let client = client.clone();
        let tx = tx.clone();
        let progress = progress.clone();
        let active = active_requests.clone();

        let messages = prompts[i % prompts.len()].clone();
        let config_base_url = config.base_url.clone();
        let config_api_key = config.api_key.clone();
        let config_model = config.model.clone();
        let config_max_tokens = config.max_tokens;
        let config_timeout = config.timeout();

        active.fetch_add(1, Ordering::SeqCst);

        let handle = tokio::spawn(async move {
            let request = ChatCompletionRequest {
                model: config_model,
                messages,
                max_tokens: config_max_tokens,
                stream: true,
                temperature: Some(0.7),
                // Request usage stats (NEAR AI includes by default, Bedrock needs this)
                stream_options: Some(StreamOptions {
                    include_usage: true,
                }),
            };

            let result = send_streaming_request(
                &client,
                &config_base_url,
                &config_api_key,
                request,
                config_timeout,
            )
            .await;

            active.fetch_sub(1, Ordering::SeqCst);
            progress.inc(1);
            drop(permit);

            match result {
                Ok(metrics) => {
                    let _ = tx.send(metrics).await;
                }
                Err(e) => {
                    error!("Request failed: {}", e);
                    let _ = tx
                        .send(RequestMetrics {
                            success: false,
                            input_tokens: 0,
                            output_tokens: 0,
                            output_chars: 0,
                            chunk_count: 0,
                            got_usage: false,
                            ttft_ms: 0.0,
                            total_time_ms: 0.0,
                            inter_chunk_latencies: Vec::new(),
                        })
                        .await;
                }
            }
        });

        handles.push(handle);

        if request_delay > Duration::ZERO && i < num_requests - 1 {
            sleep(request_delay).await;
        }
    }

    drop(tx);

    for handle in handles {
        let _ = handle.await;
    }

    progress.finish_with_message("Benchmark complete");

    let duration = start_time.elapsed();

    let mut successful = 0;
    let mut failed = 0;
    let mut requests_with_usage = 0;
    let mut total_input_tokens: u64 = 0;
    let mut total_output_tokens: u64 = 0;
    let mut total_chunks: u64 = 0;
    let mut ttft_values: Vec<f64> = Vec::new();
    let mut tpot_values: Vec<f64> = Vec::new();
    let mut itl_values: Vec<f64> = Vec::new();
    let mut request_duration_values: Vec<f64> = Vec::new();
    let mut tokens_per_request: Vec<u32> = Vec::new();

    while let Ok(metrics) = rx.recv().await {
        if metrics.success {
            successful += 1;
            if metrics.got_usage {
                requests_with_usage += 1;
            }
            total_input_tokens += metrics.input_tokens as u64;
            total_output_tokens += metrics.output_tokens as u64;
            total_chunks += metrics.chunk_count as u64;
            ttft_values.push(metrics.ttft_ms);
            request_duration_values.push(metrics.total_time_ms);
            tokens_per_request.push(metrics.output_tokens);

            if let Some(tpot) = metrics.tpot_ms() {
                tpot_values.push(tpot);
            }

            itl_values.extend(metrics.inter_chunk_latencies);
        } else {
            failed += 1;
        }
    }

    ttft_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tpot_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    itl_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    request_duration_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Log summary warning if many requests didn't get usage stats
    if requests_with_usage < successful {
        let missing = successful - requests_with_usage;
        warn!(
            "{}/{} requests did not return usage stats. Token counts may be inaccurate (based on chunk count).",
            missing, successful
        );
    }

    Ok(BenchmarkResult {
        name: config.name.clone(),
        successful_requests: successful,
        failed_requests: failed,
        total_requests: num_requests,
        requests_with_usage,
        concurrency: config.concurrency,
        rps_configured: config.rps,
        duration_secs: duration.as_secs_f64(),
        total_input_tokens,
        total_output_tokens,
        total_chunks,
        ttft_values,
        tpot_values,
        itl_values,
        request_duration_values,
        tokens_per_request,
    })
}

/// Run a complete scenario (multiple providers)
pub async fn run_scenario(scenario: &Scenario) -> Result<Vec<BenchmarkResult>> {
    info!("Running scenario: {}", scenario.name);
    if let Some(desc) = &scenario.description {
        info!("Description: {}", desc);
    }

    // Load dataset once for all providers
    let prompts = load_dataset(&scenario.dataset, scenario.num_requests).await?;

    if prompts.is_empty() {
        return Err(anyhow!("No prompts loaded from dataset"));
    }

    info!("Loaded {} prompts", prompts.len());

    let mut results = Vec::new();

    for provider in &scenario.providers {
        // Use provider-specific model if set, otherwise use scenario's model
        let model = provider
            .model
            .clone()
            .unwrap_or_else(|| scenario.model.clone());

        info!(
            "Testing provider: {} ({}) with model: {}",
            provider.name, provider.base_url, model
        );

        let config = BenchmarkConfig {
            name: Some(provider.name.clone()),
            base_url: provider.base_url.clone(),
            api_key: provider.api_key.clone(),
            model,
            max_tokens: scenario.max_tokens,
            concurrency: scenario.concurrency,
            rps: scenario.rps,
            timeout_secs: scenario.timeout_secs,
            disable_prewarm: false,
        };

        let result = run_benchmark(&config, prompts.clone(), scenario.num_requests).await?;
        results.push(result);
    }

    Ok(results)
}

// ============================================================================
// Results Display
// ============================================================================

pub fn print_result(result: &BenchmarkResult) {
    println!();
    if let Some(name) = &result.name {
        println!("============ {} ============", name);
    }
    println!("============ Serving Benchmark Result ============");
    println!(
        "Successful requests:                     {}",
        result.successful_requests
    );
    println!(
        "Failed requests:                         {}",
        result.failed_requests
    );

    // Show usage stats availability
    if result.requests_with_usage < result.successful_requests {
        println!(
            "Requests with usage stats:               {}/{} ⚠️",
            result.requests_with_usage, result.successful_requests
        );
    } else {
        println!(
            "Requests with usage stats:               {}/{}",
            result.requests_with_usage, result.successful_requests
        );
    }

    println!(
        "Maximum request concurrency:             {}",
        result.concurrency
    );
    println!(
        "Request rate configured (RPS):           {:.2}",
        result.rps_configured
    );
    println!(
        "Benchmark duration (s):                  {:.2}",
        result.duration_secs
    );
    println!(
        "Total input tokens:                      {}",
        result.total_input_tokens
    );
    println!(
        "Total generated tokens:                  {}",
        result.total_output_tokens
    );
    println!(
        "Total chunks received:                   {}",
        result.total_chunks
    );
    println!(
        "Avg tokens per request:                  {:.1}",
        result.avg_tokens_per_request()
    );

    // Tokens per chunk ratio - indicates streaming granularity
    let tpc = result.tokens_per_chunk();
    if tpc > 0.0 {
        let tpc_note = if tpc > 1.5 {
            " (batched)"
        } else if tpc < 0.8 {
            " (fragmented)"
        } else {
            ""
        };
        println!(
            "Tokens per chunk:                        {:.2}{}",
            tpc, tpc_note
        );
    }

    println!(
        "Request throughput (req/s):              {:.2}",
        result.request_throughput()
    );
    println!(
        "Output token throughput (tok/s):         {:.2}",
        result.output_token_throughput()
    );
    println!(
        "Total Token throughput (tok/s):          {:.2}",
        result.total_token_throughput()
    );
    println!(
        "Chunks throughput (chunks/s):            {:.2}",
        result.chunks_per_sec()
    );

    if !result.ttft_values.is_empty() {
        println!("---------------Time to First Token----------------");
        println!(
            "Mean TTFT (ms):                          {:.2}",
            mean(&result.ttft_values)
        );
        println!(
            "Median TTFT (ms):                        {:.2}",
            median(&result.ttft_values)
        );
        println!(
            "P90 TTFT (ms):                           {:.2}",
            percentile(&result.ttft_values, 90.0)
        );
        println!(
            "P95 TTFT (ms):                           {:.2}",
            percentile(&result.ttft_values, 95.0)
        );
        println!(
            "P99 TTFT (ms):                           {:.2}",
            percentile(&result.ttft_values, 99.0)
        );
        println!(
            "P100 TTFT (ms):                          {:.2}",
            percentile(&result.ttft_values, 100.0)
        );
    }

    if !result.tpot_values.is_empty() {
        println!("-----Time per Output Token (excl. 1st token)------");
        println!(
            "Mean TPOT (ms):                          {:.2}",
            mean(&result.tpot_values)
        );
        println!(
            "Median TPOT (ms):                        {:.2}",
            median(&result.tpot_values)
        );
        println!(
            "P90 TPOT (ms):                           {:.2}",
            percentile(&result.tpot_values, 90.0)
        );
        println!(
            "P95 TPOT (ms):                           {:.2}",
            percentile(&result.tpot_values, 95.0)
        );
        println!(
            "P99 TPOT (ms):                           {:.2}",
            percentile(&result.tpot_values, 99.0)
        );
        println!(
            "P100 TPOT (ms):                          {:.2}",
            percentile(&result.tpot_values, 100.0)
        );
    }

    if !result.itl_values.is_empty() {
        println!("-----------Inter-Chunk Latency (ICL)--------------");
        println!("(Time between SSE chunks, not individual tokens)");
        println!(
            "Mean ICL (ms):                           {:.2}",
            mean(&result.itl_values)
        );
        println!(
            "Median ICL (ms):                         {:.2}",
            median(&result.itl_values)
        );
        println!(
            "P90 ICL (ms):                            {:.2}",
            percentile(&result.itl_values, 90.0)
        );
        println!(
            "P95 ICL (ms):                            {:.2}",
            percentile(&result.itl_values, 95.0)
        );
        println!(
            "P99 ICL (ms):                            {:.2}",
            percentile(&result.itl_values, 99.0)
        );
        println!(
            "P100 ICL (ms):                           {:.2}",
            percentile(&result.itl_values, 100.0)
        );
    }

    if !result.request_duration_values.is_empty() {
        println!("--------------Request Duration--------------------");
        println!(
            "Mean Duration (ms):                      {:.2}",
            mean(&result.request_duration_values)
        );
        println!(
            "Median Duration (ms):                    {:.2}",
            median(&result.request_duration_values)
        );
        println!(
            "P90 Duration (ms):                       {:.2}",
            percentile(&result.request_duration_values, 90.0)
        );
        println!(
            "P95 Duration (ms):                       {:.2}",
            percentile(&result.request_duration_values, 95.0)
        );
        println!(
            "P99 Duration (ms):                       {:.2}",
            percentile(&result.request_duration_values, 99.0)
        );
        println!(
            "P100 Duration (ms):                      {:.2}",
            percentile(&result.request_duration_values, 100.0)
        );
    }

    println!("==================================================");
}

pub fn print_comparison(results: &[BenchmarkResult]) {
    if results.len() < 2 {
        if results.len() == 1 {
            println!("\n(Need at least 2 providers to show comparison)");
        }
        return;
    }

    // Collect provider names
    let names: Vec<String> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            r.name
                .clone()
                .unwrap_or_else(|| format!("Provider {}", i + 1))
        })
        .collect();

    let separator = "=".repeat(100);

    println!("\n{}", separator);
    println!("Comparison: {}", names.join(" vs "));
    println!("{}", separator);

    // Helper to find winner (lowest value for latency metrics, highest for throughput)
    let find_winner = |values: &[f64], lower_is_better: bool| -> Option<usize> {
        if values.iter().all(|&v| v == 0.0) {
            return None;
        }
        if lower_is_better {
            values
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 0.0)
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
        } else {
            values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
        }
    };

    // Build dynamic header
    let col_width = 12;
    let metric_width = 28;

    // Summary Metrics
    println!("\n## Summary Metrics\n");

    // Header row
    print!("{:<width$}", "Metric", width = metric_width);
    for name in &names {
        print!(" | {:<width$}", name, width = col_width);
    }
    println!(" | Winner");

    // Separator
    let total_width = metric_width + (col_width + 3) * names.len() + 10;
    println!("{}", "-".repeat(total_width));

    // Macro to print a comparison row
    macro_rules! print_row {
        ($label:expr, $values:expr, $format_fn:expr, $lower_is_better:expr) => {{
            let values: Vec<f64> = $values;
            let winner_idx = find_winner(&values, $lower_is_better);
            print!("{:<width$}", $label, width = metric_width);
            for v in &values {
                print!(" | {:<width$}", $format_fn(*v), width = col_width);
            }
            match winner_idx {
                Some(idx) => println!(" | {}", names[idx]),
                None => println!(" | -"),
            }
        }};
    }

    // Test Duration (no winner)
    print!("{:<width$}", "Test Duration", width = metric_width);
    for r in results {
        print!(
            " | {:<width$}",
            format!("{:.2}s", r.duration_secs),
            width = col_width
        );
    }
    println!(" | -");

    // Total Requests (no winner if equal)
    let req_values: Vec<f64> = results.iter().map(|r| r.total_requests as f64).collect();
    let all_equal = req_values.windows(2).all(|w| w[0] == w[1]);
    print!("{:<width$}", "Total Requests", width = metric_width);
    for r in results {
        print!(" | {:<width$}", r.total_requests, width = col_width);
    }
    if all_equal {
        println!(" | -");
    } else {
        let winner = find_winner(&req_values, false);
        match winner {
            Some(idx) => println!(" | {}", names[idx]),
            None => println!(" | -"),
        }
    }

    // Success Rate
    let sr_values: Vec<f64> = results.iter().map(|r| r.success_rate()).collect();
    let sr_all_equal = sr_values.windows(2).all(|w| (w[0] - w[1]).abs() < 0.01);
    print!("{:<width$}", "Success Rate", width = metric_width);
    for r in results {
        print!(
            " | {:<width$}",
            format!("{:.2}%", r.success_rate()),
            width = col_width
        );
    }
    if sr_all_equal {
        println!(" | -");
    } else {
        let winner = find_winner(&sr_values, false);
        match winner {
            Some(idx) => println!(" | {}", names[idx]),
            None => println!(" | -"),
        }
    }

    // Avg TTFT
    print_row!(
        "Avg TTFT (ms)",
        results.iter().map(|r| mean(&r.ttft_values)).collect(),
        |v: f64| format!("{:.0} ms", v),
        true
    );

    // P95 TTFT
    print_row!(
        "P95 TTFT (ms)",
        results
            .iter()
            .map(|r| percentile(&r.ttft_values, 95.0))
            .collect(),
        |v: f64| format!("{:.0} ms", v),
        true
    );

    // P99 TTFT
    print_row!(
        "P99 TTFT (ms)",
        results
            .iter()
            .map(|r| percentile(&r.ttft_values, 99.0))
            .collect(),
        |v: f64| format!("{:.0} ms", v),
        true
    );

    // Avg Inter-Chunk Latency
    print_row!(
        "Avg ICL (ms)",
        results.iter().map(|r| mean(&r.itl_values)).collect(),
        |v: f64| format!("{:.1} ms", v),
        true
    );

    // P95 Inter-Chunk Latency
    print_row!(
        "P95 ICL (ms)",
        results
            .iter()
            .map(|r| percentile(&r.itl_values, 95.0))
            .collect(),
        |v: f64| format!("{:.1} ms", v),
        true
    );

    // Avg TPOT
    print_row!(
        "Avg TPOT (ms)",
        results.iter().map(|r| mean(&r.tpot_values)).collect(),
        |v: f64| format!("{:.1} ms", v),
        true
    );

    // Avg Tokens per Request (no winner - informational)
    print!("{:<width$}", "Avg Tokens/Request", width = metric_width);
    for r in results {
        print!(
            " | {:<width$}",
            format!("{:.0}", r.avg_tokens_per_request()),
            width = col_width
        );
    }
    println!(" | -");

    // Tokens per Chunk (informational)
    print!("{:<width$}", "Tokens/Chunk", width = metric_width);
    for r in results {
        print!(
            " | {:<width$}",
            format!("{:.2}", r.tokens_per_chunk()),
            width = col_width
        );
    }
    println!(" | -");

    // Avg Request Duration
    print_row!(
        "Avg Duration (ms)",
        results.iter().map(|r| r.avg_request_duration()).collect(),
        |v: f64| format!("{:.0} ms", v),
        true
    );

    // P95 Request Duration
    print_row!(
        "P95 Duration (ms)",
        results
            .iter()
            .map(|r| percentile(&r.request_duration_values, 95.0))
            .collect(),
        |v: f64| format!("{:.0} ms", v),
        true
    );

    // Throughput Section
    println!("\n## Throughput\n");

    print!("{:<width$}", "Metric", width = metric_width);
    for name in &names {
        print!(" | {:<width$}", name, width = col_width);
    }
    println!(" | Winner");
    println!("{}", "-".repeat(total_width));

    // Requests/sec
    print_row!(
        "Requests/sec",
        results.iter().map(|r| r.request_throughput()).collect(),
        |v: f64| format!("{:.2}", v),
        false
    );

    // Tokens/sec
    print_row!(
        "Tokens/sec",
        results
            .iter()
            .map(|r| r.output_token_throughput())
            .collect(),
        |v: f64| format!("{:.2}", v),
        false
    );

    // Chunks/sec
    print_row!(
        "Chunks/sec",
        results.iter().map(|r| r.chunks_per_sec()).collect(),
        |v: f64| format!("{:.2}", v),
        false
    );

    println!("\n{}", separator);
}

/// Load a scenario from a YAML file
/// Expand environment variables in a string
/// Replaces ${VAR_NAME} with the value of the environment variable
fn expand_env_vars(s: &str) -> Result<String> {
    let re = regex::Regex::new(r"\$\{([A-Z_][A-Z0-9_]*)\}").unwrap();
    let mut result = s.to_string();
    let mut missing_vars = Vec::new();

    for caps in re.captures_iter(s) {
        if let Some(var_name) = caps.get(1) {
            let var_name_str = var_name.as_str();
            match std::env::var(var_name_str) {
                Ok(value) => {
                    let pattern = format!("${{{}}}", var_name_str);
                    result = result.replace(&pattern, &value);
                }
                Err(_) => {
                    missing_vars.push(var_name_str.to_string());
                }
            }
        }
    }

    if !missing_vars.is_empty() {
        return Err(anyhow!(
            "Missing required environment variables: {}",
            missing_vars.join(", ")
        ));
    }

    Ok(result)
}

/// Expand environment variables in a scenario
pub fn expand_scenario_env_vars(mut scenario: Scenario) -> Result<Scenario> {
    for provider in &mut scenario.providers {
        provider.api_key = expand_env_vars(&provider.api_key)?;
        provider.base_url = expand_env_vars(&provider.base_url)?;
    }
    Ok(scenario)
}

pub fn load_scenario_from_file(path: &str) -> Result<Scenario> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read scenario file: {}", path))?;
    let scenario: Scenario = serde_yaml::from_str(&content)
        .with_context(|| format!("Failed to parse scenario file: {}", path))?;
    expand_scenario_env_vars(scenario)
}

/// Generate a timestamped output filename
pub fn generate_output_filename(scenario_name: &str, output_dir: &str) -> String {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let safe_name = scenario_name.replace([' ', '/'], "_").to_lowercase();
    format!("{}/{}_{}.txt", output_dir, safe_name, timestamp)
}

/// Save benchmark results to a file
pub fn save_results_to_file(
    results: &[BenchmarkResult],
    scenario_name: &str,
    output_path: &str,
) -> Result<()> {
    use std::io::Write;

    // Create output directory if it doesn't exist
    if let Some(parent) = std::path::Path::new(output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = std::fs::File::create(output_path)?;

    // Write header
    writeln!(file, "# Benchmark Results: {}", scenario_name)?;
    writeln!(
        file,
        "# Generated: {}",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    )?;
    writeln!(file, "#")?;
    writeln!(file)?;

    // Write individual results
    for result in results {
        write_result_to_file(&mut file, result)?;
        writeln!(file)?;
    }

    // Write comparison if multiple results
    if results.len() > 1 {
        write_comparison_to_file(&mut file, results)?;
    }

    info!("Results saved to: {}", output_path);
    Ok(())
}

fn write_result_to_file<W: std::io::Write>(file: &mut W, result: &BenchmarkResult) -> Result<()> {
    if let Some(name) = &result.name {
        writeln!(file, "============ {} ============", name)?;
    }
    writeln!(file, "============ Serving Benchmark Result ============")?;
    writeln!(
        file,
        "Successful requests:                     {}",
        result.successful_requests
    )?;
    writeln!(
        file,
        "Failed requests:                         {}",
        result.failed_requests
    )?;
    writeln!(
        file,
        "Requests with usage stats:               {}/{}",
        result.requests_with_usage, result.successful_requests
    )?;
    writeln!(
        file,
        "Maximum request concurrency:             {}",
        result.concurrency
    )?;
    writeln!(
        file,
        "Request rate configured (RPS):           {:.2}",
        result.rps_configured
    )?;
    writeln!(
        file,
        "Benchmark duration (s):                  {:.2}",
        result.duration_secs
    )?;
    writeln!(
        file,
        "Total input tokens:                      {}",
        result.total_input_tokens
    )?;
    writeln!(
        file,
        "Total generated tokens:                  {}",
        result.total_output_tokens
    )?;
    writeln!(
        file,
        "Total chunks received:                   {}",
        result.total_chunks
    )?;
    writeln!(
        file,
        "Avg tokens per request:                  {:.1}",
        result.avg_tokens_per_request()
    )?;

    let tpc = result.tokens_per_chunk();
    if tpc > 0.0 {
        let tpc_note = if tpc > 1.5 {
            " (batched)"
        } else if tpc < 0.8 {
            " (fragmented)"
        } else {
            ""
        };
        writeln!(
            file,
            "Tokens per chunk:                        {:.2}{}",
            tpc, tpc_note
        )?;
    }

    writeln!(
        file,
        "Request throughput (req/s):              {:.2}",
        result.request_throughput()
    )?;
    writeln!(
        file,
        "Output token throughput (tok/s):         {:.2}",
        result.output_token_throughput()
    )?;
    writeln!(
        file,
        "Total Token throughput (tok/s):          {:.2}",
        result.total_token_throughput()
    )?;
    writeln!(
        file,
        "Chunks throughput (chunks/s):            {:.2}",
        result.chunks_per_sec()
    )?;

    if !result.ttft_values.is_empty() {
        writeln!(file, "---------------Time to First Token----------------")?;
        writeln!(
            file,
            "Mean TTFT (ms):                          {:.2}",
            mean(&result.ttft_values)
        )?;
        writeln!(
            file,
            "Median TTFT (ms):                        {:.2}",
            median(&result.ttft_values)
        )?;
        writeln!(
            file,
            "P90 TTFT (ms):                           {:.2}",
            percentile(&result.ttft_values, 90.0)
        )?;
        writeln!(
            file,
            "P95 TTFT (ms):                           {:.2}",
            percentile(&result.ttft_values, 95.0)
        )?;
        writeln!(
            file,
            "P99 TTFT (ms):                           {:.2}",
            percentile(&result.ttft_values, 99.0)
        )?;
        writeln!(
            file,
            "P100 TTFT (ms):                          {:.2}",
            percentile(&result.ttft_values, 100.0)
        )?;
    }

    if !result.tpot_values.is_empty() {
        writeln!(file, "-----Time per Output Token (excl. 1st token)------")?;
        writeln!(
            file,
            "Mean TPOT (ms):                          {:.2}",
            mean(&result.tpot_values)
        )?;
        writeln!(
            file,
            "Median TPOT (ms):                        {:.2}",
            median(&result.tpot_values)
        )?;
        writeln!(
            file,
            "P90 TPOT (ms):                           {:.2}",
            percentile(&result.tpot_values, 90.0)
        )?;
        writeln!(
            file,
            "P95 TPOT (ms):                           {:.2}",
            percentile(&result.tpot_values, 95.0)
        )?;
        writeln!(
            file,
            "P99 TPOT (ms):                           {:.2}",
            percentile(&result.tpot_values, 99.0)
        )?;
        writeln!(
            file,
            "P100 TPOT (ms):                          {:.2}",
            percentile(&result.tpot_values, 100.0)
        )?;
    }

    if !result.itl_values.is_empty() {
        writeln!(file, "-----------Inter-Chunk Latency (ICL)--------------")?;
        writeln!(file, "(Time between SSE chunks, not individual tokens)")?;
        writeln!(
            file,
            "Mean ICL (ms):                           {:.2}",
            mean(&result.itl_values)
        )?;
        writeln!(
            file,
            "Median ICL (ms):                         {:.2}",
            median(&result.itl_values)
        )?;
        writeln!(
            file,
            "P90 ICL (ms):                            {:.2}",
            percentile(&result.itl_values, 90.0)
        )?;
        writeln!(
            file,
            "P95 ICL (ms):                            {:.2}",
            percentile(&result.itl_values, 95.0)
        )?;
        writeln!(
            file,
            "P99 ICL (ms):                            {:.2}",
            percentile(&result.itl_values, 99.0)
        )?;
        writeln!(
            file,
            "P100 ICL (ms):                           {:.2}",
            percentile(&result.itl_values, 100.0)
        )?;
    }

    if !result.request_duration_values.is_empty() {
        writeln!(file, "--------------Request Duration--------------------")?;
        writeln!(
            file,
            "Mean Duration (ms):                      {:.2}",
            mean(&result.request_duration_values)
        )?;
        writeln!(
            file,
            "Median Duration (ms):                    {:.2}",
            median(&result.request_duration_values)
        )?;
        writeln!(
            file,
            "P90 Duration (ms):                       {:.2}",
            percentile(&result.request_duration_values, 90.0)
        )?;
        writeln!(
            file,
            "P95 Duration (ms):                       {:.2}",
            percentile(&result.request_duration_values, 95.0)
        )?;
        writeln!(
            file,
            "P99 Duration (ms):                       {:.2}",
            percentile(&result.request_duration_values, 99.0)
        )?;
        writeln!(
            file,
            "P100 Duration (ms):                      {:.2}",
            percentile(&result.request_duration_values, 100.0)
        )?;
    }

    writeln!(file, "==================================================")?;
    Ok(())
}

fn write_comparison_to_file<W: std::io::Write>(
    file: &mut W,
    results: &[BenchmarkResult],
) -> Result<()> {
    let names: Vec<String> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            r.name
                .clone()
                .unwrap_or_else(|| format!("Provider {}", i + 1))
        })
        .collect();

    let separator = "=".repeat(100);
    let col_width = 12;
    let metric_width = 28;

    writeln!(file, "{}", separator)?;
    writeln!(file, "Comparison: {}", names.join(" vs "))?;
    writeln!(file, "{}", separator)?;

    let find_winner = |values: &[f64], lower_is_better: bool| -> Option<usize> {
        if values.iter().all(|&v| v == 0.0) {
            return None;
        }
        if lower_is_better {
            values
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 0.0)
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
        } else {
            values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
        }
    };

    writeln!(file, "\n## Summary Metrics\n")?;

    // Header
    write!(file, "{:<width$}", "Metric", width = metric_width)?;
    for name in &names {
        write!(file, " | {:<width$}", name, width = col_width)?;
    }
    writeln!(file, " | Winner")?;

    let total_width = metric_width + (col_width + 3) * names.len() + 10;
    writeln!(file, "{}", "-".repeat(total_width))?;

    // Helper macro for writing rows
    macro_rules! write_row {
        ($label:expr, $values:expr, $format_fn:expr, $lower_is_better:expr) => {{
            let values: Vec<f64> = $values;
            let winner_idx = find_winner(&values, $lower_is_better);
            write!(file, "{:<width$}", $label, width = metric_width)?;
            for v in &values {
                write!(file, " | {:<width$}", $format_fn(*v), width = col_width)?;
            }
            match winner_idx {
                Some(idx) => writeln!(file, " | {}", names[idx])?,
                None => writeln!(file, " | -")?,
            }
        }};
    }

    // Metrics
    write_row!(
        "Avg TTFT (ms)",
        results.iter().map(|r| mean(&r.ttft_values)).collect(),
        |v: f64| format!("{:.0} ms", v),
        true
    );
    write_row!(
        "P95 TTFT (ms)",
        results
            .iter()
            .map(|r| percentile(&r.ttft_values, 95.0))
            .collect(),
        |v: f64| format!("{:.0} ms", v),
        true
    );
    write_row!(
        "Avg TPOT (ms)",
        results.iter().map(|r| mean(&r.tpot_values)).collect(),
        |v: f64| format!("{:.1} ms", v),
        true
    );
    write_row!(
        "Avg Duration (ms)",
        results.iter().map(|r| r.avg_request_duration()).collect(),
        |v: f64| format!("{:.0} ms", v),
        true
    );
    write_row!(
        "P95 Duration (ms)",
        results
            .iter()
            .map(|r| percentile(&r.request_duration_values, 95.0))
            .collect(),
        |v: f64| format!("{:.0} ms", v),
        true
    );

    writeln!(file, "\n## Throughput\n")?;
    write!(file, "{:<width$}", "Metric", width = metric_width)?;
    for name in &names {
        write!(file, " | {:<width$}", name, width = col_width)?;
    }
    writeln!(file, " | Winner")?;
    writeln!(file, "{}", "-".repeat(total_width))?;

    write_row!(
        "Requests/sec",
        results.iter().map(|r| r.request_throughput()).collect(),
        |v: f64| format!("{:.2}", v),
        false
    );
    write_row!(
        "Tokens/sec",
        results
            .iter()
            .map(|r| r.output_token_throughput())
            .collect(),
        |v: f64| format!("{:.2}", v),
        false
    );

    writeln!(file, "\n{}", separator)?;
    Ok(())
}
