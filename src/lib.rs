//! GenAI Benchmark - Load testing library for OpenAI-compatible LLM APIs

use anyhow::{anyhow, Context, Result};
use async_channel::{bounded, Receiver, Sender};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, SeedableRng};
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tracing::{debug, info, warn};

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

/// Type of benchmark request
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RequestType {
    /// Standard chat completion requests
    #[default]
    ChatCompletion,
    /// Image generation requests
    ImageGeneration,
}

/// Configuration for image generation benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationConfig {
    /// Image size (e.g., "1024x1024", "512x512")
    #[serde(default = "default_image_size")]
    pub size: String,
    /// Number of images to generate per request
    #[serde(default = "default_image_n")]
    pub n: u32,
    /// Response format: "b64_json" or "url"
    #[serde(default = "default_image_response_format")]
    pub response_format: String,
    /// Image quality: "standard" or "hd"
    #[serde(default)]
    pub quality: Option<String>,
    /// Image style: "vivid" or "natural"
    #[serde(default)]
    pub style: Option<String>,
    /// Save generated images to output/images/
    #[serde(default)]
    pub save_images: bool,
}

fn default_image_size() -> String {
    "1024x1024".to_string()
}

fn default_image_n() -> u32 {
    1
}

fn default_image_response_format() -> String {
    "b64_json".to_string()
}

impl Default for ImageGenerationConfig {
    fn default() -> Self {
        Self {
            size: default_image_size(),
            n: default_image_n(),
            response_format: default_image_response_format(),
            quality: None,
            style: None,
            save_images: false,
        }
    }
}

/// Configuration for audio input in chat completions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInputConfig {
    /// Base64-encoded audio data or URL
    #[serde(default)]
    pub audio_url: Option<String>,
    /// Generate test audio if no URL provided
    #[serde(default = "default_true")]
    pub use_test_audio: bool,
}

fn default_true() -> bool {
    true
}

impl Default for AudioInputConfig {
    fn default() -> Self {
        Self {
            audio_url: None,
            use_test_audio: true,
        }
    }
}

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
    /// Enable TEE signature verification
    #[serde(default)]
    pub verify: bool,
    /// Enable random prompt selection (default: false, uses sequential)
    #[serde(default)]
    pub random_prompt_selection: bool,
    /// Random seed for prompt selection (None = use system entropy)
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Type of request to benchmark
    #[serde(default)]
    pub request_type: RequestType,
    /// Image generation configuration (when request_type is ImageGeneration)
    #[serde(default)]
    pub image_config: Option<ImageGenerationConfig>,
    /// Enable audio input in chat completions
    #[serde(default)]
    pub audio_input: Option<AudioInputConfig>,
    /// Enable audio output (modalities: ["audio"]) for models like Qwen3-Omni
    #[serde(default)]
    pub audio_output: bool,
    /// Output directory for saving generated images (computed at runtime)
    #[serde(skip)]
    pub image_output_dir: Option<String>,
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
    /// Enable TEE signature verification
    #[serde(default)]
    pub verify: bool,
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
    /// Enable random prompt selection (default: false, uses sequential)
    #[serde(default)]
    pub random_prompt_selection: bool,
    /// Random seed for prompt selection (None = use system entropy)
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Type of request to benchmark
    #[serde(default)]
    pub request_type: RequestType,
    /// Image generation configuration (when request_type is ImageGeneration)
    #[serde(default)]
    pub image_config: Option<ImageGenerationConfig>,
    /// Enable audio input in chat completions
    #[serde(default)]
    pub audio_input: Option<AudioInputConfig>,
    /// Enable audio output (modalities: ["audio"]) for models like Qwen3-Omni
    #[serde(default)]
    pub audio_output: bool,
}

fn default_num_requests() -> usize {
    100
}

/// Content part for multimodal messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content
    Text { text: String },
    /// Input audio - OpenAI format
    InputAudio { input_audio: InputAudioData },
    /// Audio URL - vLLM format (data URI with base64)
    AudioUrl { audio_url: AudioUrlData },
    /// Video URL - vLLM format (data URI with base64)
    VideoUrl { video_url: VideoUrlData },
    /// Image URL content (for vision models)
    ImageUrl { image_url: ImageUrl },
}

/// Input audio data (OpenAI format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioData {
    /// Base64-encoded audio data
    pub data: String,
    /// Audio format (e.g., "wav", "mp3")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// Audio URL specification (vLLM format)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AudioUrlData {
    String(String),
    Object { url: String },
}

/// Video URL specification (vLLM format)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VideoUrlData {
    String(String),
    Object { url: String },
}

/// Image URL specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL or data URI
    pub url: String,
    /// Detail level: "auto", "low", or "high"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Message content - can be simple string or array of content parts
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text(String),
    /// Multimodal content parts
    Parts(Vec<ContentPart>),
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

impl Message {
    /// Create a simple text message
    pub fn text(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: MessageContent::Text(content.to_string()),
        }
    }

    /// Create a message with audio URL content (vLLM format) - default for compatibility
    pub fn with_audio_url(role: &str, text: &str, audio_base64: &str) -> Self {
        let data_url = format!("data:audio/wav;base64,{}", audio_base64);
        Self {
            role: role.to_string(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: text.to_string(),
                },
                ContentPart::AudioUrl {
                    audio_url: AudioUrlData::String(data_url),
                },
            ]),
        }
    }

    /// Create a message with input audio content (OpenAI format)
    pub fn with_input_audio(role: &str, text: &str, audio_base64: &str) -> Self {
        Self {
            role: role.to_string(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: text.to_string(),
                },
                ContentPart::InputAudio {
                    input_audio: InputAudioData {
                        data: audio_base64.to_string(),
                        format: Some("wav".to_string()),
                    },
                },
            ]),
        }
    }

    /// Get text content (for preview)
    pub fn text_content(&self) -> String {
        match &self.content {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
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
    /// Whether TEE signature verification was attempted
    pub verification_attempted: bool,
    /// Whether TEE signature verification succeeded
    pub verification_success: bool,
    /// Time taken for TEE signature verification (ms)
    pub verification_time_ms: f64,
    /// First 20 tokens of the prompt for reference
    pub prompt_preview: String,
    /// Type of request
    pub request_type: RequestType,
    // Audio-specific metrics
    /// Size of audio input in bytes (if any)
    pub audio_input_size_bytes: Option<u64>,
    /// Size of audio output in bytes (if any)
    pub audio_output_size_bytes: Option<u64>,
    /// Whether audio output was received
    pub has_audio_output: bool,
    // Image generation metrics
    /// Number of images generated
    pub image_count: u32,
    /// Total size of generated images in bytes
    pub total_image_size_bytes: u64,
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

/// Extract first N tokens (words) from messages for preview
fn extract_prompt_preview(messages: &[Message], max_tokens: usize) -> String {
    let full_text: String = messages
        .iter()
        .filter(|m| m.role == "user")
        .map(|m| m.text_content())
        .collect::<Vec<_>>()
        .join(" ");

    let tokens: Vec<&str> = full_text.split_whitespace().collect();
    let preview_tokens: Vec<&str> = tokens.iter().take(max_tokens).copied().collect();
    let preview = preview_tokens.join(" ");

    if tokens.len() > max_tokens {
        format!("{}...", preview)
    } else {
        preview
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
    /// Number of requests where TEE verification was attempted
    pub verification_attempted: usize,
    /// Number of requests where TEE verification succeeded
    pub verification_success: usize,
    /// Number of requests where TEE verification failed
    pub verification_failed: usize,
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
    /// Verification latency values (ms) for statistics
    #[serde(skip)]
    pub verification_time_values: Vec<f64>,
    /// Sample prompt previews (first 5 unique prompts)
    #[serde(skip)]
    pub sample_prompts: Vec<String>,
    /// Type of benchmark request
    pub request_type: RequestType,
    // Audio-specific aggregate metrics
    /// Number of requests with audio input
    pub audio_input_requests: usize,
    /// Number of requests with audio output
    pub audio_output_requests: usize,
    /// Total audio input bytes processed
    pub total_audio_input_bytes: u64,
    /// Total audio output bytes received
    pub total_audio_output_bytes: u64,
    // Image generation aggregate metrics
    /// Total images generated
    pub total_images_generated: u64,
    /// Total image data size in bytes
    pub total_image_bytes: u64,
    /// Image generation time values (ms) for statistics
    #[serde(skip)]
    pub image_generation_time_values: Vec<f64>,
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
    /// Output modalities (e.g., ["text"], ["audio"], or ["text", "audio"])
    #[serde(skip_serializing_if = "Option::is_none")]
    modalities: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    /// Chat completion ID for TEE signature verification
    #[serde(default)]
    id: Option<String>,
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChunkChoice {
    delta: Option<Delta>,
    /// For non-streaming audio responses
    message: Option<ChunkMessage>,
    finish_reason: Option<String>,
    #[serde(default)]
    index: u32,
}

#[derive(Debug, Deserialize)]
struct Delta {
    content: Option<String>,
    /// Modality indicator for streaming (e.g., "text" or "audio")
    #[serde(default)]
    modality: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChunkMessage {
    role: Option<String>,
    content: Option<String>,
    /// Audio output data
    #[serde(default)]
    audio: Option<AudioOutput>,
}

/// Audio output data from Qwen3-Omni and similar models
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct AudioOutput {
    /// Base64-encoded audio data
    pub data: String,
    /// Audio format (e.g., "wav")
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

/// TEE signature response from /signature/{chat_id} endpoint
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct SignatureResponse {
    text: String,
    signature: String,
    signing_address: String,
    signing_algo: String,
}

/// Image generation request
#[derive(Debug, Clone, Serialize)]
struct ImageGenerationRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<String>,
}

/// Image generation response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ImageGenerationResponse {
    /// Response ID for signature verification
    #[serde(default)]
    id: Option<String>,
    created: i64,
    data: Vec<ImageDataResponse>,
}

/// Individual generated image in response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ImageDataResponse {
    /// Base64-encoded image data
    #[serde(default)]
    b64_json: Option<String>,
    /// URL to the generated image
    #[serde(default)]
    url: Option<String>,
    /// Revised prompt used for generation
    #[serde(default)]
    revised_prompt: Option<String>,
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
// Audio Helpers
// ============================================================================

/// Generate a minimal WAV file for testing audio input
/// Creates a 1-second mono 8kHz silent WAV file (~8KB)
pub fn generate_test_audio_base64() -> String {
    // WAV file header for 8-bit mono PCM at 8000 Hz, 1 second duration
    let sample_rate: u32 = 8000;
    let duration_seconds: u32 = 1;
    let bits_per_sample: u16 = 8;
    let num_channels: u16 = 1;
    let byte_rate: u32 = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align: u16 = num_channels * bits_per_sample / 8;
    let data_size: u32 =
        sample_rate * duration_seconds * num_channels as u32 * bits_per_sample as u32 / 8;
    let file_size: u32 = 36 + data_size;

    let mut wav_data = Vec::with_capacity(file_size as usize + 8);

    // RIFF header
    wav_data.extend_from_slice(b"RIFF");
    wav_data.extend_from_slice(&file_size.to_le_bytes());
    wav_data.extend_from_slice(b"WAVE");

    // fmt chunk
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // audio format (PCM)
    wav_data.extend_from_slice(&num_channels.to_le_bytes());
    wav_data.extend_from_slice(&sample_rate.to_le_bytes());
    wav_data.extend_from_slice(&byte_rate.to_le_bytes());
    wav_data.extend_from_slice(&block_align.to_le_bytes());
    wav_data.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&data_size.to_le_bytes());

    // Silent audio data (128 is silence for 8-bit audio)
    wav_data.extend(vec![128u8; data_size as usize]);

    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(&wav_data)
}

/// Calculate audio data size from base64 string
pub fn audio_base64_size_bytes(base64_audio: &str) -> u64 {
    // Base64 encodes 3 bytes as 4 characters
    // So decoded size is approximately (len * 3) / 4
    (base64_audio.len() as u64 * 3) / 4
}

/// Generate test prompts for image generation
pub fn generate_image_prompts(count: usize, seed: Option<u64>) -> Vec<String> {
    use rand::{Rng, SeedableRng};

    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

    let prompts = [
        "A futuristic cityscape at sunset with flying cars",
        "A serene mountain landscape with a crystal clear lake",
        "An abstract digital art piece with vibrant colors",
        "A steampunk-style mechanical robot in a workshop",
        "A cozy coffee shop interior with warm lighting",
        "A space station orbiting Earth with stars in the background",
        "A fantasy forest with glowing mushrooms and fireflies",
        "A minimalist geometric pattern in pastel colors",
        "A vintage retro poster design for a music festival",
        "An underwater scene with colorful coral reef and fish",
    ];

    (0..count)
        .map(|_| prompts[rng.gen_range(0..prompts.len())].to_string())
        .collect()
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
                    Message::text(role, &m.value)
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
        .map(|line| vec![Message::text("user", line)])
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
            vec![Message::text("user", topic)]
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

/// Fetch TEE signature for a chat completion with retry logic
async fn fetch_tee_signature(
    client: &Client,
    base_url: &str,
    api_key: &str,
    chat_id: &str,
) -> Result<SignatureResponse> {
    let url = format!("{}/signature/{}", base_url.trim_end_matches('/'), chat_id);

    // Retry up to 3 times with exponential backoff
    const MAX_RETRIES: u32 = 3;
    let mut last_error = None;

    for attempt in 0..MAX_RETRIES {
        if attempt > 0 {
            let delay_ms = 500 * (1 << (attempt - 1)); // 500ms, 1000ms
            debug!(
                "Retrying signature fetch (attempt {}/{}) after {}ms",
                attempt + 1,
                MAX_RETRIES,
                delay_ms
            );
            sleep(Duration::from_millis(delay_ms)).await;
        }

        let mut req_builder = client.get(&url);

        if !api_key.is_empty() {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = match req_builder.send().await {
            Ok(r) => r,
            Err(e) => {
                last_error = Some(anyhow!("Request error: {}", e));
                continue;
            }
        };

        if !response.status().is_success() {
            if response.status().as_u16() == 404 && attempt < MAX_RETRIES - 1 {
                // 404 might mean signature not cached yet, retry
                last_error = Some(anyhow!("Signature not found (404), retrying..."));
                continue;
            }
            last_error = Some(anyhow!(
                "TEE signature request failed with status: {}",
                response.status()
            ));
            continue;
        }

        let signature: SignatureResponse = match response.json().await {
            Ok(s) => s,
            Err(e) => {
                last_error = Some(anyhow!("Failed to parse response: {}", e));
                continue;
            }
        };

        return Ok(signature);
    }

    Err(last_error.unwrap_or_else(|| {
        anyhow!(
            "Failed to fetch TEE signature after {} retries",
            MAX_RETRIES
        )
    }))
}

/// Send an image generation request and collect metrics
async fn send_image_generation_request(
    client: &Client,
    base_url: &str,
    api_key: &str,
    prompt: &str,
    config: &ImageGenerationConfig,
    model: &str,
    timeout: Duration,
    verify: bool,
    request_index: usize,
    output_dir: Option<&str>,
) -> Result<RequestMetrics> {
    let url = format!("{}/images/generations", base_url.trim_end_matches('/'));
    let start_time = Instant::now();

    let request = ImageGenerationRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        n: Some(config.n),
        size: Some(config.size.clone()),
        response_format: Some(config.response_format.clone()),
        quality: config.quality.clone(),
        style: config.style.clone(),
    };

    let mut req_builder = client
        .post(&url)
        .header("Content-Type", "application/json")
        .timeout(timeout);

    if !api_key.is_empty() {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
    }

    let response = req_builder
        .json(&request)
        .send()
        .await
        .context("Failed to send image generation request")?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Image generation request failed with status: {}",
            response.status()
        ));
    }

    let image_response: ImageGenerationResponse = response
        .json()
        .await
        .context("Failed to parse image generation response")?;

    let total_time = start_time.elapsed();

    // Calculate total image size
    let mut total_image_size_bytes: u64 = 0;
    let image_count = image_response.data.len() as u32;

    for img in &image_response.data {
        if let Some(b64) = &img.b64_json {
            // Base64 decodes to approximately (len * 3) / 4 bytes
            total_image_size_bytes += (b64.len() as u64 * 3) / 4;
        }
    }

    // Save images to disk if configured
    if config.save_images {
        if let Some(output_dir) = output_dir {
            use base64::Engine;

            // Create directory if needed
            if let Err(e) = std::fs::create_dir_all(output_dir) {
                warn!(
                    "Failed to create image output directory {}: {}",
                    output_dir, e
                );
            } else {
                // Sanitize prompt for filename (first ~50 chars, alphanumeric + underscores)
                let safe_prompt: String = prompt
                    .chars()
                    .take(50)
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect();
                let safe_prompt = safe_prompt.trim_matches('_').to_lowercase();

                for (img_idx, img) in image_response.data.iter().enumerate() {
                    if let Some(b64) = &img.b64_json {
                        match base64::engine::general_purpose::STANDARD.decode(b64) {
                            Ok(bytes) => {
                                let filename = format!(
                                    "{:03}_{}{}.png",
                                    request_index,
                                    safe_prompt,
                                    if img_idx > 0 {
                                        format!("_{}", img_idx)
                                    } else {
                                        String::new()
                                    }
                                );
                                let filepath = format!("{}/{}", output_dir, filename);
                                if let Err(e) = std::fs::write(&filepath, bytes) {
                                    warn!("Failed to save image to {}: {}", filepath, e);
                                } else {
                                    debug!("Saved image to {}", filepath);
                                }
                            }
                            Err(e) => {
                                warn!("Failed to decode base64 image: {}", e);
                            }
                        }
                    }
                }
            }
        }
    }

    // TEE signature verification for image generation
    let (verification_attempted, verification_success, verification_time_ms) = if verify {
        if let Some(id) = &image_response.id {
            // Wait for signature to be cached by vLLM-proxy
            sleep(Duration::from_millis(1000)).await;

            debug!("Verifying TEE signature for image ID: {}", id);
            let verify_start = Instant::now();
            match fetch_tee_signature(client, base_url, api_key, id).await {
                Ok(sig) => {
                    let verify_time = verify_start.elapsed().as_secs_f64() * 1000.0;
                    debug!(
                        "TEE signature verified in {:.2}ms: signing_address={}, algo={}",
                        verify_time, sig.signing_address, sig.signing_algo
                    );
                    (true, true, verify_time)
                }
                Err(e) => {
                    let verify_time = verify_start.elapsed().as_secs_f64() * 1000.0;
                    warn!(
                        "TEE signature verification failed for {} in {:.2}ms: {}",
                        id, verify_time, e
                    );
                    (true, false, verify_time)
                }
            }
        } else {
            warn!("Cannot verify TEE signature: no image ID received");
            (true, false, 0.0)
        }
    } else {
        (false, false, 0.0)
    };

    // Truncate prompt for preview
    let prompt_preview = if prompt.len() > 100 {
        format!("{}...", &prompt[..100])
    } else {
        prompt.to_string()
    };

    Ok(RequestMetrics {
        success: true,
        input_tokens: 0,
        output_tokens: 0,
        output_chars: 0,
        chunk_count: 0,
        got_usage: false,
        ttft_ms: total_time.as_secs_f64() * 1000.0, // Use total time as TTFT for images
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        inter_chunk_latencies: Vec::new(),
        verification_attempted,
        verification_success,
        verification_time_ms,
        prompt_preview,
        request_type: RequestType::ImageGeneration,
        audio_input_size_bytes: None,
        audio_output_size_bytes: None,
        has_audio_output: false,
        image_count,
        total_image_size_bytes,
    })
}

async fn send_streaming_request(
    client: &Client,
    base_url: &str,
    api_key: &str,
    request: ChatCompletionRequest,
    timeout: Duration,
    verify: bool,
    prompt_preview: String,
    audio_input_size: Option<u64>,
    request_type: RequestType,
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
    let mut chat_id: Option<String> = None;
    // Audio tracking
    let audio_input_size_bytes: Option<u64> = audio_input_size;
    let mut audio_output_size_bytes: Option<u64> = None;
    let mut has_audio_output: bool = false;

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
                        // Capture chat ID from first chunk (same across all chunks)
                        if chat_id.is_none() {
                            if let Some(id) = &chunk.id {
                                chat_id = Some(id.clone());
                                debug!("Captured chat ID: {}", id);
                            }
                        }

                        // Check for usage stats (may come multiple times, always use latest)
                        if let Some(usage) = &chunk.usage {
                            let previous_output = output_tokens;
                            input_tokens = usage.prompt_tokens;
                            output_tokens = usage.completion_tokens;
                            got_usage = true;
                            debug!(
                                "Got usage stats: input={}, output={} (previous: {}), chunk_count={}",
                                input_tokens, output_tokens, previous_output, chunk_count
                            );
                        }

                        // Process content from choices
                        if let Some(choice) = chunk.choices.first() {
                            // Handle streaming delta content
                            if let Some(delta) = &choice.delta {
                                if let Some(content) = &delta.content {
                                    if !content.is_empty() {
                                        output_chars += content.len();

                                        if first_token_time.is_none() {
                                            first_token_time = Some(now);
                                        } else {
                                            let icl =
                                                now.duration_since(last_chunk_time).as_secs_f64()
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

                                // Track audio modality if present
                                if let Some(modality) = &delta.modality {
                                    if modality == "audio" {
                                        debug!("Received audio modality chunk");
                                    }
                                }
                            }

                            // Handle non-streaming message (for audio responses)
                            if let Some(message) = &choice.message {
                                // Check for audio output
                                if let Some(audio) = &message.audio {
                                    has_audio_output = true;
                                    audio_output_size_bytes = Some(audio.data.len() as u64);
                                    debug!("Got audio output: {} bytes", audio.data.len());
                                }
                                // Check for text content in message
                                if let Some(content) = &message.content {
                                    if !content.is_empty() {
                                        output_chars += content.len();
                                        if first_token_time.is_none() {
                                            first_token_time = Some(now);
                                        }
                                    }
                                }
                            }

                            // Don't break early - some models (like Qwen3-Omni) send usage updates
                            // in later chunks after finish_reason. Wait for [DONE] instead.
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

    // TEE signature verification
    let (verification_attempted, verification_success, verification_time_ms) = if verify {
        if let Some(id) = &chat_id {
            // Wait for signature to be cached by vLLM-proxy (signatures are stored after stream completes)
            sleep(Duration::from_millis(1000)).await;

            debug!("Verifying TEE signature for chat ID: {}", id);
            let verify_start = Instant::now();
            match fetch_tee_signature(client, base_url, api_key, id).await {
                Ok(sig) => {
                    let verify_time = verify_start.elapsed().as_secs_f64() * 1000.0;
                    debug!(
                        "TEE signature verified in {:.2}ms: signing_address={}, algo={}",
                        verify_time, sig.signing_address, sig.signing_algo
                    );
                    (true, true, verify_time)
                }
                Err(e) => {
                    let verify_time = verify_start.elapsed().as_secs_f64() * 1000.0;
                    warn!(
                        "TEE signature verification failed for {} in {:.2}ms: {}",
                        id, verify_time, e
                    );
                    (true, false, verify_time)
                }
            }
        } else {
            warn!("Cannot verify TEE signature: no chat ID received");
            (true, false, 0.0)
        }
    } else {
        (false, false, 0.0)
    };

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
        verification_attempted,
        verification_success,
        verification_time_ms,
        prompt_preview,
        request_type,
        audio_input_size_bytes,
        audio_output_size_bytes,
        has_audio_output,
        image_count: 0,
        total_image_size_bytes: 0,
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
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA {eta}) {msg}",
            )?
            .progress_chars("#>-"),
    );

    let active_requests = Arc::new(AtomicUsize::new(0));
    let success_count = Arc::new(AtomicUsize::new(0));
    let failed_count = Arc::new(AtomicUsize::new(0));
    let start_time = Instant::now();

    let request_delay = if config.rps > 0.0 {
        Duration::from_secs_f64(1.0 / config.rps)
    } else {
        Duration::ZERO
    };

    // Pre-generate prompt indices based on selection strategy
    let prompt_indices: Vec<usize> = if config.random_prompt_selection {
        // Generate random indices for all requests
        let mut rng: Box<dyn rand::RngCore + Send> = match config.random_seed {
            Some(seed) => {
                info!("Using random prompt selection with seed: {}", seed);
                Box::new(rand::rngs::StdRng::seed_from_u64(seed))
            }
            None => {
                info!("Using random prompt selection with system entropy");
                Box::new(rand::rngs::StdRng::from_entropy())
            }
        };
        (0..num_requests)
            .map(|_| rng.gen_range(0..prompts.len()))
            .collect()
    } else {
        info!("Using sequential prompt selection");
        (0..num_requests).map(|i| i % prompts.len()).collect()
    };

    let mut handles = Vec::new();

    // Spawn background task to update progress message with live stats
    let progress_updater = progress.clone();
    let active_for_display = active_requests.clone();
    let success_for_display = success_count.clone();
    let failed_for_display = failed_count.clone();
    let max_concurrency = config.concurrency;
    let rps_limit = config.rps;

    tokio::spawn(async move {
        loop {
            let active = active_for_display.load(Ordering::Relaxed);
            let ok = success_for_display.load(Ordering::Relaxed);
            let err = failed_for_display.load(Ordering::Relaxed);

            let rps_str = if rps_limit > 0.0 {
                format!(" | rps: {}", rps_limit)
            } else {
                String::new()
            };

            progress_updater.set_message(format!(
                "| {}/{} active | {} ok, {} err{}",
                active, max_concurrency, ok, err, rps_str
            ));

            if progress_updater.is_finished() {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
    });

    // Pre-generate test audio if needed (just base64, not data URL)
    let test_audio_base64 = if config.audio_input.is_some() {
        Some(generate_test_audio_base64())
    } else {
        None
    };

    for i in 0..num_requests {
        let permit = semaphore.clone().acquire_owned().await?;
        let client = client.clone();
        let tx = tx.clone();
        let progress = progress.clone();
        let active = active_requests.clone();
        let success = success_count.clone();
        let failed = failed_count.clone();

        let mut messages = prompts[prompt_indices[i]].clone();

        // Calculate audio input size if adding audio
        let audio_input_size = if let Some(audio_base64) = &test_audio_base64 {
            // Add audio content to the last user message
            // Use audio_url format for vLLM compatibility
            if let Some(last_user_msg) = messages.iter_mut().rev().find(|m| m.role == "user") {
                let text_content = last_user_msg.text_content();
                *last_user_msg = Message::with_audio_url("user", &text_content, audio_base64);
            }
            Some(audio_base64_size_bytes(audio_base64))
        } else {
            None
        };

        let prompt_preview = extract_prompt_preview(&messages, 20);
        let config_base_url = config.base_url.clone();
        let config_api_key = config.api_key.clone();
        let config_model = config.model.clone();
        let config_max_tokens = config.max_tokens;
        let config_timeout = config.timeout();
        let config_verify = config.verify;
        let config_audio_output = config.audio_output;
        let config_request_type = config.request_type.clone();
        let config_image_config = config.image_config.clone();
        let config_image_output_dir = config.image_output_dir.clone();
        let request_index = i;

        active.fetch_add(1, Ordering::SeqCst);

        let handle = tokio::spawn(async move {
            let result = match config_request_type {
                RequestType::ImageGeneration => {
                    // For image generation, extract text prompt from messages
                    let text_prompt = messages
                        .iter()
                        .filter(|m| m.role == "user")
                        .map(|m| m.text_content())
                        .collect::<Vec<_>>()
                        .join(" ");

                    let img_config = config_image_config.unwrap_or_default();

                    send_image_generation_request(
                        &client,
                        &config_base_url,
                        &config_api_key,
                        &text_prompt,
                        &img_config,
                        &config_model,
                        config_timeout,
                        config_verify,
                        request_index,
                        config_image_output_dir.as_deref(),
                    )
                    .await
                }
                RequestType::ChatCompletion => {
                    // Set modalities for audio output if requested
                    let modalities = if config_audio_output {
                        Some(vec!["text".to_string(), "audio".to_string()])
                    } else {
                        None
                    };

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
                        modalities,
                    };

                    send_streaming_request(
                        &client,
                        &config_base_url,
                        &config_api_key,
                        request,
                        config_timeout,
                        config_verify,
                        prompt_preview.clone(),
                        audio_input_size,
                        RequestType::ChatCompletion,
                    )
                    .await
                }
            };

            active.fetch_sub(1, Ordering::SeqCst);
            progress.inc(1);
            drop(permit);

            match result {
                Ok(metrics) => {
                    if metrics.success {
                        success.fetch_add(1, Ordering::Relaxed);
                    } else {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                    let _ = tx.send(metrics).await;
                }
                Err(e) => {
                    failed.fetch_add(1, Ordering::Relaxed);
                    progress.println(format!("ERROR: Request failed: {}", e));
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
                            verification_attempted: false,
                            verification_success: false,
                            verification_time_ms: 0.0,
                            prompt_preview,
                            request_type: config_request_type,
                            audio_input_size_bytes: None,
                            audio_output_size_bytes: None,
                            has_audio_output: false,
                            image_count: 0,
                            total_image_size_bytes: 0,
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
    let mut verification_attempted = 0;
    let mut verification_success = 0;
    let mut verification_failed = 0;
    let mut ttft_values: Vec<f64> = Vec::new();
    let mut tpot_values: Vec<f64> = Vec::new();
    let mut itl_values: Vec<f64> = Vec::new();
    let mut request_duration_values: Vec<f64> = Vec::new();
    let mut tokens_per_request: Vec<u32> = Vec::new();
    let mut verification_time_values: Vec<f64> = Vec::new();
    let mut sample_prompts: Vec<String> = Vec::new();
    // Audio/image metrics
    let mut audio_input_requests = 0;
    let mut audio_output_requests = 0;
    let mut total_audio_input_bytes: u64 = 0;
    let mut total_audio_output_bytes: u64 = 0;
    let mut total_images_generated: u64 = 0;
    let mut total_image_bytes: u64 = 0;
    let mut image_generation_time_values: Vec<f64> = Vec::new();

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

            // Collect sample prompts (first 5 unique ones)
            if sample_prompts.len() < 5 && !sample_prompts.contains(&metrics.prompt_preview) {
                sample_prompts.push(metrics.prompt_preview.clone());
            }

            // Track verification stats
            if metrics.verification_attempted {
                verification_attempted += 1;
                verification_time_values.push(metrics.verification_time_ms);
                if metrics.verification_success {
                    verification_success += 1;
                } else {
                    verification_failed += 1;
                }
            }

            if let Some(tpot) = metrics.tpot_ms() {
                tpot_values.push(tpot);
            }

            itl_values.extend(metrics.inter_chunk_latencies);

            // Track audio metrics
            if let Some(audio_in) = metrics.audio_input_size_bytes {
                audio_input_requests += 1;
                total_audio_input_bytes += audio_in;
            }
            if metrics.has_audio_output {
                audio_output_requests += 1;
                if let Some(audio_out) = metrics.audio_output_size_bytes {
                    total_audio_output_bytes += audio_out;
                }
            }

            // Track image generation metrics
            if metrics.request_type == RequestType::ImageGeneration {
                total_images_generated += metrics.image_count as u64;
                total_image_bytes += metrics.total_image_size_bytes;
                image_generation_time_values.push(metrics.total_time_ms);
            }
        } else {
            failed += 1;
        }
    }

    ttft_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tpot_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    itl_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    request_duration_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    verification_time_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    image_generation_time_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
        verification_attempted,
        verification_success,
        verification_failed,
        ttft_values,
        tpot_values,
        itl_values,
        request_duration_values,
        tokens_per_request,
        verification_time_values,
        sample_prompts,
        request_type: config.request_type.clone(),
        audio_input_requests,
        audio_output_requests,
        total_audio_input_bytes,
        total_audio_output_bytes,
        total_images_generated,
        total_image_bytes,
        image_generation_time_values,
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

    // Generate image output directory if save_images is enabled
    let image_output_dir = if let Some(ref img_config) = scenario.image_config {
        if img_config.save_images {
            let safe_name = scenario.name.replace(' ', "_").to_lowercase();
            let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
            Some(format!("output/images/{}_{}", safe_name, timestamp))
        } else {
            None
        }
    } else {
        None
    };

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
            verify: provider.verify,
            random_prompt_selection: scenario.random_prompt_selection,
            random_seed: scenario.random_seed,
            request_type: scenario.request_type.clone(),
            image_config: scenario.image_config.clone(),
            audio_input: scenario.audio_input.clone(),
            audio_output: scenario.audio_output,
            image_output_dir: image_output_dir.clone(),
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

    // Show sample prompts
    if !result.sample_prompts.is_empty() {
        println!();
        println!("Sample prompts used (first 20 tokens):");
        for (i, prompt) in result.sample_prompts.iter().enumerate() {
            println!("  {}. {}", i + 1, prompt);
        }
        println!();
    }

    // Show TEE verification stats if any verification was attempted
    if result.verification_attempted > 0 {
        if result.verification_failed > 0 {
            println!(
                "TEE verification:                        {}/{} ⚠️",
                result.verification_success, result.verification_attempted
            );
        } else {
            println!(
                "TEE verification:                        {}/{} ✓",
                result.verification_success, result.verification_attempted
            );
        }
    }

    // Show request type
    match result.request_type {
        RequestType::ImageGeneration => {
            println!("Request type:                            Image Generation");
        }
        RequestType::ChatCompletion => {
            println!("Request type:                            Chat Completion");
        }
    }

    // Show audio metrics if any
    if result.audio_input_requests > 0 || result.audio_output_requests > 0 {
        println!();
        println!("-----------Audio Metrics--------------");
        if result.audio_input_requests > 0 {
            println!(
                "Requests with audio input:               {}",
                result.audio_input_requests
            );
            if result.total_audio_input_bytes > 0 {
                println!(
                    "Total audio input:                       {:.2} KB",
                    result.total_audio_input_bytes as f64 / 1024.0
                );
            }
        }
        if result.audio_output_requests > 0 {
            println!(
                "Requests with audio output:              {}",
                result.audio_output_requests
            );
            if result.total_audio_output_bytes > 0 {
                println!(
                    "Total audio output:                      {:.2} KB",
                    result.total_audio_output_bytes as f64 / 1024.0
                );
            }
        }
    }

    // Show image generation metrics if any
    if result.request_type == RequestType::ImageGeneration {
        println!();
        println!("-----------Image Generation Metrics-----------");
        println!(
            "Total images generated:                  {}",
            result.total_images_generated
        );
        if result.total_image_bytes > 0 {
            println!(
                "Total image data:                        {:.2} MB",
                result.total_image_bytes as f64 / (1024.0 * 1024.0)
            );
            println!(
                "Avg image size:                          {:.2} KB",
                (result.total_image_bytes as f64 / result.total_images_generated.max(1) as f64)
                    / 1024.0
            );
        }
        if !result.image_generation_time_values.is_empty() {
            println!(
                "Mean generation time (ms):               {:.2}",
                mean(&result.image_generation_time_values)
            );
            println!(
                "P95 generation time (ms):                {:.2}",
                percentile(&result.image_generation_time_values, 95.0)
            );
        }
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
        println!("-----------Total Request Duration----------------");
        println!(
            "Mean Total Duration (ms):                 {:.2}",
            mean(&result.request_duration_values)
        );
        println!(
            "Median Total Duration (ms):               {:.2}",
            median(&result.request_duration_values)
        );
        println!(
            "P90 Total Duration (ms):                  {:.2}",
            percentile(&result.request_duration_values, 90.0)
        );
        println!(
            "P95 Total Duration (ms):                  {:.2}",
            percentile(&result.request_duration_values, 95.0)
        );
        println!(
            "P99 Total Duration (ms):                  {:.2}",
            percentile(&result.request_duration_values, 99.0)
        );
        println!(
            "P100 Total Duration (ms):                 {:.2}",
            percentile(&result.request_duration_values, 100.0)
        );
    }

    if !result.verification_time_values.is_empty() {
        println!("-----------TEE Verification Latency---------------");
        println!(
            "Mean Verification (ms):                  {:.2}",
            mean(&result.verification_time_values)
        );
        println!(
            "Median Verification (ms):                {:.2}",
            median(&result.verification_time_values)
        );
        println!(
            "P90 Verification (ms):                   {:.2}",
            percentile(&result.verification_time_values, 90.0)
        );
        println!(
            "P95 Verification (ms):                   {:.2}",
            percentile(&result.verification_time_values, 95.0)
        );
        println!(
            "P99 Verification (ms):                   {:.2}",
            percentile(&result.verification_time_values, 99.0)
        );
        println!(
            "P100 Verification (ms):                  {:.2}",
            percentile(&result.verification_time_values, 100.0)
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

    // TEE Verification (only show if any provider has verification enabled)
    let any_verification = results.iter().any(|r| r.verification_attempted > 0);
    if any_verification {
        print!("{:<width$}", "TEE Verification", width = metric_width);
        for r in results {
            if r.verification_attempted > 0 {
                print!(
                    " | {:<width$}",
                    format!("{}/{}", r.verification_success, r.verification_attempted),
                    width = col_width
                );
            } else {
                print!(" | {:<width$}", "-", width = col_width);
            }
        }
        println!(" | -");

        // Avg Verification Latency
        print_row!(
            "Avg Verify (ms)",
            results
                .iter()
                .map(|r| {
                    if r.verification_time_values.is_empty() {
                        0.0
                    } else {
                        mean(&r.verification_time_values)
                    }
                })
                .collect(),
            |v: f64| if v > 0.0 {
                format!("{:.0} ms", v)
            } else {
                "-".to_string()
            },
            true
        );
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

    // Avg Total Request Duration
    print_row!(
        "Avg Total Duration (ms)",
        results.iter().map(|r| r.avg_request_duration()).collect(),
        |v: f64| format!("{:.0} ms", v),
        true
    );

    // P95 Total Request Duration
    print_row!(
        "P95 Total Duration (ms)",
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
    // Match ${VAR} or ${VAR:-default}
    let re = regex::Regex::new(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}").unwrap();
    let mut result = s.to_string();
    let mut missing_vars = Vec::new();

    for caps in re.captures_iter(s) {
        let full_match = caps.get(0).unwrap().as_str();
        let var_name = caps.get(1).unwrap().as_str();
        let default_value = caps.get(2).map(|m| m.as_str());

        match std::env::var(var_name) {
            Ok(value) => {
                result = result.replace(full_match, &value);
            }
            Err(_) => {
                if let Some(default) = default_value {
                    result = result.replace(full_match, default);
                } else {
                    missing_vars.push(var_name.to_string());
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

    // Write sample prompts
    if !result.sample_prompts.is_empty() {
        writeln!(file)?;
        writeln!(file, "Sample prompts used (first 20 tokens):")?;
        for (i, prompt) in result.sample_prompts.iter().enumerate() {
            writeln!(file, "  {}. {}", i + 1, prompt)?;
        }
        writeln!(file)?;
    }

    // Write TEE verification stats if any verification was attempted
    if result.verification_attempted > 0 {
        writeln!(
            file,
            "TEE verification:                        {}/{}",
            result.verification_success, result.verification_attempted
        )?;
    }

    // Write request type
    match result.request_type {
        RequestType::ImageGeneration => {
            writeln!(
                file,
                "Request type:                            Image Generation"
            )?;
        }
        RequestType::ChatCompletion => {
            writeln!(
                file,
                "Request type:                            Chat Completion"
            )?;
        }
    }

    // Write audio metrics if any
    if result.audio_input_requests > 0 || result.audio_output_requests > 0 {
        writeln!(file)?;
        writeln!(file, "-----------Audio Metrics--------------")?;
        if result.audio_input_requests > 0 {
            writeln!(
                file,
                "Requests with audio input:               {}",
                result.audio_input_requests
            )?;
            if result.total_audio_input_bytes > 0 {
                writeln!(
                    file,
                    "Total audio input:                       {:.2} KB",
                    result.total_audio_input_bytes as f64 / 1024.0
                )?;
            }
        }
        if result.audio_output_requests > 0 {
            writeln!(
                file,
                "Requests with audio output:              {}",
                result.audio_output_requests
            )?;
            if result.total_audio_output_bytes > 0 {
                writeln!(
                    file,
                    "Total audio output:                      {:.2} KB",
                    result.total_audio_output_bytes as f64 / 1024.0
                )?;
            }
        }
    }

    // Write image generation metrics if any
    if result.request_type == RequestType::ImageGeneration {
        writeln!(file)?;
        writeln!(file, "-----------Image Generation Metrics-----------")?;
        writeln!(
            file,
            "Total images generated:                  {}",
            result.total_images_generated
        )?;
        if result.total_image_bytes > 0 {
            writeln!(
                file,
                "Total image data:                        {:.2} MB",
                result.total_image_bytes as f64 / (1024.0 * 1024.0)
            )?;
            writeln!(
                file,
                "Avg image size:                          {:.2} KB",
                (result.total_image_bytes as f64 / result.total_images_generated.max(1) as f64)
                    / 1024.0
            )?;
        }
        if !result.image_generation_time_values.is_empty() {
            writeln!(
                file,
                "Mean generation time (ms):               {:.2}",
                mean(&result.image_generation_time_values)
            )?;
            writeln!(
                file,
                "P95 generation time (ms):                {:.2}",
                percentile(&result.image_generation_time_values, 95.0)
            )?;
        }
    }

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
        writeln!(file, "-----------Total Request Duration----------------")?;
        writeln!(
            file,
            "Mean Total Duration (ms):                 {:.2}",
            mean(&result.request_duration_values)
        )?;
        writeln!(
            file,
            "Median Total Duration (ms):               {:.2}",
            median(&result.request_duration_values)
        )?;
        writeln!(
            file,
            "P90 Total Duration (ms):                  {:.2}",
            percentile(&result.request_duration_values, 90.0)
        )?;
        writeln!(
            file,
            "P95 Total Duration (ms):                  {:.2}",
            percentile(&result.request_duration_values, 95.0)
        )?;
        writeln!(
            file,
            "P99 Total Duration (ms):                  {:.2}",
            percentile(&result.request_duration_values, 99.0)
        )?;
        writeln!(
            file,
            "P100 Total Duration (ms):                 {:.2}",
            percentile(&result.request_duration_values, 100.0)
        )?;
    }

    if !result.verification_time_values.is_empty() {
        writeln!(file, "-----------TEE Verification Latency---------------")?;
        writeln!(
            file,
            "Mean Verification (ms):                  {:.2}",
            mean(&result.verification_time_values)
        )?;
        writeln!(
            file,
            "Median Verification (ms):                {:.2}",
            median(&result.verification_time_values)
        )?;
        writeln!(
            file,
            "P90 Verification (ms):                   {:.2}",
            percentile(&result.verification_time_values, 90.0)
        )?;
        writeln!(
            file,
            "P95 Verification (ms):                   {:.2}",
            percentile(&result.verification_time_values, 95.0)
        )?;
        writeln!(
            file,
            "P99 Verification (ms):                   {:.2}",
            percentile(&result.verification_time_values, 99.0)
        )?;
        writeln!(
            file,
            "P100 Verification (ms):                  {:.2}",
            percentile(&result.verification_time_values, 100.0)
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

    // TEE Verification (only show if any provider has verification enabled)
    let any_verification = results.iter().any(|r| r.verification_attempted > 0);
    if any_verification {
        write!(file, "{:<width$}", "TEE Verification", width = metric_width)?;
        for r in results {
            if r.verification_attempted > 0 {
                write!(
                    file,
                    " | {:<width$}",
                    format!("{}/{}", r.verification_success, r.verification_attempted),
                    width = col_width
                )?;
            } else {
                write!(file, " | {:<width$}", "-", width = col_width)?;
            }
        }
        writeln!(file, " | -")?;

        // Avg Verification Latency
        write_row!(
            "Avg Verify (ms)",
            results
                .iter()
                .map(|r| {
                    if r.verification_time_values.is_empty() {
                        0.0
                    } else {
                        mean(&r.verification_time_values)
                    }
                })
                .collect(),
            |v: f64| if v > 0.0 {
                format!("{:.0} ms", v)
            } else {
                "-".to_string()
            },
            true
        );
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
        "Avg Total Duration (ms)",
        results.iter().map(|r| r.avg_request_duration()).collect(),
        |v: f64| format!("{:.0} ms", v),
        true
    );
    write_row!(
        "P95 Total Duration (ms)",
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
