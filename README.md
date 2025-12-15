# GenAI Benchmark

A high-performance load testing tool for OpenAI-compatible LLM APIs, written in Rust.

## Features

- **OpenAI-compatible API support**: Works with any OpenAI-compatible endpoint (vLLM, TGI, Ollama, etc.)
- **Streaming metrics**: Measures Time to First Token (TTFT), Inter-token Latency (ITL), and Time per Output Token (TPOT)
- **Built-in scenarios**: Pre-configured benchmarks included in the binary
- **Provider comparison**: Test the same model across multiple providers and compare results
- **Detailed statistics**: P50, P90, P95, P99, P100 percentiles for all metrics

## Quick Start

### List available scenarios
```bash
genai-benchmark list
```

### Describe a scenario (see config and required env vars)
```bash
genai-benchmark describe near-vs-bedrock
```

### Run a scenario
```bash
export NEAR_API_KEY=your-key
export AWS_BEARER_TOKEN_BEDROCK=your-token
genai-benchmark run near-vs-bedrock
```

### Export and customize a scenario
```bash
genai-benchmark export near-vs-bedrock > my-benchmark.yaml
# Edit my-benchmark.yaml
genai-benchmark scenario my-benchmark.yaml
```

## Installation

**One-liner install:**
```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/nearai/genai-benchmark/releases/latest/download/genai-benchmark-installer.sh | sh
```

Or download pre-built binaries from [Releases](https://github.com/nearai/genai-benchmark/releases), or build from source:

```bash
cargo install --path .
```

## Library Usage

```rust
use genai_benchmark::{BenchmarkConfig, run_benchmark, load_dataset, DatasetConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = BenchmarkConfig {
        name: Some("My Test".to_string()),
        base_url: "https://api.example.com/v1".to_string(),
        api_key: "your-key".to_string(),
        model: "gpt-4".to_string(),
        max_tokens: 256,
        concurrency: 5,
        rps: 10.0,
        timeout_secs: 300,
    };

    let dataset = DatasetConfig::Synthetic { seed: Some(42) };
    let prompts = load_dataset(&dataset, 100).await?;

    let result = run_benchmark(&config, prompts, 100).await?;
    genai_benchmark::print_result(&result);

    Ok(())
}
```

## License

MIT
