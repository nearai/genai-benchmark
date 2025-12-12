# GenAI Benchmark

A high-performance load testing tool for OpenAI-compatible LLM APIs, written in Rust.

## Features

- **OpenAI-compatible API support**: Works with any OpenAI-compatible endpoint (vLLM, TGI, Ollama, NEAR AI, etc.)
- **Streaming metrics**: Measures Time to First Token (TTFT), Inter-token Latency (ITL), and Time per Output Token (TPOT)
- **YAML scenario files**: Define complex multi-provider benchmarks in YAML
- **Provider comparison**: Test the same model across multiple providers and compare results
- **Hugging Face dataset support**: Load prompts from ShareGPT and other datasets
- **Configurable concurrency**: Control parallel requests and rate limiting
- **Detailed statistics**: P50, P90, P95, P99, P100 percentiles for all metrics

## Usage

```
cargo run -- scenario scenarios/near_vs_bedrock.yaml
```

## Library Usage

You can also use `genai-benchmark` as a library in your Rust projects:

```rust
use genai_benchmark::{BenchmarkConfig, run_benchmark, load_dataset, DatasetConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = BenchmarkConfig {
        name: Some("My Test".to_string()),
        base_url: "https://api.example.com/v1".to_string(),
        api_key: "sk-xxx".to_string(),
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
