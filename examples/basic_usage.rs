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
        disable_prewarm: false,
        verify: false,
        random_prompt_selection: false,
        random_seed: None,
    };

    let dataset = DatasetConfig::Synthetic { seed: Some(42) };
    let prompts = load_dataset(&dataset, 100).await?;

    let result = run_benchmark(&config, prompts, 100).await?;
    genai_benchmark::print_result(&result);

    Ok(())
}
