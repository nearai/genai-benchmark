use genai_benchmark::{
    run_multi_phase_benchmark, load_dataset, BenchmarkConfig, BenchmarkPhase, PhaseConfig,
    DatasetConfig, ConversationManager,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure the benchmark
    let config = BenchmarkConfig {
        name: Some("Multi-Phase Benchmark".to_string()),
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

    // Load prompts
    let dataset = DatasetConfig::Synthetic { seed: Some(42) };
    let prompts = load_dataset(&dataset, 200).await?;

    // Define phases: warmup + query
    let phases = vec![
        PhaseConfig {
            phase: BenchmarkPhase::Warmup,
            num_requests: 20,
            concurrency: Some(2),
            rps: Some(5.0),
        },
        PhaseConfig {
            phase: BenchmarkPhase::Query,
            num_requests: 100,
            concurrency: Some(5),
            rps: Some(10.0),
        },
    ];

    // Run multi-phase benchmark with conversation manager for multi-turn support
    let conversation_manager = ConversationManager::new();
    let results = run_multi_phase_benchmark(&config, prompts, phases, Some(&conversation_manager))
        .await?;

    // Print aggregated results
    for result in results {
        genai_benchmark::print_result(&result);
    }

    Ok(())
}
