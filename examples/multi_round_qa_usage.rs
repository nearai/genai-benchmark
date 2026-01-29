use genai_benchmark::{
    run_benchmark, load_dataset, BenchmarkConfig, DatasetConfig, ConversationManager, Message,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure the benchmark for multi-round QA
    let config = BenchmarkConfig {
        name: Some("Multi-Round QA".to_string()),
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

    // Load multi-round QA dataset
    let dataset = DatasetConfig::MultiRoundQa {
        path: "./datasets/multi_round_qa.jsonl".to_string(),
        num_rounds: 4,
        users_per_round: Some(vec![1, 2, 4, 8]),
    };

    let prompts = load_dataset(&dataset, 100).await?;

    // Create conversation manager for tracking multi-turn state
    let conversation_manager = ConversationManager::new();

    // Simulate conversation with different users
    for (i, prompt) in prompts.iter().take(10).enumerate() {
        let user_id = format!("user_{}", i % 5);

        // Get existing conversation history
        let history = conversation_manager.get_history(&user_id).await;
        println!(
            "User {} conversation history length: {}",
            user_id,
            history.len()
        );

        // Append this turn's message
        if let Some(msg) = prompt.first() {
            conversation_manager
                .append_message(&user_id, msg.clone())
                .await;
        }

        // Move to next round
        conversation_manager.next_round(&user_id).await;
        let round = conversation_manager.get_round(&user_id).await;
        println!("User {} is now on round {}", user_id, round);
    }

    // Run actual benchmark with conversation support
    let result = run_benchmark(&config, prompts, 100).await?;
    genai_benchmark::print_result(&result);

    // Print cache metrics if available
    println!("\nCache Metrics:");
    println!(
        "  Cache hits: {} / {} ({:.1}%)",
        result.cache_hits,
        result.cache_hits + result.cache_misses,
        result.avg_cache_hit_rate * 100.0
    );
    println!("  Token savings: {}", result.cache_token_savings);

    Ok(())
}
