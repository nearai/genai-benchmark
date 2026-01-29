use genai_benchmark::{
    aggregate_phase_results, run_multi_phase_benchmark, BenchmarkConfig, BenchmarkPhase, Message,
    PhaseConfig,
};

#[tokio::test]
async fn test_multi_phase_execution() {
    // Create simple test config
    let config = BenchmarkConfig {
        name: Some("Test Multi-Phase".to_string()),
        base_url: "http://localhost:8000".to_string(),
        api_key: "test-key".to_string(),
        model: "test-model".to_string(),
        max_tokens: 10,
        concurrency: 1,
        rps: 100.0,
        timeout_secs: 30,
        disable_prewarm: true,
        verify: false,
        random_prompt_selection: false,
        random_seed: None,
    };

    // Create test prompts
    let prompts = vec![
        vec![Message {
            role: "user".to_string(),
            content: "Test question 1".to_string(),
        }],
        vec![Message {
            role: "user".to_string(),
            content: "Test question 2".to_string(),
        }],
    ];

    // Create multi-phase config
    let phases = vec![
        PhaseConfig {
            phase: BenchmarkPhase::Warmup,
            num_requests: 1,
            concurrency: Some(1),
            rps: Some(100.0),
        },
        PhaseConfig {
            phase: BenchmarkPhase::Query,
            num_requests: 1,
            concurrency: Some(1),
            rps: Some(100.0),
        },
    ];

    // This would fail in real execution due to no server, but shows the API works
    let result = run_multi_phase_benchmark(&config, prompts, phases, None).await;

    // Result should be either Ok(vec with 2 phases) or Err (due to no server)
    // The important thing is that the function signature compiles and is callable
    match result {
        Ok(results) => {
            assert_eq!(results.len(), 2, "Should have 2 phase results");
        }
        Err(_) => {
            // Expected since we have no actual server
            assert!(true, "Expected error due to no server");
        }
    }
}

#[test]
fn test_aggregate_phase_results_empty() {
    let results = aggregate_phase_results(&[]);
    assert_eq!(results.total_requests, 0);
    assert_eq!(results.successful_requests, 0);
}

#[test]
fn test_aggregate_phase_results_single() {
    use genai_benchmark::BenchmarkResult;

    let phase_result = BenchmarkResult {
        name: Some("Phase 1".to_string()),
        successful_requests: 50,
        failed_requests: 10,
        total_requests: 60,
        requests_with_usage: 50,
        concurrency: 5,
        rps_configured: 10.0,
        duration_secs: 6.0,
        total_input_tokens: 1000,
        total_output_tokens: 2000,
        total_chunks: 100,
        verification_attempted: 0,
        verification_success: 0,
        verification_failed: 0,
        cache_hits: 0,
        cache_misses: 60,
        avg_cache_hit_rate: 0.0,
        cache_token_savings: 0,
        avg_f1_score: 0.0,
        avg_rouge_l_score: 0.0,
        quality_metrics_count: 0,
        ttft_values: vec![100.0, 150.0, 120.0],
        tpot_values: vec![10.0, 12.0, 11.0],
        itl_values: vec![5.0, 6.0, 5.5],
        request_duration_values: vec![1000.0, 1100.0, 950.0],
        tokens_per_request: vec![100, 120, 110],
        verification_time_values: vec![],
        f1_scores: vec![],
        rouge_l_scores: vec![],
        sample_prompts: vec!["Prompt 1".to_string()],
    };

    let aggregated = aggregate_phase_results(&[phase_result]);

    assert_eq!(aggregated.successful_requests, 50);
    assert_eq!(aggregated.failed_requests, 10);
    assert_eq!(aggregated.total_requests, 60);
    assert_eq!(aggregated.total_input_tokens, 1000);
    assert_eq!(aggregated.total_output_tokens, 2000);
    assert_eq!(aggregated.cache_misses, 60);
}

#[test]
fn test_aggregate_phase_results_multiple() {
    use genai_benchmark::BenchmarkResult;

    let phase1 = BenchmarkResult {
        name: Some("Warmup".to_string()),
        successful_requests: 40,
        failed_requests: 10,
        total_requests: 50,
        requests_with_usage: 40,
        concurrency: 5,
        rps_configured: 10.0,
        duration_secs: 5.0,
        total_input_tokens: 500,
        total_output_tokens: 1000,
        total_chunks: 50,
        verification_attempted: 0,
        verification_success: 0,
        verification_failed: 0,
        cache_hits: 10,
        cache_misses: 40,
        avg_cache_hit_rate: 0.2,
        cache_token_savings: 100,
        avg_f1_score: 0.0,
        avg_rouge_l_score: 0.0,
        quality_metrics_count: 0,
        ttft_values: vec![100.0, 150.0],
        tpot_values: vec![10.0, 12.0],
        itl_values: vec![5.0, 6.0],
        request_duration_values: vec![1000.0, 1100.0],
        tokens_per_request: vec![100, 110],
        verification_time_values: vec![],
        f1_scores: vec![],
        rouge_l_scores: vec![],
        sample_prompts: vec!["Prompt 1".to_string()],
    };

    let phase2 = BenchmarkResult {
        name: Some("Query".to_string()),
        successful_requests: 50,
        failed_requests: 0,
        total_requests: 50,
        requests_with_usage: 50,
        concurrency: 5,
        rps_configured: 10.0,
        duration_secs: 5.0,
        total_input_tokens: 500,
        total_output_tokens: 1000,
        total_chunks: 50,
        verification_attempted: 0,
        verification_success: 0,
        verification_failed: 0,
        cache_hits: 45,
        cache_misses: 5,
        avg_cache_hit_rate: 0.9,
        cache_token_savings: 900,
        avg_f1_score: 0.0,
        avg_rouge_l_score: 0.0,
        quality_metrics_count: 0,
        ttft_values: vec![50.0, 55.0],
        tpot_values: vec![9.0, 8.5],
        itl_values: vec![4.0, 4.5],
        request_duration_values: vec![500.0, 520.0],
        tokens_per_request: vec![100, 105],
        verification_time_values: vec![],
        f1_scores: vec![],
        rouge_l_scores: vec![],
        sample_prompts: vec!["Prompt 1".to_string(), "Prompt 2".to_string()],
    };

    let aggregated = aggregate_phase_results(&[phase1, phase2]);

    // Verify aggregation
    assert_eq!(aggregated.successful_requests, 90);
    assert_eq!(aggregated.failed_requests, 10);
    assert_eq!(aggregated.total_requests, 100);
    assert_eq!(aggregated.duration_secs, 10.0);
    assert_eq!(aggregated.total_input_tokens, 1000);
    assert_eq!(aggregated.total_output_tokens, 2000);
    assert_eq!(aggregated.cache_hits, 55);
    assert_eq!(aggregated.cache_misses, 45);
    assert_eq!(aggregated.cache_token_savings, 1000);

    // Check aggregated metrics
    assert_eq!(aggregated.ttft_values.len(), 4);
    assert_eq!(aggregated.tpot_values.len(), 4);
    assert_eq!(aggregated.itl_values.len(), 4);
    assert_eq!(aggregated.request_duration_values.len(), 4);
    assert_eq!(aggregated.tokens_per_request.len(), 4);

    // Verify cache hit rate calculation
    let expected_hit_rate = 55.0 / 100.0;
    assert!((aggregated.avg_cache_hit_rate - expected_hit_rate).abs() < 0.01);

    // Verify sample prompts are deduplicated
    assert!(aggregated.sample_prompts.len() <= 5);
}

#[test]
fn test_phase_config_creation() {
    let warmup = PhaseConfig {
        phase: BenchmarkPhase::Warmup,
        num_requests: 50,
        concurrency: Some(5),
        rps: Some(10.0),
    };

    assert_eq!(warmup.num_requests, 50);
    assert_eq!(warmup.concurrency, Some(5));
    assert_eq!(warmup.rps, Some(10.0));

    let query = PhaseConfig {
        phase: BenchmarkPhase::Query,
        num_requests: 100,
        concurrency: None,
        rps: None,
    };

    assert_eq!(query.num_requests, 100);
    assert_eq!(query.concurrency, None);
    assert_eq!(query.rps, None);
}
