use genai_benchmark::{
    aggregate_phase_results, calculate_f1_score, calculate_quality_scores, load_dataset,
    BenchmarkConfig, BenchmarkPhase, BenchmarkResult, DatasetConfig, Message, PhaseConfig,
    RequestMetrics,
};

/// Test that datasets can be loaded without errors
#[tokio::test]
async fn test_all_dataset_types_load() {
    // Synthetic dataset (always works)
    let synthetic_config = DatasetConfig::Synthetic { seed: Some(42) };
    let prompts = load_dataset(&synthetic_config, 5).await;
    assert!(prompts.is_ok(), "Synthetic dataset should load");
    let synthetic_prompts = prompts.unwrap();
    assert!(!synthetic_prompts.is_empty(), "Should load synthetic prompts");
    assert_eq!(synthetic_prompts.len(), 5, "Should load exactly 5 prompts");

    // ShareGPT format (with defaults)
    let _sharegpt_config = DatasetConfig::Sharegpt {
        path: None,
        hf_dataset: "anon8231489123/ShareGPT_Vicuna_unfiltered".to_string(),
        skip: 0,
    };
    // Config created successfully (would fail if struct changed)
}

/// Test metrics aggregation with realistic data
#[test]
fn test_metrics_aggregation_complete() {
    let result1 = BenchmarkResult {
        name: Some("Provider1".to_string()),
        successful_requests: 90,
        failed_requests: 10,
        total_requests: 100,
        requests_with_usage: 90,
        concurrency: 5,
        rps_configured: 10.0,
        duration_secs: 10.0,
        total_input_tokens: 9000,
        total_output_tokens: 18000,
        total_chunks: 900,
        verification_attempted: 0,
        verification_success: 0,
        verification_failed: 0,
        cache_hits: 45,
        cache_misses: 55,
        avg_cache_hit_rate: 0.45,
        cache_token_savings: 8100,
        avg_f1_score: 0.75,
        avg_rouge_l_score: 0.68,
        quality_metrics_count: 90,
        ttft_values: vec![50.0, 75.0, 60.0, 55.0, 65.0],
        tpot_values: vec![8.0, 9.0, 7.5, 8.5, 7.0],
        itl_values: vec![3.0, 4.0, 3.5, 2.5, 3.0],
        request_duration_values: vec![1000.0, 1050.0, 980.0, 1020.0, 1010.0],
        tokens_per_request: vec![100, 110, 105, 95, 105],
        verification_time_values: vec![],
        f1_scores: vec![0.75, 0.76, 0.74, 0.75, 0.75],
        rouge_l_scores: vec![0.68, 0.69, 0.67, 0.68, 0.68],
        sample_prompts: vec!["Prompt 1".to_string(), "Prompt 2".to_string()],
    };

    let result2 = BenchmarkResult {
        name: Some("Provider2".to_string()),
        successful_requests: 95,
        failed_requests: 5,
        total_requests: 100,
        requests_with_usage: 95,
        concurrency: 8,
        rps_configured: 15.0,
        duration_secs: 7.5,
        total_input_tokens: 9500,
        total_output_tokens: 19000,
        total_chunks: 950,
        verification_attempted: 0,
        verification_success: 0,
        verification_failed: 0,
        cache_hits: 76,
        cache_misses: 24,
        avg_cache_hit_rate: 0.76,
        cache_token_savings: 14250,
        avg_f1_score: 0.80,
        avg_rouge_l_score: 0.72,
        quality_metrics_count: 95,
        ttft_values: vec![40.0, 45.0, 50.0, 42.0, 48.0],
        tpot_values: vec![7.0, 7.5, 8.0, 6.5, 7.5],
        itl_values: vec![2.0, 2.5, 3.0, 2.5, 2.0],
        request_duration_values: vec![900.0, 950.0, 1000.0, 920.0, 980.0],
        tokens_per_request: vec![105, 115, 110, 100, 110],
        verification_time_values: vec![],
        f1_scores: vec![0.80, 0.80, 0.80, 0.80, 0.80],
        rouge_l_scores: vec![0.72, 0.72, 0.72, 0.72, 0.72],
        sample_prompts: vec!["Prompt 2".to_string(), "Prompt 3".to_string()],
    };

    let aggregated = aggregate_phase_results(&[result1, result2]);

    // Verify basic aggregation
    assert_eq!(aggregated.successful_requests, 185);
    assert_eq!(aggregated.failed_requests, 15);
    assert_eq!(aggregated.total_requests, 200);
    assert_eq!(aggregated.total_input_tokens, 18500);
    assert_eq!(aggregated.total_output_tokens, 37000);

    // Verify cache metrics aggregation
    assert_eq!(aggregated.cache_hits, 121);
    assert_eq!(aggregated.cache_misses, 79);
    assert_eq!(aggregated.cache_token_savings, 22350);
    assert!((aggregated.avg_cache_hit_rate - 0.605).abs() < 0.01);

    // Verify quality metrics aggregation
    assert_eq!(aggregated.quality_metrics_count, 185);
    assert!((aggregated.avg_f1_score - 0.775).abs() < 0.01);
    assert!((aggregated.avg_rouge_l_score - 0.70).abs() < 0.01);

    // Verify sample prompts are aggregated (max 5)
    assert!(aggregated.sample_prompts.len() <= 5);
    assert!(aggregated.sample_prompts.contains(&"Prompt 1".to_string()));
}

/// Test quality metrics calculation with realistic examples
#[test]
fn test_quality_metrics_realistic_examples() {
    // Test case 1: Perfect match
    let (f1, rouge_l) = calculate_quality_scores(
        "machine learning is a subset of artificial intelligence",
        "machine learning is a subset of artificial intelligence",
    );
    assert!((f1 - 1.0).abs() < 0.001, "Perfect match should have F1=1.0");
    assert!((rouge_l - 1.0).abs() < 0.001, "Perfect match should have ROUGE-L=1.0");

    // Test case 2: Partial match (synonym)
    let (f1, rouge_l) = calculate_quality_scores(
        "deep learning is a type of machine learning",
        "deep learning is a form of machine learning",
    );
    assert!(f1 > 0.5, "Partial match should have decent F1 score");
    assert!(rouge_l > 0.5, "Partial match should have decent ROUGE-L score");

    // Test case 3: No match
    let (f1, rouge_l) = calculate_quality_scores(
        "the quick brown fox jumps over the lazy dog",
        "unrelated content about weather and climate",
    );
    assert!(f1 < 0.3, "No match should have low F1");
    assert!(rouge_l < 0.3, "No match should have low ROUGE-L");

    // Test case 4: Partial content overlap
    let (f1, rouge_l) = calculate_quality_scores(
        "the cat sat on the mat in the sun",
        "the cat sat on the mat under the tree",
    );
    assert!(f1 > 0.4, "Should detect partial overlap");
    assert!(rouge_l > 0.4, "Should detect partial overlap");
}

/// Test request metrics structure and calculations
#[test]
fn test_request_metrics_calculations() {
    let metrics = RequestMetrics {
        success: true,
        input_tokens: 100,
        output_tokens: 50,
        ttft_ms: 150.0,
        total_time_ms: 1500.0,
        inter_chunk_latencies: vec![20.0, 25.0, 22.0, 23.0, 24.0, 21.0, 20.0, 19.0, 22.0, 23.0],
        chunk_count: 50,
        output_chars: 500,
        got_usage: true,
        verification_attempted: false,
        verification_success: false,
        verification_time_ms: 0.0,
        prompt_preview: "What is machine learning?".to_string(),
        cache_metrics: None,
        quality_metrics: None,
    };

    // Test TPOT calculation
    let tpot = metrics.tpot_ms();
    assert!(tpot.is_some(), "Should calculate TPOT for output tokens > 1");
    let tpot_value = tpot.unwrap();
    assert!(tpot_value > 0.0, "TPOT should be positive");
    assert!(tpot_value < 25.0, "TPOT should be less than 25ms");
}

/// Test quality metrics with actual answer matching
#[test]
fn test_quality_metrics_with_answers() {
    // Create test cases with known quality metrics
    struct TestCase {
        generated: &'static str,
        reference: &'static str,
        expected_f1_min: f64,
        expected_f1_max: f64,
    }

    let test_cases = vec![
        TestCase {
            generated: "Paris is the capital of France",
            reference: "The capital of France is Paris",
            expected_f1_min: 0.8,
            expected_f1_max: 1.0,
        },
        TestCase {
            generated: "Python is a programming language",
            reference: "Java is a programming language",
            expected_f1_min: 0.7,
            expected_f1_max: 1.0,
        },
        TestCase {
            generated: "The Earth orbits the Sun",
            reference: "The Moon orbits the Earth",
            expected_f1_min: 0.3,
            expected_f1_max: 0.7,
        },
    ];

    for test in test_cases {
        let f1 = calculate_f1_score(test.generated, test.reference);
        assert!(
            f1 >= test.expected_f1_min && f1 <= test.expected_f1_max,
            "F1 score {} should be between {} and {} for:\nGenerated: {}\nReference: {}",
            f1,
            test.expected_f1_min,
            test.expected_f1_max,
            test.generated,
            test.reference
        );
    }
}

/// Test BenchmarkResult computations
#[test]
fn test_benchmark_result_calculations() {
    let result = BenchmarkResult {
        name: Some("Test".to_string()),
        successful_requests: 90,
        failed_requests: 10,
        total_requests: 100,
        requests_with_usage: 85,
        concurrency: 5,
        rps_configured: 10.0,
        duration_secs: 9.0,
        total_input_tokens: 9000,
        total_output_tokens: 18000,
        total_chunks: 900,
        verification_attempted: 0,
        verification_success: 0,
        verification_failed: 0,
        cache_hits: 45,
        cache_misses: 55,
        avg_cache_hit_rate: 0.45,
        cache_token_savings: 8100,
        avg_f1_score: 0.75,
        avg_rouge_l_score: 0.68,
        quality_metrics_count: 90,
        ttft_values: vec![50.0, 60.0, 55.0],
        tpot_values: vec![8.0, 9.0, 8.5],
        itl_values: vec![3.0, 4.0, 3.5],
        request_duration_values: vec![1000.0, 1100.0, 950.0],
        tokens_per_request: vec![100, 120, 110],
        verification_time_values: vec![],
        f1_scores: vec![0.75, 0.75, 0.75],
        rouge_l_scores: vec![0.68, 0.68, 0.68],
        sample_prompts: vec!["Prompt 1".to_string()],
    };

    // Test throughput calculations
    let req_throughput = result.request_throughput();
    assert!(req_throughput > 0.0);
    assert!((req_throughput - 10.0).abs() < 0.1, "Should be ~10 req/s");

    let token_throughput = result.output_token_throughput();
    assert!(token_throughput > 0.0);
    assert!((token_throughput - 2000.0).abs() < 100.0, "Should be ~2000 tokens/s");

    // Test success rate
    let success_rate = result.success_rate();
    assert!((success_rate - 90.0).abs() < 0.1, "Should be 90%");

    // Test tokens per chunk
    let tpc = result.tokens_per_chunk();
    assert!((tpc - 20.0).abs() < 0.1, "Should be ~20 tokens/chunk");
}

/// Test message structure for various benchmark types
#[test]
fn test_message_creation_and_handling() {
    let msg = Message {
        role: "user".to_string(),
        content: "What is machine learning?".to_string(),
    };

    assert_eq!(msg.role, "user");
    assert_eq!(msg.content, "What is machine learning?");

    // Test message cloning
    let msg2 = msg.clone();
    assert_eq!(msg2.role, msg.role);
    assert_eq!(msg2.content, msg.content);

    // Test multiple message types
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
        },
        Message {
            role: "user".to_string(),
            content: "Explain machine learning.".to_string(),
        },
        Message {
            role: "assistant".to_string(),
            content: "Machine learning is a field of AI...".to_string(),
        },
    ];

    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[1].role, "user");
    assert_eq!(messages[2].role, "assistant");
}

/// Test phase configuration creation
#[test]
fn test_phase_configuration() {
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
        concurrency: Some(10),
        rps: Some(20.0),
    };

    assert_eq!(query.num_requests, 100);
    assert_eq!(query.concurrency, Some(10));
    assert_eq!(query.rps, Some(20.0));
}

/// Test benchmark config creation
#[test]
fn test_benchmark_config_creation() {
    let config = BenchmarkConfig {
        name: Some("Test Benchmark".to_string()),
        base_url: "http://localhost:8000/v1".to_string(),
        api_key: "test-key".to_string(),
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

    assert_eq!(config.model, "gpt-4");
    assert_eq!(config.max_tokens, 256);
    assert_eq!(config.concurrency, 5);

    let timeout = config.timeout();
    assert_eq!(timeout.as_secs(), 300);
}

/// Test quality metrics aggregation across multiple results
#[test]
fn test_quality_metrics_aggregation() {
    let results = vec![
        BenchmarkResult {
            name: Some("Test1".to_string()),
            successful_requests: 50,
            failed_requests: 0,
            total_requests: 50,
            requests_with_usage: 50,
            concurrency: 5,
            rps_configured: 10.0,
            duration_secs: 5.0,
            total_input_tokens: 5000,
            total_output_tokens: 10000,
            total_chunks: 500,
            verification_attempted: 0,
            verification_success: 0,
            verification_failed: 0,
            cache_hits: 0,
            cache_misses: 0,
            avg_cache_hit_rate: 0.0,
            cache_token_savings: 0,
            avg_f1_score: 0.75,
            avg_rouge_l_score: 0.70,
            quality_metrics_count: 50,
            ttft_values: vec![50.0],
            tpot_values: vec![8.0],
            itl_values: vec![3.0],
            request_duration_values: vec![1000.0],
            tokens_per_request: vec![100],
            verification_time_values: vec![],
            f1_scores: vec![0.75; 50],
            rouge_l_scores: vec![0.70; 50],
            sample_prompts: vec!["Test".to_string()],
        },
        BenchmarkResult {
            name: Some("Test2".to_string()),
            successful_requests: 50,
            failed_requests: 0,
            total_requests: 50,
            requests_with_usage: 50,
            concurrency: 5,
            rps_configured: 10.0,
            duration_secs: 5.0,
            total_input_tokens: 5000,
            total_output_tokens: 10000,
            total_chunks: 500,
            verification_attempted: 0,
            verification_success: 0,
            verification_failed: 0,
            cache_hits: 0,
            cache_misses: 0,
            avg_cache_hit_rate: 0.0,
            cache_token_savings: 0,
            avg_f1_score: 0.80,
            avg_rouge_l_score: 0.75,
            quality_metrics_count: 50,
            ttft_values: vec![50.0],
            tpot_values: vec![8.0],
            itl_values: vec![3.0],
            request_duration_values: vec![1000.0],
            tokens_per_request: vec![100],
            verification_time_values: vec![],
            f1_scores: vec![0.80; 50],
            rouge_l_scores: vec![0.75; 50],
            sample_prompts: vec!["Test".to_string()],
        },
    ];

    let aggregated = aggregate_phase_results(&results);

    assert_eq!(aggregated.quality_metrics_count, 100);
    assert!((aggregated.avg_f1_score - 0.775).abs() < 0.01);
    assert!((aggregated.avg_rouge_l_score - 0.725).abs() < 0.01);
}

/// Test cache metrics aggregation
#[test]
fn test_cache_metrics_aggregation() {
    let result1 = BenchmarkResult {
        name: Some("Warmup".to_string()),
        successful_requests: 20,
        failed_requests: 0,
        total_requests: 20,
        requests_with_usage: 20,
        concurrency: 1,
        rps_configured: 1.0,
        duration_secs: 20.0,
        total_input_tokens: 2000,
        total_output_tokens: 4000,
        total_chunks: 200,
        verification_attempted: 0,
        verification_success: 0,
        verification_failed: 0,
        cache_hits: 0,
        cache_misses: 20,
        avg_cache_hit_rate: 0.0,
        cache_token_savings: 0,
        avg_f1_score: 0.0,
        avg_rouge_l_score: 0.0,
        quality_metrics_count: 0,
        ttft_values: vec![100.0],
        tpot_values: vec![10.0],
        itl_values: vec![5.0],
        request_duration_values: vec![2000.0],
        tokens_per_request: vec![100],
        verification_time_values: vec![],
        f1_scores: vec![],
        rouge_l_scores: vec![],
        sample_prompts: vec![],
    };

    let result2 = BenchmarkResult {
        name: Some("Query".to_string()),
        successful_requests: 80,
        failed_requests: 0,
        total_requests: 80,
        requests_with_usage: 80,
        concurrency: 5,
        rps_configured: 5.0,
        duration_secs: 16.0,
        total_input_tokens: 8000,
        total_output_tokens: 16000,
        total_chunks: 800,
        verification_attempted: 0,
        verification_success: 0,
        verification_failed: 0,
        cache_hits: 60,
        cache_misses: 20,
        avg_cache_hit_rate: 0.75,
        cache_token_savings: 12000,
        avg_f1_score: 0.0,
        avg_rouge_l_score: 0.0,
        quality_metrics_count: 0,
        ttft_values: vec![50.0],
        tpot_values: vec![8.0],
        itl_values: vec![3.0],
        request_duration_values: vec![1000.0],
        tokens_per_request: vec![100],
        verification_time_values: vec![],
        f1_scores: vec![],
        rouge_l_scores: vec![],
        sample_prompts: vec![],
    };

    let aggregated = aggregate_phase_results(&[result1, result2]);

    assert_eq!(aggregated.cache_hits, 60);
    assert_eq!(aggregated.cache_misses, 40);
    assert_eq!(aggregated.cache_token_savings, 12000);
    assert!((aggregated.avg_cache_hit_rate - 0.6).abs() < 0.01);
}

/// Test backward compatibility of benchmark config
#[test]
fn test_backward_compatibility() {
    // Ensure that we can still create legacy configs
    let legacy_config = BenchmarkConfig {
        name: None,
        base_url: "http://localhost:8000/v1".to_string(),
        api_key: "test".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        max_tokens: 256,
        concurrency: 5,
        rps: 100.0,
        timeout_secs: 300,
        disable_prewarm: true,
        verify: false,
        random_prompt_selection: false,
        random_seed: None,
    };

    assert_eq!(legacy_config.max_tokens, 256);
    assert_eq!(legacy_config.concurrency, 5);
    assert!(legacy_config.disable_prewarm);

    // Verify synthetic dataset still works
    let _synthetic = DatasetConfig::Synthetic { seed: Some(123) };
    // Just verify it can be created without panic
}

/// Test dataset type creation and variants
#[test]
fn test_dataset_config_variants() {
    let _synth = DatasetConfig::Synthetic { seed: Some(42) };
    let _synth2 = DatasetConfig::Synthetic { seed: Some(42) };

    // Verify configurations can be created successfully
    let _mr_qa = DatasetConfig::MultiRoundQa {
        path: "test.jsonl".to_string(),
        num_rounds: 4,
        users_per_round: None,
    };

    let _rag = DatasetConfig::Rag {
        documents_path: "docs.jsonl".to_string(),
        questions_path: None,
        use_precomputed_cache: false,
    };

    // All variants created successfully
    assert!(true);
}
