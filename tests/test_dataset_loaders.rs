use genai_benchmark::{load_dataset, DatasetConfig};

#[tokio::test]
async fn test_multi_round_qa_loader() {
    let config = DatasetConfig::MultiRoundQa {
        path: "datasets/multi_round_qa.jsonl".to_string(),
        num_rounds: 3,
        users_per_round: None,
    };

    let result = load_dataset(&config, 10).await;
    assert!(result.is_ok(), "Multi-round QA loader failed");

    let prompts = result.unwrap();
    assert!(!prompts.is_empty(), "No prompts loaded");
    assert!(
        prompts[0].iter().any(|m| m.role == "user"),
        "No user message found"
    );
}

#[tokio::test]
async fn test_rag_loader() {
    let config = DatasetConfig::Rag {
        documents_path: "datasets/rag.jsonl".to_string(),
        questions_path: None,
        use_precomputed_cache: false,
    };

    let result = load_dataset(&config, 10).await;
    assert!(result.is_ok(), "RAG loader failed");

    let prompts = result.unwrap();
    assert!(!prompts.is_empty(), "No prompts loaded");
    assert!(prompts[0].len() >= 2, "RAG should have multiple messages");
}

#[tokio::test]
async fn test_long_doc_qa_loader() {
    let config = DatasetConfig::LongDocQa {
        documents_path: "datasets/long_doc_qa.jsonl".to_string(),
        questions_path: None,
        doc_token_length: 20000,
        warmup_rounds: 1,
    };

    let result = load_dataset(&config, 10).await;
    assert!(result.is_ok(), "Long document QA loader failed");

    let prompts = result.unwrap();
    assert!(!prompts.is_empty(), "No prompts loaded");
}

#[tokio::test]
async fn test_multi_doc_qa_loader() {
    let config = DatasetConfig::MultiDocQa {
        documents_path: "datasets/multi_doc_qa.jsonl".to_string(),
        questions_path: None,
        docs_per_request: 3,
        warmup_rounds: 1,
        sample_strategy: genai_benchmark::SampleStrategy::Random,
    };

    let result = load_dataset(&config, 10).await;
    assert!(result.is_ok(), "Multi-document QA loader failed");

    let prompts = result.unwrap();
    assert!(!prompts.is_empty(), "No prompts loaded");
}

#[tokio::test]
async fn test_random_tokens_loader() {
    let config = DatasetConfig::RandomTokens {
        token_count: 100,
        seed: Some(42),
    };

    let result = load_dataset(&config, 5).await;
    assert!(
        result.is_ok(),
        "RandomTokens loader failed: {:?}",
        result.err()
    );

    let prompts = result.unwrap();
    assert_eq!(prompts.len(), 5, "Should load exactly 5 prompts");

    for prompt in &prompts {
        assert_eq!(
            prompt.len(),
            2,
            "Each prompt should have system and user messages"
        );
        assert_eq!(prompt[0].role, "system", "First message should be system");
        assert_eq!(prompt[1].role, "user", "Second message should be user");
        assert!(
            prompt[1]
                .content
                .as_text()
                .map(|t| !t.is_empty())
                .unwrap_or(false),
            "User message should have non-empty content"
        );
    }
}

#[tokio::test]
async fn test_random_tokens_reproducibility() {
    let config1 = DatasetConfig::RandomTokens {
        token_count: 50,
        seed: Some(12345),
    };
    let config2 = DatasetConfig::RandomTokens {
        token_count: 50,
        seed: Some(12345),
    };

    let prompts1 = load_dataset(&config1, 3).await.expect("First load failed");
    let prompts2 = load_dataset(&config2, 3).await.expect("Second load failed");

    assert_eq!(
        prompts1.len(),
        prompts2.len(),
        "Should have same number of prompts"
    );

    for (p1, p2) in prompts1.iter().zip(prompts2.iter()) {
        let content1 = p1[1].content.as_text().expect("Should be text");
        let content2 = p2[1].content.as_text().expect("Should be text");
        assert_eq!(content1, content2, "Same seed should produce same content");
    }
}

#[tokio::test]
async fn test_random_tokens_different_seeds() {
    let config1 = DatasetConfig::RandomTokens {
        token_count: 50,
        seed: Some(111),
    };
    let config2 = DatasetConfig::RandomTokens {
        token_count: 50,
        seed: Some(222),
    };

    let prompts1 = load_dataset(&config1, 3).await.expect("First load failed");
    let prompts2 = load_dataset(&config2, 3).await.expect("Second load failed");

    let mut all_same = true;
    for (p1, p2) in prompts1.iter().zip(prompts2.iter()) {
        let content1 = p1[1].content.as_text().expect("Should be text");
        let content2 = p2[1].content.as_text().expect("Should be text");
        if content1 != content2 {
            all_same = false;
            break;
        }
    }
    assert!(
        !all_same,
        "Different seeds should produce different content"
    );
}
