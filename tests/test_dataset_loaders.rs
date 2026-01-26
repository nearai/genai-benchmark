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
    assert!(prompts[0].iter().any(|m| m.role == "user"), "No user message found");
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
