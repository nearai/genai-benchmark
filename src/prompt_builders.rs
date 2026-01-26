//! Specialized prompt builders for different benchmark types
//!
//! This module provides trait-based prompt construction for various LMCache benchmarks,
//! allowing customization of how prompts are formatted for different use cases.

use crate::{ConversationManager, Message, MessageContent};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;

/// Context for building a prompt
#[derive(Debug, Clone)]
pub struct PromptContext {
    /// User identifier (for multi-round scenarios)
    pub user_id: Option<String>,
    /// Current round number
    pub round: usize,
    /// Document(s) for RAG/long-doc scenarios
    pub documents: Vec<String>,
    /// Question or query
    pub query: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl PromptContext {
    /// Create a new prompt context with just a query
    pub fn new(query: String) -> Self {
        PromptContext {
            user_id: None,
            round: 0,
            documents: Vec::new(),
            query,
            metadata: HashMap::new(),
        }
    }

    /// Create context for a user
    pub fn with_user(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Add a document
    pub fn with_document(mut self, doc: String) -> Self {
        self.documents.push(doc);
        self
    }

    /// Add multiple documents
    pub fn with_documents(mut self, docs: Vec<String>) -> Self {
        self.documents.extend(docs);
        self
    }

    /// Set round number
    pub fn with_round(mut self, round: usize) -> Self {
        self.round = round;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Trait for building prompts in different formats
#[async_trait::async_trait]
pub trait PromptBuilder: Send + Sync {
    /// Build a prompt from the given context
    async fn build_prompt(&self, context: &PromptContext) -> crate::Result<Vec<Message>>;

    /// Get a descriptive name for this builder
    fn name(&self) -> &str;
}

/// RAG prompt builder - formats prompts as system + document + question
pub struct RagPromptBuilder {
    /// System prompt to use
    pub system_prompt: String,
    /// Separator between document and question
    pub separator: String,
}

impl RagPromptBuilder {
    /// Create a new RAG prompt builder with defaults
    pub fn new() -> Self {
        RagPromptBuilder {
            system_prompt: "Answer the following question based only on the provided document. Be concise and accurate.".to_string(),
            separator: "\n\nQuestion: ".to_string(),
        }
    }

    /// Set custom system prompt
    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = prompt;
        self
    }

    /// Set custom separator
    pub fn with_separator(mut self, sep: String) -> Self {
        self.separator = sep;
        self
    }
}

impl Default for RagPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl PromptBuilder for RagPromptBuilder {
    async fn build_prompt(&self, context: &PromptContext) -> crate::Result<Vec<Message>> {
        let mut messages = vec![Message {
            role: "system".to_string(),
            content: MessageContent::Text(self.system_prompt.clone()),
        }];

        // Combine all documents
        let doc_text = if context.documents.is_empty() {
            "[No document provided]".to_string()
        } else {
            context
                .documents
                .iter()
                .enumerate()
                .map(|(i, doc)| {
                    if context.documents.len() > 1 {
                        format!("Document {}:\n{}", i + 1, doc)
                    } else {
                        format!("Document:\n{}", doc)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n\n---\n\n")
        };

        let user_content = format!("{}{}{}", doc_text, self.separator, context.query);

        messages.push(Message {
            role: "user".to_string(),
            content: MessageContent::Text(user_content),
        });

        Ok(messages)
    }

    fn name(&self) -> &str {
        "RAG"
    }
}

/// Long document QA prompt builder - similar to RAG but optimized for very long documents
pub struct LongDocPromptBuilder {
    /// System prompt to use
    pub system_prompt: String,
    /// Separator between document and question
    pub separator: String,
}

impl LongDocPromptBuilder {
    /// Create a new long document QA builder
    pub fn new() -> Self {
        LongDocPromptBuilder {
            system_prompt: "You are an expert at reading and comprehending long documents. Answer the following question based on the provided document.".to_string(),
            separator: "\n\nQuestion: ".to_string(),
        }
    }

    /// Set custom system prompt
    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = prompt;
        self
    }

    /// Set custom separator
    pub fn with_separator(mut self, sep: String) -> Self {
        self.separator = sep;
        self
    }
}

impl Default for LongDocPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl PromptBuilder for LongDocPromptBuilder {
    async fn build_prompt(&self, context: &PromptContext) -> crate::Result<Vec<Message>> {
        let mut messages = vec![Message {
            role: "system".to_string(),
            content: MessageContent::Text(self.system_prompt.clone()),
        }];

        let doc_text = if context.documents.is_empty() {
            "[No document provided]".to_string()
        } else {
            // For very long docs, we might want to add length info
            let combined = context.documents.join("\n\n");
            let word_count = combined.split_whitespace().count();
            if word_count > 5000 {
                format!(
                    "[Document: ~{} words]\n\n{}",
                    word_count, combined
                )
            } else {
                combined
            }
        };

        let user_content = format!("{}{}{}", doc_text, self.separator, context.query);

        messages.push(Message {
            role: "user".to_string(),
            content: MessageContent::Text(user_content),
        });

        Ok(messages)
    }

    fn name(&self) -> &str {
        "LongDocQA"
    }
}

/// Multi-document QA prompt builder - concatenates documents with context
pub struct MultiDocPromptBuilder {
    /// System prompt
    pub system_prompt: String,
    /// Document separator
    pub doc_separator: String,
    /// Question separator
    pub query_separator: String,
    /// Maximum documents to use (if set, samples from larger set)
    pub max_docs: Option<usize>,
}

impl MultiDocPromptBuilder {
    /// Create a new multi-document QA builder
    pub fn new() -> Self {
        MultiDocPromptBuilder {
            system_prompt: "You are an expert at synthesizing information from multiple documents. Answer the following question based on the provided documents.".to_string(),
            doc_separator: "\n\n---\n\n".to_string(),
            query_separator: "\n\nQuestion: ".to_string(),
            max_docs: None,
        }
    }

    /// Set custom system prompt
    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = prompt;
        self
    }

    /// Set maximum number of documents to use
    pub fn with_max_docs(mut self, max: usize) -> Self {
        self.max_docs = Some(max);
        self
    }

    /// Set document separator
    pub fn with_doc_separator(mut self, sep: String) -> Self {
        self.doc_separator = sep;
        self
    }

    /// Set query separator
    pub fn with_query_separator(mut self, sep: String) -> Self {
        self.query_separator = sep;
        self
    }
}

impl Default for MultiDocPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl PromptBuilder for MultiDocPromptBuilder {
    async fn build_prompt(&self, context: &PromptContext) -> crate::Result<Vec<Message>> {
        let mut messages = vec![Message {
            role: "system".to_string(),
            content: MessageContent::Text(self.system_prompt.clone()),
        }];

        // Limit documents if max_docs is set
        let docs_to_use = if let Some(max) = self.max_docs {
            if context.documents.len() > max {
                context.documents[..max].to_vec()
            } else {
                context.documents.clone()
            }
        } else {
            context.documents.clone()
        };

        // Format documents with headers
        let formatted_docs: Vec<String> = docs_to_use
            .iter()
            .enumerate()
            .map(|(i, doc)| format!("Document {}:\n{}", i + 1, doc))
            .collect();

        let doc_text = if formatted_docs.is_empty() {
            "[No documents provided]".to_string()
        } else {
            formatted_docs.join(&self.doc_separator)
        };

        let user_content = format!("{}{}{}", doc_text, self.query_separator, context.query);

        messages.push(Message {
            role: "user".to_string(),
            content: MessageContent::Text(user_content),
        });

        Ok(messages)
    }

    fn name(&self) -> &str {
        "MultiDocQA"
    }
}

/// Multi-round QA prompt builder - integrates with conversation history
pub struct MultiRoundPromptBuilder {
    /// System prompt
    pub system_prompt: String,
    /// Conversation manager (optional, for accessing full history)
    pub conversation_manager: Option<crate::ConversationManager>,
    /// Include conversation history in prompt
    pub include_history: bool,
}

impl MultiRoundPromptBuilder {
    /// Create a new multi-round QA builder
    pub fn new() -> Self {
        MultiRoundPromptBuilder {
            system_prompt: "You are a helpful assistant. Answer questions accurately and concisely.".to_string(),
            conversation_manager: None,
            include_history: true,
        }
    }

    /// Set custom system prompt
    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = prompt;
        self
    }

    /// Set conversation manager for history access
    pub fn with_conversation_manager(mut self, manager: ConversationManager) -> Self {
        self.conversation_manager = Some(manager);
        self
    }

    /// Whether to include conversation history
    pub fn with_include_history(mut self, include: bool) -> Self {
        self.include_history = include;
        self
    }
}

impl Default for MultiRoundPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl PromptBuilder for MultiRoundPromptBuilder {
    async fn build_prompt(&self, context: &PromptContext) -> crate::Result<Vec<Message>> {
        let mut messages = vec![Message {
            role: "system".to_string(),
            content: MessageContent::Text(self.system_prompt.clone()),
        }];

        // If we have a conversation manager and user_id, get history
        if self.include_history {
            if let Some(user_id) = &context.user_id {
                if let Some(manager) = &self.conversation_manager {
                    let history = manager.get_history(user_id).await;
                    // Add historical messages (excluding the new query)
                    messages.extend(history);
                }
            }
        }

        // Add current question
        messages.push(Message {
            role: "user".to_string(),
            content: MessageContent::Text(context.query.clone()),
        });

        Ok(messages)
    }

    fn name(&self) -> &str {
        "MultiRoundQA"
    }
}

/// Document store for multi-document scenarios
pub struct DocumentStore {
    /// Documents indexed by ID
    documents: HashMap<String, String>,
}

impl DocumentStore {
    /// Create a new document store
    pub fn new() -> Self {
        DocumentStore {
            documents: HashMap::new(),
        }
    }

    /// Add a document
    pub fn add_document(&mut self, id: String, content: String) {
        self.documents.insert(id, content);
    }

    /// Add multiple documents
    pub fn add_documents(&mut self, docs: Vec<(String, String)>) {
        for (id, content) in docs {
            self.documents.insert(id, content);
        }
    }

    /// Get a document by ID
    pub fn get_document(&self, id: &str) -> Option<String> {
        self.documents.get(id).cloned()
    }

    /// Get multiple documents by IDs
    pub fn get_documents(&self, ids: &[&str]) -> Vec<String> {
        ids.iter()
            .filter_map(|id| self.documents.get(*id).cloned())
            .collect()
    }

    /// Get random document IDs
    pub fn get_random_documents(
        &self,
        count: usize,
        seed: Option<u64>,
    ) -> Vec<String> {
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let mut ids: Vec<String> = self.documents.keys().cloned().collect();
        ids.shuffle(&mut rng);

        ids.into_iter().take(count).collect()
    }

    /// Get total number of documents
    pub fn count(&self) -> usize {
        self.documents.len()
    }

    /// Get all document IDs
    pub fn get_all_ids(&self) -> Vec<String> {
        self.documents.keys().cloned().collect()
    }
}

impl Default for DocumentStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rag_prompt_builder() {
        let builder = RagPromptBuilder::new();
        let context = PromptContext::new("What is ML?".to_string())
            .with_document("Machine learning is...".to_string());

        let prompt = builder.build_prompt(&context).await.unwrap();

        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, "system");
        assert_eq!(prompt[1].role, "user");
        assert!(prompt[1].content.contains("What is ML?"));
        assert!(prompt[1].content.contains("Machine learning is..."));
    }

    #[tokio::test]
    async fn test_rag_prompt_multiple_documents() {
        let builder = RagPromptBuilder::new();
        let context = PromptContext::new("Compare these".to_string())
            .with_document("Doc 1".to_string())
            .with_document("Doc 2".to_string());

        let prompt = builder.build_prompt(&context).await.unwrap();

        assert_eq!(prompt.len(), 2);
        assert!(prompt[1].content.contains("Document 1:"));
        assert!(prompt[1].content.contains("Document 2:"));
    }

    #[tokio::test]
    async fn test_long_doc_prompt_builder() {
        let builder = LongDocPromptBuilder::new();
        let long_doc = "word ".repeat(2000); // ~10K words
        let context = PromptContext::new("What's the main point?".to_string())
            .with_document(long_doc);

        let prompt = builder.build_prompt(&context).await.unwrap();

        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, "system");
        assert_eq!(prompt[1].role, "user");
        assert!(prompt[1].content.contains("What's the main point?"));
        assert!(prompt[1].content.contains("word"));
    }

    #[tokio::test]
    async fn test_multi_doc_prompt_builder() {
        let builder = MultiDocPromptBuilder::new().with_max_docs(2);
        let context = PromptContext::new("Answer this".to_string())
            .with_documents(vec!["Doc A".to_string(), "Doc B".to_string(), "Doc C".to_string()]);

        let prompt = builder.build_prompt(&context).await.unwrap();

        assert_eq!(prompt.len(), 2);
        // Only 2 docs should be included due to max_docs
        assert!(prompt[1].content.contains("Document 1:"));
        assert!(prompt[1].content.contains("Document 2:"));
        assert!(!prompt[1].content.contains("Document 3:"));
    }

    #[tokio::test]
    async fn test_multi_round_prompt_builder() {
        let builder = MultiRoundPromptBuilder::new();
        let context = PromptContext::new("What's your name?".to_string())
            .with_user("user1".to_string());

        let prompt = builder.build_prompt(&context).await.unwrap();

        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, "system");
        assert_eq!(prompt[1].role, "user");
        assert_eq!(prompt[1].content, "What's your name?");
    }

    #[tokio::test]
    async fn test_document_store() {
        let mut store = DocumentStore::new();

        store.add_document("doc1".to_string(), "Content 1".to_string());
        store.add_document("doc2".to_string(), "Content 2".to_string());

        assert_eq!(store.count(), 2);
        assert_eq!(store.get_document("doc1").unwrap(), "Content 1");
        assert_eq!(store.get_document("doc2").unwrap(), "Content 2");
        assert!(store.get_document("doc3").is_none());
    }

    #[tokio::test]
    async fn test_document_store_get_multiple() {
        let mut store = DocumentStore::new();
        store.add_documents(vec![
            ("doc1".to_string(), "Content 1".to_string()),
            ("doc2".to_string(), "Content 2".to_string()),
            ("doc3".to_string(), "Content 3".to_string()),
        ]);

        let docs = store.get_documents(&["doc1", "doc3"]);
        assert_eq!(docs.len(), 2);
        assert!(docs.contains(&"Content 1".to_string()));
        assert!(docs.contains(&"Content 3".to_string()));
    }

    #[tokio::test]
    async fn test_document_store_random() {
        let mut store = DocumentStore::new();
        for i in 1..=10 {
            store.add_document(format!("doc{}", i), format!("Content {}", i));
        }

        let random_ids = store.get_random_documents(5, Some(42));
        assert_eq!(random_ids.len(), 5);

        // Same seed should produce same order
        let random_ids_2 = store.get_random_documents(5, Some(42));
        assert_eq!(random_ids, random_ids_2);
    }
}
