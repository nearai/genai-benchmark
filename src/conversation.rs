//! Conversation state management for multi-turn benchmarks
//!
//! This module provides thread-safe conversation history tracking for multi-round
//! QA benchmarks where users maintain state across multiple turns.

use crate::Message;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

/// Represents a single conversation's state
#[derive(Debug, Clone)]
pub struct ConversationState {
    /// Unique user identifier
    pub user_id: String,
    /// All messages in this conversation (including system, user, assistant)
    pub messages: Vec<Message>,
    /// Current round number
    pub round: usize,
}

impl ConversationState {
    /// Create a new conversation for a user
    pub fn new(user_id: String) -> Self {
        ConversationState {
            user_id,
            messages: Vec::new(),
            round: 0,
        }
    }

    /// Append a message to the conversation
    pub fn append_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Get all messages in the conversation
    pub fn get_history(&self) -> Vec<Message> {
        self.messages.clone()
    }

    /// Move to the next round
    pub fn next_round(&mut self) {
        self.round += 1;
    }

    /// Get current round
    pub fn get_round(&self) -> usize {
        self.round
    }

    /// Clear messages but keep user_id and round for next iteration
    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }
}

/// Thread-safe manager for conversation state across multiple concurrent users
#[derive(Clone)]
pub struct ConversationManager {
    /// Conversation state indexed by user ID
    conversations: Arc<RwLock<HashMap<String, ConversationState>>>,
}

impl ConversationManager {
    /// Create a new conversation manager
    pub fn new() -> Self {
        ConversationManager {
            conversations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create a conversation for a user
    pub async fn get_or_create(&self, user_id: &str) -> ConversationState {
        let mut conversations = self.conversations.write().await;

        conversations
            .entry(user_id.to_string())
            .or_insert_with(|| ConversationState::new(user_id.to_string()))
            .clone()
    }

    /// Append a message to a user's conversation
    pub async fn append_message(&self, user_id: &str, message: Message) {
        let mut conversations = self.conversations.write().await;

        conversations
            .entry(user_id.to_string())
            .or_insert_with(|| ConversationState::new(user_id.to_string()))
            .append_message(message);

        debug!("Message appended to user {}", user_id);
    }

    /// Get the conversation history for a user
    pub async fn get_history(&self, user_id: &str) -> Vec<Message> {
        let conversations = self.conversations.read().await;

        conversations
            .get(user_id)
            .map(|conv| conv.get_history())
            .unwrap_or_default()
    }

    /// Get the current round for a user
    pub async fn get_round(&self, user_id: &str) -> usize {
        let conversations = self.conversations.read().await;

        conversations
            .get(user_id)
            .map(|conv| conv.get_round())
            .unwrap_or(0)
    }

    /// Advance to the next round for a user
    pub async fn next_round(&self, user_id: &str) {
        let mut conversations = self.conversations.write().await;

        let conv = conversations
            .entry(user_id.to_string())
            .or_insert_with(|| ConversationState::new(user_id.to_string()));

        conv.next_round();
        debug!("User {} advanced to round {}", user_id, conv.get_round());
    }

    /// Clear messages for a user (but keep conversation metadata)
    pub async fn clear_messages(&self, user_id: &str) {
        let mut conversations = self.conversations.write().await;

        if let Some(conv) = conversations.get_mut(user_id) {
            conv.clear_messages();
            debug!("Messages cleared for user {}", user_id);
        }
    }

    /// Get total number of active conversations
    pub async fn active_conversation_count(&self) -> usize {
        let conversations = self.conversations.read().await;
        conversations.len()
    }

    /// Get all user IDs with active conversations
    pub async fn get_user_ids(&self) -> Vec<String> {
        let conversations = self.conversations.read().await;
        conversations.keys().cloned().collect()
    }

    /// Reset all conversations (for testing or between phases)
    pub async fn reset_all(&self) {
        let mut conversations = self.conversations.write().await;
        conversations.clear();
        debug!("All conversations reset");
    }
}

impl Default for ConversationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_new_conversation() {
        let manager = ConversationManager::new();
        let conv = manager.get_or_create("user1").await;

        assert_eq!(conv.user_id, "user1");
        assert_eq!(conv.messages.len(), 0);
        assert_eq!(conv.round, 0);
    }

    #[tokio::test]
    async fn test_append_message() {
        let manager = ConversationManager::new();

        let msg = Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };

        manager.append_message("user1", msg.clone()).await;

        let history = manager.get_history("user1").await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "Hello");
    }

    #[tokio::test]
    async fn test_conversation_history() {
        let manager = ConversationManager::new();

        manager
            .append_message(
                "user1",
                Message {
                    role: "user".to_string(),
                    content: "Hi".to_string(),
                },
            )
            .await;

        manager
            .append_message(
                "user1",
                Message {
                    role: "assistant".to_string(),
                    content: "Hello!".to_string(),
                },
            )
            .await;

        let history = manager.get_history("user1").await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[1].role, "assistant");
    }

    #[tokio::test]
    async fn test_multiple_users() {
        let manager = ConversationManager::new();

        manager
            .append_message(
                "user1",
                Message {
                    role: "user".to_string(),
                    content: "User1 message".to_string(),
                },
            )
            .await;

        manager
            .append_message(
                "user2",
                Message {
                    role: "user".to_string(),
                    content: "User2 message".to_string(),
                },
            )
            .await;

        let user1_history = manager.get_history("user1").await;
        let user2_history = manager.get_history("user2").await;

        assert_eq!(user1_history.len(), 1);
        assert_eq!(user2_history.len(), 1);
        assert_eq!(user1_history[0].content, "User1 message");
        assert_eq!(user2_history[0].content, "User2 message");
    }

    #[tokio::test]
    async fn test_round_tracking() {
        let manager = ConversationManager::new();

        assert_eq!(manager.get_round("user1").await, 0);

        manager.next_round("user1").await;
        assert_eq!(manager.get_round("user1").await, 1);

        manager.next_round("user1").await;
        assert_eq!(manager.get_round("user1").await, 2);
    }

    #[tokio::test]
    async fn test_clear_messages() {
        let manager = ConversationManager::new();

        manager
            .append_message(
                "user1",
                Message {
                    role: "user".to_string(),
                    content: "Test".to_string(),
                },
            )
            .await;

        manager.next_round("user1").await;

        assert_eq!(manager.get_history("user1").await.len(), 1);
        assert_eq!(manager.get_round("user1").await, 1);

        manager.clear_messages("user1").await;

        assert_eq!(manager.get_history("user1").await.len(), 0);
        assert_eq!(manager.get_round("user1").await, 1); // Round should persist
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let manager = ConversationManager::new();
        let mut tasks = vec![];

        for user_id in 0..10 {
            let manager_clone = manager.clone();
            let task = tokio::spawn(async move {
                let user_id_str = format!("user{}", user_id);

                for round in 0..5 {
                    manager_clone
                        .append_message(
                            &user_id_str,
                            Message {
                                role: "user".to_string(),
                                content: format!("Message {}", round),
                            },
                        )
                        .await;

                    manager_clone.next_round(&user_id_str).await;
                }
            });
            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            task.await.unwrap();
        }

        // Verify state
        assert_eq!(manager.active_conversation_count().await, 10);

        for user_id in 0..10 {
            let user_id_str = format!("user{}", user_id);
            let history = manager.get_history(&user_id_str).await;
            let round = manager.get_round(&user_id_str).await;

            assert_eq!(history.len(), 5);
            assert_eq!(round, 5);
        }
    }

    #[tokio::test]
    async fn test_reset_all() {
        let manager = ConversationManager::new();

        manager
            .append_message(
                "user1",
                Message {
                    role: "user".to_string(),
                    content: "Test".to_string(),
                },
            )
            .await;

        manager
            .append_message(
                "user2",
                Message {
                    role: "user".to_string(),
                    content: "Test".to_string(),
                },
            )
            .await;

        assert_eq!(manager.active_conversation_count().await, 2);

        manager.reset_all().await;

        assert_eq!(manager.active_conversation_count().await, 0);
    }

    #[tokio::test]
    async fn test_get_user_ids() {
        let manager = ConversationManager::new();

        manager.get_or_create("user1").await;
        manager.get_or_create("user2").await;
        manager.get_or_create("user3").await;

        let user_ids = manager.get_user_ids().await;
        assert_eq!(user_ids.len(), 3);
        assert!(user_ids.contains(&"user1".to_string()));
        assert!(user_ids.contains(&"user2".to_string()));
        assert!(user_ids.contains(&"user3".to_string()));
    }
}
