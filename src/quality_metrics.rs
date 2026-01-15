//! Quality metrics for RAG and question-answering benchmarks
//!
//! This module provides functions to evaluate the quality of responses
//! against expected answers using standard NLP metrics:
//! - F1 Score: Token-based precision and recall
//! - ROUGE-L: Longest common subsequence-based evaluation

use std::collections::HashSet;

/// Tokenize text into a vector of words
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

/// Calculate F1 score between generated and reference text
///
/// Uses token-based matching:
/// - Precision = overlapping tokens / generated tokens
/// - Recall = overlapping tokens / reference tokens
/// - F1 = 2 * (precision * recall) / (precision + recall)
///
/// Returns a score from 0.0 to 1.0
pub fn calculate_f1_score(generated: &str, reference: &str) -> f64 {
    let generated_tokens = tokenize(generated);
    let reference_tokens = tokenize(reference);

    if generated_tokens.is_empty() || reference_tokens.is_empty() {
        if generated_tokens.is_empty() && reference_tokens.is_empty() {
            return 1.0; // Both empty
        }
        return 0.0; // One is empty
    }

    // Count overlapping tokens
    let generated_set: HashSet<_> = generated_tokens.iter().collect();
    let reference_set: HashSet<_> = reference_tokens.iter().collect();

    let overlap = generated_set
        .intersection(&reference_set)
        .count();

    let precision = overlap as f64 / generated_tokens.len() as f64;
    let recall = overlap as f64 / reference_tokens.len() as f64;

    if precision + recall == 0.0 {
        return 0.0;
    }

    2.0 * (precision * recall) / (precision + recall)
}

/// Calculate Longest Common Subsequence length
fn lcs_length(a: &[String], b: &[String]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 || n == 0 {
        return 0;
    }

    // Create DP table
    let mut dp = vec![vec![0; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    dp[m][n]
}

/// Calculate ROUGE-L score (Longest Common Subsequence F-score)
///
/// ROUGE-L measures the longest matching sequence of words between texts.
/// Precision = LCS / length(generated)
/// Recall = LCS / length(reference)
/// F-score = 2 * (precision * recall) / (precision + recall)
///
/// Returns a score from 0.0 to 1.0
pub fn calculate_rouge_l_score(generated: &str, reference: &str) -> f64 {
    let generated_tokens = tokenize(generated);
    let reference_tokens = tokenize(reference);

    if generated_tokens.is_empty() || reference_tokens.is_empty() {
        if generated_tokens.is_empty() && reference_tokens.is_empty() {
            return 1.0; // Both empty
        }
        return 0.0; // One is empty
    }

    let lcs = lcs_length(&generated_tokens, &reference_tokens) as f64;
    let gen_len = generated_tokens.len() as f64;
    let ref_len = reference_tokens.len() as f64;

    let precision = lcs / gen_len;
    let recall = lcs / ref_len;

    if precision + recall == 0.0 {
        return 0.0;
    }

    2.0 * (precision * recall) / (precision + recall)
}

/// Calculate both F1 and ROUGE-L scores
pub fn calculate_quality_scores(generated: &str, reference: &str) -> (f64, f64) {
    let f1 = calculate_f1_score(generated, reference);
    let rouge_l = calculate_rouge_l_score(generated, reference);
    (f1, rouge_l)
}

/// Average multiple F1 scores
pub fn average_f1_scores(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().sum::<f64>() / scores.len() as f64
}

/// Average multiple ROUGE-L scores
pub fn average_rouge_l_scores(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().sum::<f64>() / scores.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_simple() {
        let tokens = tokenize("Hello World");
        assert_eq!(tokens, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn test_tokenize_with_punctuation() {
        let tokens = tokenize("Hello, World!");
        assert_eq!(tokens, vec!["hello,".to_string(), "world!".to_string()]);
    }

    #[test]
    fn test_f1_perfect_match() {
        let score = calculate_f1_score("machine learning", "machine learning");
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f1_no_match() {
        let score = calculate_f1_score("apple orange", "dog cat");
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_f1_partial_match() {
        let score = calculate_f1_score("machine learning algorithms", "machine learning");
        // overlap: 2 tokens, generated: 3 tokens, reference: 2 tokens
        // precision = 2/3 = 0.667, recall = 2/2 = 1.0
        // f1 = 2 * (0.667 * 1.0) / (0.667 + 1.0) = 0.8
        assert!((score - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_f1_empty_generated() {
        let score = calculate_f1_score("", "machine learning");
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_f1_empty_reference() {
        let score = calculate_f1_score("machine learning", "");
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_f1_both_empty() {
        let score = calculate_f1_score("", "");
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f1_case_insensitive() {
        let score1 = calculate_f1_score("MACHINE LEARNING", "machine learning");
        let score2 = calculate_f1_score("machine learning", "MACHINE LEARNING");
        assert!((score1 - 1.0).abs() < 0.001);
        assert!((score2 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_lcs_simple() {
        let a = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let b = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let lcs = lcs_length(&a, &b);
        assert_eq!(lcs, 3);
    }

    #[test]
    fn test_lcs_partial() {
        let a = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let b = vec!["a".to_string(), "c".to_string()];
        let lcs = lcs_length(&a, &b);
        assert_eq!(lcs, 2); // "a" and "c"
    }

    #[test]
    fn test_lcs_no_match() {
        let a = vec!["a".to_string(), "b".to_string()];
        let b = vec!["c".to_string(), "d".to_string()];
        let lcs = lcs_length(&a, &b);
        assert_eq!(lcs, 0);
    }

    #[test]
    fn test_lcs_empty() {
        let a: Vec<String> = vec![];
        let b = vec!["a".to_string()];
        let lcs = lcs_length(&a, &b);
        assert_eq!(lcs, 0);
    }

    #[test]
    fn test_rouge_l_perfect_match() {
        let score = calculate_rouge_l_score("machine learning", "machine learning");
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rouge_l_no_match() {
        let score = calculate_rouge_l_score("apple orange", "dog cat");
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_rouge_l_partial_match_ordered() {
        // "the cat sat" vs "the dog sat"
        // LCS = 2 ("the" and "sat")
        let score = calculate_rouge_l_score("the cat sat", "the dog sat");
        let gen_len = 3.0;
        let ref_len = 3.0;
        let lcs = 2.0;
        let precision = lcs / gen_len; // 2/3
        let recall = lcs / ref_len;    // 2/3
        let expected_f1 = 2.0 * (precision * recall) / (precision + recall); // 2/3
        assert!((score - expected_f1).abs() < 0.01);
    }

    #[test]
    fn test_rouge_l_empty_generated() {
        let score = calculate_rouge_l_score("", "machine learning");
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_rouge_l_empty_reference() {
        let score = calculate_rouge_l_score("machine learning", "");
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_rouge_l_both_empty() {
        let score = calculate_rouge_l_score("", "");
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_quality_scores() {
        let (f1, rouge_l) = calculate_quality_scores("machine learning", "machine learning");
        assert!((f1 - 1.0).abs() < 0.001);
        assert!((rouge_l - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_average_f1_scores() {
        let scores = vec![1.0, 0.8, 0.6];
        let avg = average_f1_scores(&scores);
        assert!((avg - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_average_f1_scores_empty() {
        let scores: Vec<f64> = vec![];
        let avg = average_f1_scores(&scores);
        assert!((avg - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_average_rouge_l_scores() {
        let scores = vec![1.0, 0.75, 0.5];
        let avg = average_rouge_l_scores(&scores);
        assert!((avg - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_f1_common_example() {
        // "the quick brown fox" vs "the brown fox"
        // tokens match: "the", "brown", "fox" (3 overlapping)
        // generated: 4 tokens, reference: 3 tokens
        // precision = 3/4 = 0.75, recall = 3/3 = 1.0
        // f1 = 2 * (0.75 * 1.0) / (0.75 + 1.0) = 0.857...
        let score = calculate_f1_score("the quick brown fox", "the brown fox");
        let expected = 2.0 * (0.75 * 1.0) / (0.75 + 1.0);
        assert!((score - expected).abs() < 0.001);
    }

    #[test]
    fn test_rouge_l_subsequence_not_substring() {
        // ROUGE-L should handle non-contiguous sequences
        // "a b c d" vs "a x c y" - LCS = 2 ("a", "c")
        let score = calculate_rouge_l_score("a b c d", "a x c y");
        let lcs = 2.0;
        let gen_len = 4.0;
        let ref_len = 4.0;
        let precision = lcs / gen_len;
        let recall = lcs / ref_len;
        let expected_f1 = 2.0 * (precision * recall) / (precision + recall);
        assert!((score - expected_f1).abs() < 0.001);
    }
}
