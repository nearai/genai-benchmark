//! Cache effectiveness metrics for LMCache benchmarks
//!
//! This module provides functions to extract and track cache hit/miss information
//! from API responses, calculate cache token savings, and aggregate cache statistics.

use crate::CacheMetrics;
use reqwest::header::HeaderMap;
use tracing::debug;

/// Extract cache metrics from response headers
///
/// Looks for standard LMCache headers:
/// - x-cache-hit: boolean indicating if this was a cache hit
/// - x-cached-tokens: number of tokens served from cache
/// - x-computed-tokens: number of tokens computed (not cached)
/// - x-cache-hit-rate: overall cache hit rate (0.0-1.0)
pub fn extract_cache_metrics_from_headers(
    headers: &HeaderMap,
    input_tokens: u32,
    output_tokens: u32,
) -> Option<CacheMetrics> {
    // Check if x-cache-hit header exists
    let cache_hit = headers
        .get("x-cache-hit")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<bool>().ok())
        .unwrap_or(false);

    // Extract cached and computed tokens from headers
    let cached_tokens = headers
        .get("x-cached-tokens")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    let computed_tokens = headers
        .get("x-computed-tokens")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(output_tokens);

    // Extract or calculate cache hit rate
    let cache_hit_rate = headers
        .get("x-cache-hit-rate")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or_else(|| {
            // Calculate hit rate from cached vs total tokens
            if input_tokens + output_tokens > 0 {
                cached_tokens as f64 / (input_tokens + output_tokens) as f64
            } else {
                0.0
            }
        });

    debug!(
        "Cache metrics - hit: {}, cached_tokens: {}, computed_tokens: {}, hit_rate: {:.2}",
        cache_hit, cached_tokens, computed_tokens, cache_hit_rate
    );

    Some(CacheMetrics {
        cache_hit,
        cached_tokens,
        computed_tokens,
        cache_hit_rate,
    })
}

/// Determine if a request was a cache hit based on metrics
pub fn is_cache_hit(metrics: &CacheMetrics) -> bool {
    metrics.cache_hit || metrics.cache_hit_rate > 0.5
}

/// Calculate token savings from cache metrics
/// Returns the number of tokens that didn't need to be computed due to caching
pub fn calculate_cache_token_savings(metrics: &CacheMetrics) -> u64 {
    metrics.cached_tokens as u64
}

/// Aggregate cache metrics from multiple requests
pub struct CacheMetricsAggregator {
    pub total_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub total_cached_tokens: u64,
    pub total_computed_tokens: u64,
    pub hit_rates: Vec<f64>,
}

impl CacheMetricsAggregator {
    pub fn new() -> Self {
        CacheMetricsAggregator {
            total_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            total_cached_tokens: 0,
            total_computed_tokens: 0,
            hit_rates: Vec::new(),
        }
    }

    pub fn add_metrics(&mut self, metrics: &CacheMetrics) {
        self.total_requests += 1;

        if is_cache_hit(metrics) {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }

        self.total_cached_tokens += metrics.cached_tokens as u64;
        self.total_computed_tokens += metrics.computed_tokens as u64;
        self.hit_rates.push(metrics.cache_hit_rate);
    }

    pub fn average_hit_rate(&self) -> f64 {
        if self.hit_rates.is_empty() {
            0.0
        } else {
            self.hit_rates.iter().sum::<f64>() / self.hit_rates.len() as f64
        }
    }

    pub fn total_token_savings(&self) -> u64 {
        self.total_cached_tokens
    }
}

impl Default for CacheMetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderMap;

    #[test]
    fn test_extract_cache_hit_from_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-cache-hit", "true".parse().unwrap());
        headers.insert("x-cached-tokens", "100".parse().unwrap());
        headers.insert("x-computed-tokens", "10".parse().unwrap());

        let metrics = extract_cache_metrics_from_headers(&headers, 50, 100).unwrap();

        assert!(metrics.cache_hit);
        assert_eq!(metrics.cached_tokens, 100);
        assert_eq!(metrics.computed_tokens, 10);
    }

    #[test]
    fn test_extract_cache_miss_from_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-cache-hit", "false".parse().unwrap());
        headers.insert("x-cached-tokens", "0".parse().unwrap());
        headers.insert("x-computed-tokens", "100".parse().unwrap());

        let metrics = extract_cache_metrics_from_headers(&headers, 50, 100).unwrap();

        assert!(!metrics.cache_hit);
        assert_eq!(metrics.cached_tokens, 0);
        assert_eq!(metrics.computed_tokens, 100);
    }

    #[test]
    fn test_extract_cache_hit_rate_from_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-cache-hit-rate", "0.8".parse().unwrap());

        let metrics = extract_cache_metrics_from_headers(&headers, 50, 100).unwrap();

        assert!((metrics.cache_hit_rate - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_calculate_cache_hit_rate_from_tokens() {
        let mut headers = HeaderMap::new();
        headers.insert("x-cached-tokens", "80".parse().unwrap());

        let metrics = extract_cache_metrics_from_headers(&headers, 50, 100).unwrap();

        // 80 / (50 + 100) = 0.533...
        assert!((metrics.cache_hit_rate - 0.533).abs() < 0.01);
    }

    #[test]
    fn test_is_cache_hit_with_flag() {
        let metrics = CacheMetrics {
            cache_hit: true,
            cached_tokens: 100,
            computed_tokens: 10,
            cache_hit_rate: 0.9,
        };

        assert!(is_cache_hit(&metrics));
    }

    #[test]
    fn test_is_cache_hit_with_high_rate() {
        let metrics = CacheMetrics {
            cache_hit: false,
            cached_tokens: 80,
            computed_tokens: 20,
            cache_hit_rate: 0.8,
        };

        assert!(is_cache_hit(&metrics));
    }

    #[test]
    fn test_is_cache_miss_with_low_rate() {
        let metrics = CacheMetrics {
            cache_hit: false,
            cached_tokens: 10,
            computed_tokens: 90,
            cache_hit_rate: 0.1,
        };

        assert!(!is_cache_hit(&metrics));
    }

    #[test]
    fn test_calculate_token_savings() {
        let metrics = CacheMetrics {
            cache_hit: true,
            cached_tokens: 500,
            computed_tokens: 50,
            cache_hit_rate: 0.9,
        };

        let savings = calculate_cache_token_savings(&metrics);
        assert_eq!(savings, 500);
    }

    #[test]
    fn test_cache_metrics_aggregator_single_hit() {
        let mut agg = CacheMetricsAggregator::new();
        let metrics = CacheMetrics {
            cache_hit: true,
            cached_tokens: 100,
            computed_tokens: 10,
            cache_hit_rate: 0.91,
        };

        agg.add_metrics(&metrics);

        assert_eq!(agg.total_requests, 1);
        assert_eq!(agg.cache_hits, 1);
        assert_eq!(agg.cache_misses, 0);
        assert_eq!(agg.total_cached_tokens, 100);
        assert_eq!(agg.total_computed_tokens, 10);
        assert!((agg.average_hit_rate() - 0.91).abs() < 0.01);
    }

    #[test]
    fn test_cache_metrics_aggregator_mixed() {
        let mut agg = CacheMetricsAggregator::new();

        let hit_metrics = CacheMetrics {
            cache_hit: true,
            cached_tokens: 100,
            computed_tokens: 10,
            cache_hit_rate: 0.91,
        };

        let miss_metrics = CacheMetrics {
            cache_hit: false,
            cached_tokens: 0,
            computed_tokens: 120,
            cache_hit_rate: 0.0,
        };

        agg.add_metrics(&hit_metrics);
        agg.add_metrics(&miss_metrics);

        assert_eq!(agg.total_requests, 2);
        assert_eq!(agg.cache_hits, 1);
        assert_eq!(agg.cache_misses, 1);
        assert_eq!(agg.total_cached_tokens, 100);
        assert_eq!(agg.total_computed_tokens, 130);
        assert!((agg.average_hit_rate() - 0.455).abs() < 0.01);
        assert_eq!(agg.total_token_savings(), 100);
    }

    #[test]
    fn test_cache_metrics_aggregator_multiple_hits() {
        let mut agg = CacheMetricsAggregator::new();

        for i in 0..5 {
            let metrics = CacheMetrics {
                cache_hit: true,
                cached_tokens: 80 + i as u32,
                computed_tokens: 20,
                cache_hit_rate: 0.8,
            };
            agg.add_metrics(&metrics);
        }

        assert_eq!(agg.total_requests, 5);
        assert_eq!(agg.cache_hits, 5);
        assert_eq!(agg.cache_misses, 0);
        assert_eq!(agg.total_cached_tokens, 410); // 80+81+82+83+84
        assert_eq!(agg.total_computed_tokens, 100); // 20*5
        assert!((agg.average_hit_rate() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_empty_aggregator() {
        let agg = CacheMetricsAggregator::new();

        assert_eq!(agg.total_requests, 0);
        assert_eq!(agg.cache_hits, 0);
        assert_eq!(agg.cache_misses, 0);
        assert_eq!(agg.total_cached_tokens, 0);
        assert_eq!(agg.total_computed_tokens, 0);
        assert_eq!(agg.average_hit_rate(), 0.0);
    }
}
