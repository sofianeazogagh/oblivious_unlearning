use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::NUM_THREADS;

/// Global timing collector that can be shared across threads
pub struct TimingCollector {
    timings: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
}

impl TimingCollector {
    /// Create a new timing collector
    pub fn new() -> Self {
        Self {
            timings: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Record a duration for a given operation name
    pub fn record(&self, operation: &str, duration: Duration) {
        let mut timings = self.timings.lock().unwrap();
        timings
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }

    /// Get all recorded timings
    pub fn get_timings(&self) -> HashMap<String, Vec<Duration>> {
        self.timings.lock().unwrap().clone()
    }

    /// Calculate statistics for all operations
    pub fn get_statistics(&self) -> HashMap<String, TimingStats> {
        let timings = self.timings.lock().unwrap();
        timings
            .iter()
            .map(|(name, durations)| {
                (name.clone(), TimingStats::from_durations(durations))
            })
            .collect()
    }

    /// Print statistics summary
    pub fn print_summary(&self) {
        let stats = self.get_statistics();
        if stats.is_empty() {
            return;
        }

        println!("\n========== Timing Statistics ({} threads) ==========", NUM_THREADS);
        println!("{:<40} {:>12} {:>12} {:>12} {:>12}", 
                 "Operation", "Count", "Total", "Mean", "Min");
        println!("{}", "-".repeat(92));

        let mut sorted_stats: Vec<_> = stats.iter().collect();
        sorted_stats.sort_by_key(|(name, _)| name.clone());

        for (name, stat) in sorted_stats {
            println!(
                "{:<40} {:>12} {:>12.2?} {:>12.2?} {:>12.2?}",
                name,
                stat.count,
                stat.total,
                stat.mean,
                stat.min
            );
        }
        println!("{}\n", "=".repeat(92));
    }

    /// Clear all recorded timings
    pub fn clear(&self) {
        let mut timings = self.timings.lock().unwrap();
        timings.clear();
    }
}

impl Default for TimingCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a single operation
#[derive(Debug, Clone)]
pub struct TimingStats {
    pub count: usize,
    pub total: Duration,
    pub mean: Duration,
    pub min: Duration,
    pub max: Duration,
}

impl TimingStats {
    fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self {
                count: 0,
                total: Duration::ZERO,
                mean: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
            };
        }

        let count = durations.len();
        let total: Duration = durations.iter().sum();
        let mean = total / count as u32;
        let min = durations.iter().min().copied().unwrap_or(Duration::ZERO);
        let max = durations.iter().max().copied().unwrap_or(Duration::ZERO);

        Self {
            count,
            total,
            mean,
            min,
            max,
        }
    }
}

/// Helper macro to time an operation and record it
#[macro_export]
macro_rules! time_operation {
    ($collector:expr, $name:expr, $block:block) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let duration = start.elapsed();
        $collector.record($name, duration);
        result
    }};
}

