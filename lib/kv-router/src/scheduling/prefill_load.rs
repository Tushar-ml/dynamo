// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use crate::protocols::PrefillLoadHint;

pub trait PrefillLoadEstimator: Send + Sync {
    fn predict_prefill_duration(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> anyhow::Result<Duration>;
}

/// A built-in `PrefillLoadEstimator` that needs no external performance model.
///
/// `duration = fixed_overhead + effective_isl / tokens_per_second`
///
/// Why this exists: the modeled prefill backlog (`modeled_remaining_prefill_time_ms_at`) is the
/// router's only estimate of in-worker prefill wait, and it is `Err(MissingExpectedDuration)`
/// unless *some* estimator is configured. The only other implementation is the AIC callback, which
/// needs a Python perf model and an `aic_perf_config`. For a single pinned model on fixed hardware
/// a one-parameter linear fit is enough to rank and threshold wait times, which is all the
/// conditional-disagg gate needs -- it never needs an absolute latency prediction.
///
/// `tokens_per_second` is a MEASURED property of the deployment, not a guess. On dsv41-flash /
/// 8xB300 with 4 DP prefill ranks the drain rate was measured at ~175k tok/s at
/// `--max-num-batched-tokens 8192` and ~194k at 16384 (exp-0088), by differencing the router's own
/// `active_prefill_tokens` gauge over a saturated replay. Re-measure after any change to batch
/// size, parallelism or hardware; a wrong constant biases every predicted wait by the same factor,
/// which shifts the effective SLO threshold rather than breaking the ordering.
#[derive(Debug, Clone, Copy)]
pub struct LinearPrefillLoadEstimator {
    tokens_per_second: f64,
    fixed_overhead: Duration,
}

impl LinearPrefillLoadEstimator {
    /// `tokens_per_second` must be finite and > 0; anything else is a configuration error rather
    /// than something to silently paper over, because a zero rate yields an infinite predicted
    /// wait and would trip every gate.
    pub fn new(tokens_per_second: f64, fixed_overhead: Duration) -> anyhow::Result<Self> {
        if !tokens_per_second.is_finite() || tokens_per_second <= 0.0 {
            anyhow::bail!(
                "prefill linear load model: tokens_per_second must be finite and > 0, got {tokens_per_second}"
            );
        }
        Ok(Self {
            tokens_per_second,
            fixed_overhead,
        })
    }
}

impl PrefillLoadEstimator for LinearPrefillLoadEstimator {
    fn predict_prefill_duration(
        &self,
        _batch_size: usize,
        effective_isl: usize,
        _prefix: usize,
    ) -> anyhow::Result<Duration> {
        // Deliberately ignores batch_size and prefix: effective_isl is already ISL minus cached,
        // and on a dedicated prefill worker the per-step batching is what the rate constant
        // absorbs. Modelling them separately would invent precision the fit does not have.
        Ok(self.fixed_overhead
            + Duration::from_secs_f64(effective_isl as f64 / self.tokens_per_second))
    }
}

pub fn effective_prefill_tokens(isl_tokens: usize, weighted_cached_tokens: usize) -> usize {
    isl_tokens.saturating_sub(weighted_cached_tokens.min(isl_tokens))
}

pub fn prefill_load_hint_from_effective_tokens(
    isl_tokens: usize,
    effective_prefill_tokens: usize,
) -> Result<PrefillLoadHint, InvalidEffectivePrefillTokens> {
    if effective_prefill_tokens > isl_tokens {
        return Err(InvalidEffectivePrefillTokens {
            effective_prefill_tokens,
            isl_tokens,
        });
    }

    Ok(PrefillLoadHint {
        initial_effective_prefill_tokens: effective_prefill_tokens,
        expected_prefill_duration: None,
    })
}

#[derive(Debug, thiserror::Error)]
#[error(
    "effective_prefill_tokens ({effective_prefill_tokens}) must not exceed isl_tokens ({isl_tokens})"
)]
pub struct InvalidEffectivePrefillTokens {
    pub effective_prefill_tokens: usize,
    pub isl_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_prefill_clamps_weighted_cache_credit() {
        assert_eq!(effective_prefill_tokens(100, 40), 60);
        assert_eq!(effective_prefill_tokens(100, 120), 0);
    }

    #[test]
    fn direct_hint_validates_against_isl() {
        let hint = prefill_load_hint_from_effective_tokens(100, 60).unwrap();
        assert_eq!(hint.initial_effective_prefill_tokens, 60);
        assert!(prefill_load_hint_from_effective_tokens(100, 101).is_err());
    }
}
