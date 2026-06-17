// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fire-and-forget HTTP client for the ss-agent metrics relay service.
//!
//! Reads `METRICS_RELAY_ADDR` at first use. When set, emits TTFT, input/output/total
//! throughput with a `streaming` label via `tokio::spawn` so the hot path is never
//! blocked. Up to `MAX_RETRIES` retries on connection errors or 5xx responses.

use std::sync::OnceLock;
use std::time::Duration;

const MAX_RETRIES: u32 = 3;
const RETRY_DELAYS_MS: [u64; 3] = [100, 300, 900];

static RELAY: OnceLock<Option<RelayState>> = OnceLock::new();

struct RelayState {
    client: reqwest::Client,
    url: String,
    deployment_slug: Option<String>,
    namespace: Option<String>,
}

fn get_relay() -> Option<&'static RelayState> {
    RELAY
        .get_or_init(|| {
            let addr = std::env::var("METRICS_RELAY_ADDR").ok()?;
            let addr = addr.trim().trim_end_matches('/').to_string();
            if addr.is_empty() {
                return None;
            }

            let skip_tls = std::env::var("METRICS_RELAY_SKIP_TLS_VERIFY")
                .ok()
                .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
                .unwrap_or(false);

            let mut builder = reqwest::Client::builder()
                .timeout(Duration::from_secs(2))
                .danger_accept_invalid_certs(skip_tls);

            // reqwest with rustls-tls needs an explicit TLS config when skip_tls is false.
            // The default is fine; only override when skipping verification.
            if skip_tls {
                builder = builder.danger_accept_invalid_hostnames(true);
            }

            let client = match builder.build() {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("metrics relay: failed to build HTTP client: {}", e);
                    return None;
                }
            };

            tracing::info!(
                addr = %addr,
                skip_tls_verify = skip_tls,
                "metrics relay: client initialized"
            );

            let deployment_slug = std::env::var("METRICS_DEPLOYMENT_SLUG")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());

            let namespace = std::env::var("NAMESPACE")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());

            Some(RelayState {
                url: format!("{}/custom-metric", addr),
                client,
                deployment_slug,
                namespace,
            })
        })
        .as_ref()
}

fn resolve_deployment<'a>(model: &'a str, relay: &'a RelayState) -> &'a str {
    if !model.is_empty() {
        return model;
    }
    if let Some(s) = &relay.deployment_slug {
        return s.as_str();
    }
    if let Some(ns) = &relay.namespace {
        return ns.as_str();
    }
    "unknown"
}

/// Emit TTFT + throughput metrics to the relay. Fire-and-forget; never blocks the caller.
///
/// `ttft_ms`: measured first-token time. For streaming, this is the true TTFT; for
/// non-streaming callers should pass `None` and the total elapsed is used instead.
pub fn emit(
    model: &str,
    streaming: bool,
    ttft_ms: Option<f64>,
    isl: usize,
    osl: usize,
    total_elapsed_secs: f64,
) {
    let Some(relay) = get_relay() else { return };
    // Only emit when the request actually produced token output.
    if osl == 0 {
        return;
    }

    let deployment = resolve_deployment(model, relay).to_string();
    let client = relay.client.clone();
    let url = relay.url.clone();

    let ttft_val_ms = if streaming {
        ttft_ms.unwrap_or(0.0).round() as u64
    } else {
        (total_elapsed_secs * 1000.0).round() as u64
    };

    let ttft_sec = ttft_val_ms as f64 / 1000.0;
    let input_denom = if streaming {
        ttft_sec.max(1e-9)
    } else {
        total_elapsed_secs.max(1e-9)
    };
    let output_denom = if streaming {
        (total_elapsed_secs - ttft_sec).max(1e-9)
    } else {
        total_elapsed_secs.max(1e-9)
    };
    let total_denom = total_elapsed_secs.max(1e-9);

    let mut metrics: Vec<(&'static str, u64)> = vec![("ttft", ttft_val_ms)];
    if isl > 0 {
        metrics.push(("input_throughput", (isl as f64 / input_denom).round() as u64));
    }
    metrics.push(("output_throughput", (osl as f64 / output_denom).round() as u64));
    let total_tokens = isl + osl;
    metrics.push(("total_throughput", (total_tokens as f64 / total_denom).round() as u64));

    let handle = match tokio::runtime::Handle::try_current() {
        Ok(h) => h,
        Err(_) => {
            tracing::debug!("metrics relay: no tokio runtime available, metrics dropped");
            return;
        }
    };

    handle.spawn(async move {
        for (metric_type, value) in metrics {
            let payload = serde_json::json!({
                "metric_type": metric_type,
                "deployment": &deployment,
                "key": format!("dynamo_frontend_{}", metric_type),
                "value": value,
                "metadata": { "streaming": streaming },
            });

            let mut succeeded = false;
            for attempt in 0..=MAX_RETRIES {
                if attempt > 0 {
                    tokio::time::sleep(Duration::from_millis(
                        RETRY_DELAYS_MS[(attempt - 1) as usize],
                    ))
                    .await;
                    tracing::debug!(
                        "metrics relay: retry {}/{} for {}",
                        attempt,
                        MAX_RETRIES,
                        metric_type
                    );
                }
                match client.post(&url).json(&payload).send().await {
                    Ok(resp) if resp.status().is_server_error() => {
                        tracing::warn!(
                            "metrics relay: server error {} (attempt {}/{}) for {}",
                            resp.status(),
                            attempt + 1,
                            MAX_RETRIES + 1,
                            metric_type
                        );
                    }
                    Ok(resp) => {
                        if resp.status().is_client_error() {
                            tracing::warn!(
                                "metrics relay: client error {} for {} (payload={:?})",
                                resp.status(),
                                metric_type,
                                payload
                            );
                        } else {
                            tracing::info!(
                                "metrics relay: posted {} → HTTP {}",
                                metric_type,
                                resp.status(),
                            );
                        }
                        succeeded = true;
                        break;
                    }
                    Err(e) => {
                        tracing::warn!(
                            "metrics relay: post failed (attempt {}/{}): {}",
                            attempt + 1,
                            MAX_RETRIES + 1,
                            e
                        );
                    }
                }
            }
            if !succeeded {
                tracing::warn!(
                    "metrics relay: gave up after {} attempts for {}",
                    MAX_RETRIES + 1,
                    metric_type
                );
            }
        }
    });
}
