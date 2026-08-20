// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async client that emits LLM usage events to FlexPrice in the background.
//!
//! `enqueue` is non-blocking — the caller returns immediately and a background
//! worker drains the queue independently, so billing never adds latency to
//! the request path (enqueue is a plain channel push; it never awaits network
//! I/O). The queue is bounded (default 4500, override via
//! `DYN_FLEXPRICE_QUEUE_SIZE`); events are only dropped when either:
//!   - the queue is full, i.e. events are arriving faster than
//!     `MAX_CONCURRENT_SENDS` in-flight POSTs can drain them, or
//!   - a single event's POST still fails after `MAX_ATTEMPTS` retries with
//!     backoff (persistent failure — the closest local proxy for "FlexPrice
//!     is down" without a live health check).
//! A lone transient error (timeout, connection reset, one 5xx) does *not*
//! drop an event — it's retried — and draining up to `MAX_CONCURRENT_SENDS`
//! events concurrently means throughput isn't capped at one request-per-RTT,
//! so sustained billed-request load doesn't overflow the queue on its own.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::config::environment_names::llm as env_llm;
use reqwest::Client;
use serde::Serialize;
use tokio::sync::{Semaphore, mpsc};

const EVENTS_PATH: &str = "/events";
const DEFAULT_QUEUE_SIZE: usize = 4500;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Max pending usage events buffered before new events are dropped.
/// Overridable via `DYN_FLEXPRICE_QUEUE_SIZE`.
fn queue_size() -> usize {
    std::env::var(env_llm::DYN_FLEXPRICE_QUEUE_SIZE)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_QUEUE_SIZE)
}
/// Cap on in-flight POSTs so a burst of billed requests drains faster than
/// one-at-a-time (which would otherwise cap throughput at 1/RTT), without
/// spawning unbounded concurrent tasks.
const MAX_CONCURRENT_SENDS: usize = 20;
/// Total attempts (including the first) before giving up on a single event.
const MAX_ATTEMPTS: u32 = 3;
const RETRY_BASE_DELAY: Duration = Duration::from_millis(200);

#[derive(Debug, Serialize)]
struct UsageEvent {
    event_name: String,
    external_customer_id: String,
    properties: BTreeMap<String, String>,
    source: String,
    event_id: String,
    timestamp: String,
}

pub struct FlexPriceClient {
    tx: mpsc::Sender<UsageEvent>,
}

impl FlexPriceClient {
    /// Build the client and spawn its background drain worker.
    pub fn new(api_host: &str, api_key: &str) -> Arc<Self> {
        let events_url = format!("https://{api_host}{EVENTS_PATH}");
        let api_key = api_key.to_string();
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("failed to build FlexPrice HTTP client");

        let (tx, rx) = mpsc::channel::<UsageEvent>(queue_size());
        tokio::spawn(Self::worker(client, events_url, api_key, rx));

        Arc::new(Self { tx })
    }

    /// Non-blocking enqueue. Drops (with a warning) when the queue is full or
    /// the worker task is gone.
    pub fn enqueue(
        &self,
        event_name: String,
        external_customer_id: String,
        properties: BTreeMap<String, String>,
        source: String,
    ) {
        let event = UsageEvent {
            event_name,
            external_customer_id: external_customer_id.clone(),
            properties,
            source,
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        };

        match self.tx.try_send(event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!(
                    customer = %external_customer_id,
                    "FlexPrice event queue full; dropping event"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::warn!("FlexPrice event worker is not running; dropping event");
            }
        }
    }

    /// Drains the queue, dispatching up to `MAX_CONCURRENT_SENDS` POSTs at
    /// once so throughput isn't serialized behind one request-per-RTT.
    /// Acquiring a permit before spawning means backpressure (waiting for a
    /// free slot) happens here, off the request path — never a dropped event
    /// on its own.
    async fn worker(
        client: Client,
        events_url: String,
        api_key: String,
        mut rx: mpsc::Receiver<UsageEvent>,
    ) {
        let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_SENDS));
        while let Some(event) = rx.recv().await {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .expect("semaphore is never closed");
            let client = client.clone();
            let events_url = events_url.clone();
            let api_key = api_key.clone();
            tokio::spawn(async move {
                let _permit = permit;
                Self::send_with_retry(&client, &events_url, &api_key, &event).await;
            });
        }
    }

    /// Sends one event, retrying transient failures (network errors, 5xx)
    /// with backoff. Only gives up — dropping the event — after
    /// `MAX_ATTEMPTS` consecutive failures.
    async fn send_with_retry(client: &Client, events_url: &str, api_key: &str, event: &UsageEvent) {
        for attempt in 1..=MAX_ATTEMPTS {
            let result = client
                .post(events_url)
                .header("x-api-key", api_key)
                .json(event)
                .send()
                .await;

            let retryable = match &result {
                Ok(resp) if resp.status().is_success() => return,
                Ok(resp) => Some(resp.status().to_string()),
                Err(_) => None,
            };

            if attempt == MAX_ATTEMPTS {
                match retryable {
                    Some(status) => tracing::warn!(
                        status = %status,
                        event_name = %event.event_name,
                        attempts = attempt,
                        "FlexPrice API returned a non-success status after retries; dropping event"
                    ),
                    None => tracing::warn!(
                        error = %result.unwrap_err(),
                        event_name = %event.event_name,
                        attempts = attempt,
                        "Failed to emit FlexPrice event after retries; dropping event"
                    ),
                }
                return;
            }

            tracing::debug!(
                event_name = %event.event_name,
                attempt,
                "FlexPrice event send failed, retrying"
            );
            tokio::time::sleep(RETRY_BASE_DELAY * attempt).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn enqueue_does_not_panic_without_a_reachable_endpoint() {
        // No real network call should ever block or panic the caller — the
        // worker will simply log a warning when the POST fails.
        let client = FlexPriceClient::new("localhost:1", "test-key");
        let mut properties = BTreeMap::new();
        properties.insert("model_id".to_string(), "test-model".to_string());
        client.enqueue(
            "test-event".to_string(),
            "org-1".to_string(),
            properties,
            "test-model".to_string(),
        );
        // Give the worker a chance to run without asserting on network outcome.
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    fn test_event() -> UsageEvent {
        UsageEvent {
            event_name: "test-event".to_string(),
            external_customer_id: "org-1".to_string(),
            properties: BTreeMap::new(),
            source: "test-model".to_string(),
            event_id: "event-1".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    #[tokio::test]
    async fn transient_failure_is_retried_not_dropped() {
        let mut server = mockito::Server::new_async().await;
        // Fails the first two attempts, succeeds on the third — proves a
        // couple of transient errors don't drop the event.
        let mock = server
            .mock("POST", "/events")
            .with_status(500)
            .expect(2)
            .create_async()
            .await;
        let mock_ok = server
            .mock("POST", "/events")
            .with_status(200)
            .expect(1)
            .create_async()
            .await;

        let client = Client::new();
        let url = format!("{}/events", server.url());
        FlexPriceClient::send_with_retry(&client, &url, "test-key", &test_event()).await;

        mock.assert_async().await;
        mock_ok.assert_async().await;
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts() {
        let mut server = mockito::Server::new_async().await;
        // Always fails — the event should be dropped only after every
        // attempt is exhausted, never on the first failure alone.
        let mock = server
            .mock("POST", "/events")
            .with_status(500)
            .expect(MAX_ATTEMPTS as usize)
            .create_async()
            .await;

        let client = Client::new();
        let url = format!("{}/events", server.url());
        FlexPriceClient::send_with_retry(&client, &url, "test-key", &test_event()).await;

        mock.assert_async().await;
    }
}
