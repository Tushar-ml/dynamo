// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wallet-balance gate for prepaid orgs, mirroring go-proxy's
//! `pkg/flexprice.CheckBalance`.
//!
//! Postpaid orgs (FlexPrice `metadata.isKYC == "true"`) are always allowed
//! through regardless of balance. Suspended orgs are always blocked. Prepaid
//! orgs are blocked once their wallet balance drops below
//! `DYN_FLEXPRICE_MINIMUM_BALANCE` (default `0.0`, i.e. "block once negative").
//!
//! Results are cached in-process for [`CACHE_TTL`] so a burst of requests
//! from the same org doesn't hit FlexPrice's wallet API once per request —
//! go-proxy uses a shared Redis cache for the same purpose across replicas;
//! dynamo has no Redis dependency today, so this is a per-pod equivalent.
//!
//! **[`check`](BalanceChecker::check) never awaits a live FlexPrice call.**
//! On a cache hit it returns immediately; on a miss (cold cache or expired
//! entry) it fails open for *this* request — same as an API error — and
//! kicks off a deduped background refresh so the next request for that org
//! sees the real cached status. This is deliberate: this check sits in the
//! request's hot path (auth middleware, before the inference call even
//! starts), so a live network round trip to FlexPrice inline here would add
//! that latency to every request whose cache entry is cold or has just
//! expired — unacceptable for a gate whose entire job is "don't slow down
//! the request path." A billing-provider outage or a slow wallet endpoint
//! must never itself add latency to (let alone drop) inference traffic.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use moka::future::Cache;
use reqwest::Client;
use serde::Deserialize;

use super::config::FlexPriceConfig;

const CACHE_TTL: Duration = Duration::from_secs(60);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalanceStatus {
    Ok,
    Suspended,
    InsufficientBalance,
}

#[derive(Debug, Default, Deserialize)]
struct WalletMetadata {
    #[serde(rename = "isKYC")]
    is_kyc: Option<String>,
    #[serde(rename = "isSuspended")]
    is_suspended: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Wallet {
    real_time_balance: Option<String>,
    balance: Option<String>,
    #[serde(default)]
    metadata: WalletMetadata,
}

fn is_truthy(value: &Option<String>) -> bool {
    value
        .as_deref()
        .is_some_and(|s| s.eq_ignore_ascii_case("true"))
}

pub struct BalanceChecker {
    client: Client,
    wallets_url: String,
    api_key: String,
    minimum_balance: f64,
    cache: Cache<String, BalanceStatus>,
    /// Orgs with a refresh currently in flight — de-dupes concurrent misses
    /// for the same org into a single background fetch instead of one per
    /// waiting request.
    in_flight: DashMap<String, ()>,
}

impl BalanceChecker {
    pub fn new(config: &FlexPriceConfig) -> Arc<Self> {
        Self::from_wallets_url(
            format!("https://{}/customers/wallets", config.api_host),
            config.api_key.clone(),
            config.minimum_balance,
        )
    }

    fn from_wallets_url(wallets_url: String, api_key: String, minimum_balance: f64) -> Arc<Self> {
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("failed to build FlexPrice wallet HTTP client");
        Arc::new(Self {
            client,
            wallets_url,
            api_key,
            minimum_balance,
            cache: Cache::builder().time_to_live(CACHE_TTL).build(),
            in_flight: DashMap::new(),
        })
    }

    /// Whether `org_uuid` may proceed. Returns instantly from cache; on a
    /// miss, fails open for this call and refreshes the cache in the
    /// background (see module docs) — never awaits a live FlexPrice call.
    pub async fn check(self: &Arc<Self>, org_uuid: &str) -> BalanceStatus {
        if let Some(status) = self.cache.get(org_uuid).await {
            return status;
        }

        // Only the first miss for this org spawns a refresh; concurrent
        // misses for the same org while it's in flight just fail open too.
        if self.in_flight.insert(org_uuid.to_string(), ()).is_none() {
            let this = self.clone();
            let org_uuid = org_uuid.to_string();
            tokio::spawn(async move {
                let status = match this.fetch(&org_uuid).await {
                    Ok(status) => status,
                    Err(error) => {
                        tracing::warn!(
                            org = %org_uuid,
                            %error,
                            "FlexPrice wallet lookup failed; allowing request"
                        );
                        BalanceStatus::Ok
                    }
                };
                this.cache.insert(org_uuid.clone(), status).await;
                this.in_flight.remove(&org_uuid);
            });
        }
        BalanceStatus::Ok
    }

    async fn fetch(&self, org_uuid: &str) -> anyhow::Result<BalanceStatus> {
        let resp = self
            .client
            .get(&self.wallets_url)
            .header("x-api-key", &self.api_key)
            .query(&[
                ("lookup_key", org_uuid),
                ("include_real_time_balance", "true"),
            ])
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!("FlexPrice wallet lookup returned {}", resp.status());
        }

        let wallets: Vec<Wallet> = resp.json().await?;
        let wallet = wallets
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no wallet found for org {org_uuid}"))?;

        if is_truthy(&wallet.metadata.is_suspended) {
            return Ok(BalanceStatus::Suspended);
        }
        if is_truthy(&wallet.metadata.is_kyc) {
            return Ok(BalanceStatus::Ok);
        }

        let balance_str = wallet
            .real_time_balance
            .filter(|s| !s.is_empty())
            .or(wallet.balance)
            .ok_or_else(|| anyhow::anyhow!("wallet response missing balance"))?;
        let balance: f64 = balance_str.parse()?;

        if balance >= self.minimum_balance {
            Ok(BalanceStatus::Ok)
        } else {
            Ok(BalanceStatus::InsufficientBalance)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker_for(server: &mockito::ServerGuard, minimum_balance: f64) -> Arc<BalanceChecker> {
        BalanceChecker::from_wallets_url(
            format!("{}/customers/wallets", server.url()),
            "test-key".to_string(),
            minimum_balance,
        )
    }

    /// Polls the cache until the background refresh lands, or panics after
    /// a generous timeout — deterministic (no arbitrary fixed sleep) without
    /// reaching into the checker's private scheduling.
    async fn wait_for_cached(checker: &Arc<BalanceChecker>, org_uuid: &str) -> BalanceStatus {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(status) = checker.cache.get(org_uuid).await {
                    return status;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("background refresh never populated the cache")
    }

    #[tokio::test]
    async fn cold_cache_fails_open_immediately_regardless_of_real_balance() {
        // Balance here is negative and would normally block — but on a cold
        // cache, check() must fail open for *this* call without waiting on
        // the fetch, which is the whole point of the redesign.
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/customers/wallets")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(r#"[{"balance":"-1.00","metadata":{}}]"#)
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
        assert_eq!(checker.check("org-1").await, BalanceStatus::Ok);
    }

    #[tokio::test]
    async fn postpaid_org_is_allowed_with_negative_balance() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/customers/wallets")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(r#"[{"balance":"-50.00","metadata":{"isKYC":"true"}}]"#)
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
        checker.check("org-1").await; // trigger background refresh
        assert_eq!(wait_for_cached(&checker, "org-1").await, BalanceStatus::Ok);
    }

    #[tokio::test]
    async fn prepaid_org_with_negative_balance_is_blocked_once_cache_populates() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/customers/wallets")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(r#"[{"balance":"-1.00","metadata":{}}]"#)
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
        checker.check("org-1").await; // trigger background refresh
        assert_eq!(
            wait_for_cached(&checker, "org-1").await,
            BalanceStatus::InsufficientBalance
        );
        // Now that the cache is warm, the gate actually takes effect.
        assert_eq!(
            checker.check("org-1").await,
            BalanceStatus::InsufficientBalance
        );
    }

    #[tokio::test]
    async fn prepaid_org_with_positive_balance_is_allowed() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/customers/wallets")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(r#"[{"balance":"10.00","metadata":{}}]"#)
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
        checker.check("org-1").await;
        assert_eq!(wait_for_cached(&checker, "org-1").await, BalanceStatus::Ok);
    }

    #[tokio::test]
    async fn suspended_org_is_blocked_once_cache_populates() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/customers/wallets")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(
                r#"[{"balance":"100.00","metadata":{"isKYC":"true","isSuspended":"true"}}]"#,
            )
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
        checker.check("org-1").await;
        assert_eq!(
            wait_for_cached(&checker, "org-1").await,
            BalanceStatus::Suspended
        );
    }

    #[tokio::test]
    async fn flexprice_api_error_fails_open() {
        // No mock registered — the request 404s against mockito's server.
        let server = mockito::Server::new_async().await;
        let checker = checker_for(&server, 0.0);
        checker.check("org-1").await;
        assert_eq!(wait_for_cached(&checker, "org-1").await, BalanceStatus::Ok);
    }

    #[tokio::test]
    async fn result_is_cached_and_not_refetched() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/customers/wallets")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(r#"[{"balance":"10.00","metadata":{}}]"#)
            .expect(1)
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
        checker.check("org-1").await;
        wait_for_cached(&checker, "org-1").await;
        assert_eq!(checker.check("org-1").await, BalanceStatus::Ok);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn concurrent_misses_for_the_same_org_dedupe_to_one_fetch() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/customers/wallets")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(r#"[{"balance":"10.00","metadata":{}}]"#)
            .expect(1)
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
        let (a, b, c) = tokio::join!(
            checker.check("org-1"),
            checker.check("org-1"),
            checker.check("org-1"),
        );
        // All three fail open immediately — none of them waited on the fetch.
        assert_eq!((a, b, c), (BalanceStatus::Ok, BalanceStatus::Ok, BalanceStatus::Ok));
        wait_for_cached(&checker, "org-1").await;
        mock.assert_async().await;
    }
}
