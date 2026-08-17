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
//! Any FlexPrice API error fails open (allows the request) — a billing
//! provider outage must never itself drop inference traffic.

use std::sync::Arc;
use std::time::Duration;

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
        })
    }

    /// Whether `org_uuid` may proceed. Never blocks on a FlexPrice API
    /// failure — only a confirmed suspended/insufficient-balance result does.
    pub async fn check(&self, org_uuid: &str) -> BalanceStatus {
        if let Some(status) = self.cache.get(org_uuid).await {
            return status;
        }

        let status = match self.fetch(org_uuid).await {
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
        self.cache.insert(org_uuid.to_string(), status).await;
        status
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

    #[tokio::test]
    async fn postpaid_org_is_allowed_with_negative_balance() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/customers/wallets")
            .with_status(200)
            .with_body(r#"[{"balance":"-50.00","metadata":{"isKYC":"true"}}]"#)
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
        assert_eq!(checker.check("org-1").await, BalanceStatus::Ok);
    }

    #[tokio::test]
    async fn prepaid_org_with_negative_balance_is_blocked() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/customers/wallets")
            .with_status(200)
            .with_body(r#"[{"balance":"-1.00","metadata":{}}]"#)
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
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
            .with_status(200)
            .with_body(r#"[{"balance":"10.00","metadata":{}}]"#)
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
        assert_eq!(checker.check("org-1").await, BalanceStatus::Ok);
    }

    #[tokio::test]
    async fn suspended_org_is_blocked_even_if_postpaid() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/customers/wallets")
            .with_status(200)
            .with_body(
                r#"[{"balance":"100.00","metadata":{"isKYC":"true","isSuspended":"true"}}]"#,
            )
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
        assert_eq!(checker.check("org-1").await, BalanceStatus::Suspended);
    }

    #[tokio::test]
    async fn flexprice_api_error_fails_open() {
        // No mock registered — the request 404s against mockito's server.
        let server = mockito::Server::new_async().await;
        let checker = checker_for(&server, 0.0);
        assert_eq!(checker.check("org-1").await, BalanceStatus::Ok);
    }

    #[tokio::test]
    async fn result_is_cached_and_not_refetched() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/customers/wallets")
            .with_status(200)
            .with_body(r#"[{"balance":"10.00","metadata":{}}]"#)
            .expect(1)
            .create_async()
            .await;

        let checker = checker_for(&server, 0.0);
        assert_eq!(checker.check("org-1").await, BalanceStatus::Ok);
        assert_eq!(checker.check("org-1").await, BalanceStatus::Ok);
        mock.assert_async().await;
    }
}
