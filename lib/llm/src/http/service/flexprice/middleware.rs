// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum middleware gating inference routes behind JWT auth.
//!
//! Only layered onto `inference_router` when `DYN_AUTH_ENABLED=true` (see
//! `service_v2.rs`) — system routes (health/live/metrics/models) never pass
//! through this middleware, and it adds zero overhead when auth is disabled
//! since the layer itself isn't added to the router in that case.

use std::sync::Arc;

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use super::auth;
use crate::http::service::service_v2;

/// The JWT-verified org UUID. Inserted into the request's typed extension map
/// by [`auth_middleware`] on successful auth; billed handlers read it back
/// out via `Option<axum::extract::Extension<OrgUuid>>` — `None` whenever auth
/// is disabled or this particular request never passed through the
/// middleware, so handlers stay a no-op-safe shell in that case.
#[derive(Debug, Clone)]
pub struct OrgUuid(pub String);

pub async fn auth_middleware(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
    mut request: Request,
    next: Next,
) -> Response {
    let auth_header = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();

    let auth_config = state.auth_config();
    match auth::authenticate(
        &auth_header,
        &auth_config.secret_keys,
        &auth_config.valid_orgs,
    ) {
        Ok(ctx) => {
            // Request extensions are a server-internal typed map, never
            // populated from client input, so there's no spoofing vector to
            // guard against here (unlike a header-based propagation scheme).
            request.extensions_mut().insert(OrgUuid(ctx.org_uuid));
            next.run(request).await
        }
        Err(err) => err.into_response(),
    }
}
