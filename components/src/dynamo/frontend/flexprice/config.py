import os
from dataclasses import dataclass
from typing import List


@dataclass
class FlexPriceConfig:
    """Configuration for the auth + FlexPrice billing layer.

    Auth and billing are handled natively by the Rust Dynamo HTTP service
    itself (lib/llm/src/http/service/flexprice/) whenever DYN_AUTH_ENABLED=true
    — it reads the same env vars this dataclass does, since it's embedded in
    this same process. By default that's the *only* thing that runs: Rust
    binds the public port directly, so this Python auth proxy is not started
    at all.

    The proxy exists purely as an opt-in fallback (DYN_FLEXPRICE_USE_PROXY=true)
    for the case where the native Rust path needs to be bypassed. Running both
    at once double-authenticates and double-bills every request — that's a bug,
    not a supported configuration — so leave DYN_FLEXPRICE_USE_PROXY unset
    unless you have a specific reason to route through this proxy instead.

    Auth env vars (required to activate either path):
        DYN_AUTH_ENABLED    - Enable JWT authentication (default: false)
        DYN_AUTH_SECRET_KEY - HMAC secret(s) for JWT validation, comma-separated for key rotation
        DYN_AUTH_VALID_ORGS - Comma-separated org UUID allowlist; empty = allow all authenticated orgs
        DYN_FLEXPRICE_USE_PROXY - Opt into this Python proxy fronting Rust instead of Rust
                                  binding the public port directly (default: false)

    FlexPrice env vars (optional; requires auth):
        DYN_FLEXPRICE_ENABLED              - Enable async usage event emission (default: false)
        DYN_FLEXPRICE_API_KEY              - FlexPrice API key (required when enabled)
        DYN_FLEXPRICE_API_HOST             - FlexPrice API host, e.g. "api.flexprice.io"
        DYN_FLEXPRICE_EVENT_NAME           - Override billing event name (default: "{model}-llm-usage")
        DYN_FLEXPRICE_SOURCE_NAME          - Override billing source name (default: "{deployment_name}_{deployment_id}")
        DYN_FLEXPRICE_INTERNAL_PORT_OFFSET - Port offset for the internal Dynamo HTTP service (default: 1)
        DYN_FLEXPRICE_QUEUE_SIZE           - Max pending usage events buffered by FlexPriceClient
                                              before new events are dropped (default: 4500). Read
                                              directly by FlexPriceClient, not part of this dataclass.
        DYN_FLEXPRICE_MINIMUM_BALANCE      - Minimum wallet balance required for a prepaid org to be
                                              allowed through (default: 0.0, i.e. block once negative).
                                              Postpaid orgs bypass this check entirely.
        DYN_DEPLOYMENT_NAME                - Human-readable deployment name, used in the default
                                              billing source (default: "dynamo")
        DYN_DEPLOYMENT_ID                  - Deployment/instance id, used in the default billing
                                              source (default: "local")
    """

    # Auth (master switch for the native Rust auth+billing path)
    auth_enabled: bool
    auth_secret_keys: List[str]
    auth_valid_orgs: List[str]
    # Opt-in: front Rust with this Python proxy instead of binding directly.
    use_proxy: bool

    # FlexPrice billing (optional; requires auth)
    enabled: bool
    api_key: str
    api_host: str
    event_name: str
    source_name: str
    internal_port_offset: int
    minimum_balance: float
    deployment_name: str
    deployment_id: str

    @classmethod
    def from_env(cls) -> "FlexPriceConfig":
        auth_enabled = os.environ.get("DYN_AUTH_ENABLED", "false").lower() in (
            "true", "1", "yes",
        )
        enabled = os.environ.get("DYN_FLEXPRICE_ENABLED", "false").lower() in (
            "true", "1", "yes",
        )
        use_proxy = os.environ.get("DYN_FLEXPRICE_USE_PROXY", "false").lower() in (
            "true", "1", "yes",
        )

        raw_keys = os.environ.get("DYN_AUTH_SECRET_KEY", "")
        auth_secret_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

        raw_orgs = os.environ.get("DYN_AUTH_VALID_ORGS", "")
        auth_valid_orgs = [o.strip() for o in raw_orgs.split(",") if o.strip()]

        return cls(
            auth_enabled=auth_enabled,
            auth_secret_keys=auth_secret_keys,
            auth_valid_orgs=auth_valid_orgs,
            use_proxy=use_proxy,
            enabled=enabled,
            api_key=os.environ.get("DYN_FLEXPRICE_API_KEY", ""),
            api_host=os.environ.get("DYN_FLEXPRICE_API_HOST", "").rstrip("/"),
            event_name=os.environ.get("DYN_FLEXPRICE_EVENT_NAME", ""),
            source_name=os.environ.get("DYN_FLEXPRICE_SOURCE_NAME", ""),
            internal_port_offset=int(
                os.environ.get("DYN_FLEXPRICE_INTERNAL_PORT_OFFSET", "1")
            ),
            minimum_balance=float(
                os.environ.get("DYN_FLEXPRICE_MINIMUM_BALANCE", "0.0")
            ),
            deployment_name=os.environ.get("DYN_DEPLOYMENT_NAME", "dynamo"),
            deployment_id=os.environ.get("DYN_DEPLOYMENT_ID", "local"),
        )

    def validate(self) -> None:
        if self.enabled and not self.auth_enabled:
            raise ValueError(
                "DYN_FLEXPRICE_ENABLED=true requires DYN_AUTH_ENABLED=true "
                "(org ID is sourced from the authenticated JWT)"
            )
        if self.auth_enabled and not self.auth_secret_keys:
            raise ValueError(
                "DYN_AUTH_SECRET_KEY is required when DYN_AUTH_ENABLED=true"
            )
        if self.enabled:
            if not self.api_key:
                raise ValueError(
                    "DYN_FLEXPRICE_API_KEY is required when DYN_FLEXPRICE_ENABLED=true"
                )
            if not self.api_host:
                raise ValueError(
                    "DYN_FLEXPRICE_API_HOST is required when DYN_FLEXPRICE_ENABLED=true"
                )
            if self.internal_port_offset < 1:
                raise ValueError("DYN_FLEXPRICE_INTERNAL_PORT_OFFSET must be >= 1")

    @property
    def proxy_required(self) -> bool:
        """True when the Python proxy layer must be inserted in front of Dynamo.

        Default is False even with auth enabled — the native Rust auth+billing
        path (lib/llm/src/http/service/flexprice/) handles it directly, since
        it reads the same env vars from this same process. This only becomes
        True when DYN_FLEXPRICE_USE_PROXY is explicitly set, as an opt-in
        fallback. Running both at once double-authenticates and double-bills
        every request, so this is deliberately not the default.
        """
        return self.auth_enabled and self.use_proxy

    def resolve_event_name(self, model_name: str = "") -> str:
        if self.event_name:
            return self.event_name
        return f"{model_name}-llm-usage" if model_name else "dynamo-llm-usage"

    def resolve_source_name(self) -> str:
        """Identifies which deployment served the request — mirrors go-proxy's
        ``{deployment_name}_{deployment_id}`` billing source. Deliberately not
        model-based: the model is already tracked separately as
        ``properties["model_id"]`` on the billing event.
        """
        if self.source_name:
            return self.source_name
        return f"{self.deployment_name}_{self.deployment_id}"
