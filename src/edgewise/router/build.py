"""Assemble a Router from configuration: load the local model and set up the escalation tier."""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import urlparse

from edgewise.backends import Backend, get_backend
from edgewise.catalog import Catalog, estimate_memory
from edgewise.config import Config
from edgewise.hardware import detect
from edgewise.router.core import Router, Tier
from edgewise.router.rules import Connectivity
from edgewise.store import Store

# Values of router.remote_model that select the [remote] OpenAI-compatible endpoint.
OPENAI_COMPATIBLE = ("openrouter", "openai-compatible")


def _is_local(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1")


def build_router(
    config: Config,
    catalog: Catalog,
    store: Store | None = None,
    budget_gb: float | None = None,
    backend_for: Callable[[str], Backend] = get_backend,
) -> Router:
    """Raises ValueError (before loading anything) if the models won't fit in memory."""
    rc = config.router
    local_variant = catalog.get(rc.local_model)
    remote_variant = None
    if rc.remote_model not in ("cloud", "none", *OPENAI_COMPATIBLE):
        remote_variant = catalog.get(rc.remote_model)

    # Two on-device models share one memory budget; check before loading either.
    needed = estimate_memory(local_variant).total_gb
    if remote_variant is not None:
        needed += estimate_memory(remote_variant).total_gb
    budget = budget_gb if budget_gb is not None else detect().model_budget_gb
    if needed > budget:
        both = f" and {remote_variant.id}" if remote_variant else ""
        raise ValueError(
            f"{local_variant.id}{both} need ~{needed:.1f} GB together, "
            f"but the memory budget is {budget:.1f} GB"
        )

    local_backend = backend_for(local_variant.backend)
    local = Tier(local_backend, local_backend.load(local_variant))

    remote = None
    is_online: Callable[[], bool] = lambda: True  # noqa: E731
    if rc.remote_model == "cloud":
        from edgewise.backends.cloud import CloudBackend, cloud_variant

        cloud = CloudBackend(effort=config.cloud.effort)
        remote = Tier(
            cloud,
            cloud.load(cloud_variant(config.cloud.model)),
            input_price=config.cloud.input_price_per_mtok,
            output_price=config.cloud.output_price_per_mtok,
            needs_network=True,
        )
        is_online = Connectivity()
    elif rc.remote_model in OPENAI_COMPATIBLE:
        from edgewise.backends import openai_compat

        endpoint = openai_compat.OpenAICompatBackend(
            config.remote.base_url, config.remote.api_key_env
        )
        remote = Tier(
            endpoint,
            endpoint.load(openai_compat.remote_variant(config.remote.model)),
            input_price=config.remote.input_price_per_mtok,
            output_price=config.remote.output_price_per_mtok,
            # A server on this machine is reachable offline; anything else needs the network.
            needs_network=not _is_local(config.remote.base_url),
        )
        if remote.needs_network:
            is_online = Connectivity()
    elif remote_variant is not None:
        remote_backend = (
            local_backend
            if remote_variant.backend == local_variant.backend
            else backend_for(remote_variant.backend)
        )
        remote = Tier(remote_backend, remote_backend.load(remote_variant))

    return Router(
        local,
        remote,
        policy=rc.policy,
        min_logprob=rc.min_logprob,
        is_online=is_online,
        store=store,
    )
