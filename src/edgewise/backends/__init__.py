"""Inference backends. Import concrete backends lazily: MLX is an optional dependency."""

from __future__ import annotations

from edgewise.backends.base import Backend, LoadedModel, TokenStream
from edgewise.schema import BackendName

__all__ = ["Backend", "LoadedModel", "TokenStream", "get_backend"]


def get_backend(name: BackendName) -> Backend:
    if name == "mlx":
        from edgewise.backends.mlx import MLXBackend

        return MLXBackend()
    if name == "fake":
        from edgewise.backends.fake import FakeBackend

        return FakeBackend()
    raise NotImplementedError(f"backend {name!r} is not implemented yet")
