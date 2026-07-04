"""Dong Feng serving layer.

``webplay`` is a dependency-free (stdlib ``http.server``) local web UI for playing
a human against any :class:`~dongfeng.protocol.engine.Engine` (random baseline or
the trained neural engine) in the browser. Launch it with ``dfc web``.

Richer serving (gRPC / batched HTTP inference) is planned for M6.
"""

from __future__ import annotations

from .metrics_exporter import serve_metrics
from .webplay import serve

__all__ = ["serve", "serve_metrics"]
