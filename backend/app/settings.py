"""Runtime settings read from environment variables (all optional; defaults are offline)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

EMBEDDERS = ("hashing",)
ANSWERERS = ("extractive",)


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _choice(env: Mapping[str, str], name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = env.get(name, "").strip().lower() or default
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    chunk_size: int = 120
    chunk_overlap: int = 20
    top_k: int = 4
    embedder: str = "hashing"
    answerer: str = "extractive"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from `env` (defaults to os.environ), validating values."""
        env = os.environ if env is None else env
        origins = tuple(o.strip() for o in env.get("CORS_ORIGINS", "").split(",") if o.strip())
        settings = cls(
            cors_origins=origins or cls.cors_origins,
            chunk_size=_int(env, "CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int(env, "CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_int(env, "TOP_K", cls.top_k),
            embedder=_choice(env, "EMBEDDER", cls.embedder, EMBEDDERS),
            answerer=_choice(env, "ANSWERER", cls.answerer, ANSWERERS),
        )
        if settings.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if not 0 <= settings.chunk_overlap < settings.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if not 1 <= settings.top_k <= 20:
            raise ValueError("TOP_K must be between 1 and 20")
        return settings
