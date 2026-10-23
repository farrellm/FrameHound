"""Every setting is an env var with a documented default (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    return int(float(_env(name, str(default))))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "/data")))
    # Movie files (source, proxy, partial torrents, clips): big and read in order, so
    # they can live on a slower disk than DATA_DIR, which every search reads.
    media_dir: Path = field(default_factory=lambda: Path(_env("MEDIA_DIR", "")
                                                         or _env("DATA_DIR", "/data")))
    # A folder of movie files on the server to add from (read only); empty = off.
    import_dir: Path | None = field(default_factory=lambda: Path(_env("IMPORT_DIR", ""))
                                    if _env("IMPORT_DIR", "") else None)
    # Empty REDIS_URL runs jobs in background threads inside the api (dev mode).
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))
    index_backend: str = field(default_factory=lambda: _env("INDEX_BACKEND", "colab"))
    max_duration_s: float = field(default_factory=lambda: _float("MAX_DURATION", 4 * 3600))
    max_bytes: int = field(default_factory=lambda: _int("MAX_BYTES", 40 * 1024**3))
    index_timeout_s: int = field(default_factory=lambda: _int("INDEX_TIMEOUT", 3600))
    # The local backend runs on a CPU, where a feature takes 1-2 h.
    local_index_timeout_s: int = field(default_factory=lambda: _int("LOCAL_INDEX_TIMEOUT", 4 * 3600))
    stage_watchdog_s: int = field(default_factory=lambda: _int("STAGE_WATCHDOG", 600))
    max_clip_s: float = field(default_factory=lambda: _float("MAX_CLIP_S", 120))
    clip_ttl_s: int = field(default_factory=lambda: _int("CLIP_TTL", 24 * 3600))
    heat_bins: int = field(default_factory=lambda: _int("HEAT_BINS", 512))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 10))
    z_min: float = field(default_factory=lambda: _float("Z_MIN", 3.0))
    # Show a picture (design.md §6): a face match scores Z_MIN + FACE_BONUS at the
    # lock's match similarity, plus FACE_SCALE per unit of similarity above it.
    face_bonus: float = field(default_factory=lambda: _float("FACE_BONUS", 2.0))
    face_scale: float = field(default_factory=lambda: _float("FACE_SCALE", 20.0))
    # Unload the visual and face models after this long unused (they reload on the
    # next query that needs them, in a few seconds); 0 keeps them loaded.
    encoder_idle_s: float = field(default_factory=lambda: _float("ENCODER_IDLE_S", 900))
    max_query_image_mb: float = field(default_factory=lambda: _float("MAX_QUERY_IMAGE_MB", 15))
    # Dev/test only: lets downloads reach private addresses (e.g. a local http.server).
    allow_private_urls: bool = field(default_factory=lambda: _bool("FH_ALLOW_PRIVATE_URLS", False))
    ffmpeg_cpu_s: int = field(default_factory=lambda: _int("FFMPEG_CPU_S", 6 * 3600))
    ffmpeg_mem_mb: int = field(default_factory=lambda: _int("FFMPEG_MEM_MB", 6144))
    ffmpeg_threads: int = field(default_factory=lambda: _int("FFMPEG_THREADS", 0))
    # "nvenc" makes the proxy on an NVIDIA GPU (docker-compose.gpu.yml), falling
    # back to libx264; CUDA needs a larger address-space limit than FFMPEG_MEM_MB.
    transcode_hwaccel: str = field(default_factory=lambda: _env("TRANSCODE_HWACCEL", "none"))
    ffmpeg_gpu_mem_mb: int = field(default_factory=lambda: _int("FFMPEG_GPU_MEM_MB", 16384))
    index_zstd_level: int = field(default_factory=lambda: _int("INDEX_ZSTD_LEVEL", 3))
    stale_job_s: int = field(default_factory=lambda: _int("STALE_JOB_S", 300))
    # Colab (design.md §5, §10).
    colab_bin: str = field(default_factory=lambda: _env("COLAB_BIN", "colab"))
    # Empty uses the CLI's own default strategy (oauth2); "adc" for a mounted gcloud ADC.
    colab_auth: str = field(default_factory=lambda: _env("COLAB_AUTH", ""))
    colab_gpu: str = field(default_factory=lambda: _env("COLAB_GPU", "L4"))
    gpu_fallback: tuple[str, ...] = field(default_factory=lambda: tuple(
        g.strip() for g in _env("GPU_FALLBACK", "L4,T4").split(",") if g.strip()))
    min_balance_cu: float = field(default_factory=lambda: _float("MIN_BALANCE_CU", 5))
    monthly_budget_cu: float = field(default_factory=lambda: _float("MONTHLY_BUDGET_CU", 40))
    max_colab_sessions: int = field(default_factory=lambda: _int("MAX_COLAB_SESSIONS", 1))
    colab_remote_root: str = field(default_factory=lambda: _env("COLAB_REMOTE_ROOT",
                                                                "/content/fh"))
    colab_poll_s: float = field(default_factory=lambda: _float("COLAB_POLL_S", 10))
    colab_retries: int = field(default_factory=lambda: _int("COLAB_RETRIES", 2))
    reconcile_s: int = field(default_factory=lambda: _int("RECONCILE_S", 900))
    # Torrents (design.md §4): host only, the main video file only.
    seed_ratio: float = field(default_factory=lambda: _float("SEED_RATIO", 0))
    seed_max_s: int = field(default_factory=lambda: _int("SEED_MAX_S", 3600))
    # io jobs (downloads, proxies, clips) at once: worker-io runs this many RQ
    # workers, and each torrent listens on its own port from TORRENT_LISTEN up.
    max_downloads: int = field(default_factory=lambda: max(1, _int("MAX_DOWNLOADS", 3)))
    # Fail a torrent after this long with no peers or no progress; 0 waits forever.
    torrent_stall_s: float = field(default_factory=lambda: _float("TORRENT_STALL_S", 0))
    torrent_listen: str = field(default_factory=lambda: _env(
        "TORRENT_LISTEN", "0.0.0.0:6881,[::]:6881"))
    max_torrent_kb: int = field(default_factory=lambda: _int("MAX_TORRENT_KB", 2048))
    # Dev/test only: peers every torrent download connects to ("host:port,...").
    torrent_peers: tuple[str, ...] = field(default_factory=lambda: tuple(
        p.strip() for p in _env("FH_TORRENT_PEERS", "").split(",") if p.strip()))
    # Bitmagnet (a local DHT crawler) for finding movies by name; empty = off.
    bitmagnet_url: str = field(default_factory=lambda: _env("BITMAGNET_URL", "").rstrip("/"))
    bitmagnet_timeout_s: float = field(default_factory=lambda: _float("BITMAGNET_TIMEOUT", 10))
    # Clips.
    max_gif_s: float = field(default_factory=lambda: _float("MAX_GIF_S", 15))
    # Disk quota for movies/ in GB; 0 = none. LRU eviction (design.md §10).
    max_data_gb: float = field(default_factory=lambda: _float("MAX_DATA_GB", 0))
    # Shared bearer token; empty = no auth (bind to localhost only).
    token: str = field(default_factory=lambda: _env("FH_TOKEN", ""))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "framehound.sqlite"

    def movie_dir(self, movie_id: str) -> Path:
        """The movie's index (searched on every query)."""
        return self.data_dir / "movies" / movie_id

    def media_movie_dir(self, movie_id: str) -> Path:
        """The movie's files: source, ``proxy.mp4``, partial torrents, clips."""
        return self.media_dir / "movies" / movie_id

    def movie_dirs(self, movie_id: str) -> list[Path]:
        """Both, once each when they're the same directory."""
        return list(dict.fromkeys([self.movie_dir(movie_id), self.media_movie_dir(movie_id)]))


_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Re-read the environment (tests)."""
    global _settings
    _settings = None
