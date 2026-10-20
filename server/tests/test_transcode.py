"""Proxy transcode: libx264 by default; NVENC first when asked, with a CPU fallback."""

import pytest
from framehound import config, media
from framehound.errors import Cancelled, FHError


@pytest.fixture
def calls(monkeypatch):
    """Records each run_ffmpeg as (codec, mem_mb); FAIL_NVENC raises for the GPU."""
    seen: list[tuple[str, int | None]] = []
    fail: list[Exception] = []

    def fake(args, duration_s, on_progress, check_cancel, cwd=None, mem_mb=None):
        codec = args[args.index("-c:v") + 1]
        seen.append((codec, mem_mb))
        if codec == "h264_nvenc" and fail:
            raise fail[0]
        on_progress(1.0)
        open(args[-1], "wb").close()

    monkeypatch.setattr(media, "run_ffmpeg", fake)
    return seen, fail


def transcode(tmp_path):
    dst = tmp_path / "proxy.mp4"
    media.transcode_proxy(tmp_path / "source.mkv", dst, 10.0, lambda f: None, lambda: None)
    return dst


def use_nvenc(monkeypatch):
    monkeypatch.setenv("TRANSCODE_HWACCEL", "nvenc")
    config.reset_settings()


def test_cpu_by_default(tmp_path, calls):
    assert transcode(tmp_path).exists()
    assert calls[0] == [("libx264", None)]


def test_nvenc(tmp_path, calls, monkeypatch):
    use_nvenc(monkeypatch)
    assert transcode(tmp_path).exists()
    assert calls[0] == [("h264_nvenc", 16384)]


def test_nvenc_falls_back_to_cpu(tmp_path, calls, monkeypatch):
    use_nvenc(monkeypatch)
    calls[1].append(FHError("ffmpeg_failed", "no", detail="No NVENC capable devices found"))
    assert transcode(tmp_path).exists()
    assert calls[0] == [("h264_nvenc", 16384), ("libx264", None)]


def test_cancel_is_not_retried(tmp_path, calls, monkeypatch):
    use_nvenc(monkeypatch)
    calls[1].append(Cancelled())
    with pytest.raises(Cancelled):
        transcode(tmp_path)
    assert calls[0] == [("h264_nvenc", 16384)]
