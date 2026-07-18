from __future__ import annotations

import os

import modular_transcriptor.cache as cache


def _mk(p, data=b"data"):
    p.write_bytes(data)
    return p


def test_stamp_then_fresh_when_unchanged(tmp_path):
    src = _mk(tmp_path / "in.wav")
    out = _mk(tmp_path / "out.json", b"{}")
    cache.stamp(out, src)
    assert cache.is_fresh(out, src) is True


def test_not_fresh_when_no_sidecar(tmp_path):
    src = _mk(tmp_path / "in.wav")
    out = _mk(tmp_path / "out.json", b"{}")
    assert cache.is_fresh(out, src) is False  # legacy output, never stamped


def test_not_fresh_when_output_missing(tmp_path):
    src = _mk(tmp_path / "in.wav")
    out = tmp_path / "out.json"  # does not exist
    cache.stamp(out, src)  # sidecar exists but output doesn't
    assert cache.is_fresh(out, src) is False


def test_not_fresh_on_size_change(tmp_path):
    src = _mk(tmp_path / "in.wav", b"short")
    out = _mk(tmp_path / "out.json", b"{}")
    cache.stamp(out, src)
    _mk(src, b"much-longer-content")  # size changed
    assert cache.is_fresh(out, src) is False


def test_not_fresh_on_same_length_different_content(tmp_path):
    # Same byte length, different content (a re-cut seam): only mtime reveals it.
    src = _mk(tmp_path / "in.wav", b"AAAAAAAA")
    out = _mk(tmp_path / "out.json", b"{}")
    cache.stamp(out, src)
    _mk(src, b"BBBBBBBB")
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert cache.is_fresh(out, src) is False


def test_fingerprint_none_for_missing_source(tmp_path):
    assert cache.fingerprint(tmp_path / "nope.wav") is None


def test_clear_removes_sidecar(tmp_path):
    src = _mk(tmp_path / "in.wav")
    out = _mk(tmp_path / "out.json", b"{}")
    cache.stamp(out, src)
    assert cache.sidecar_path(out).exists()
    cache.clear(out)
    assert not cache.sidecar_path(out).exists()
    cache.clear(out)  # idempotent, no raise


def test_read_sidecar_corrupt_returns_none(tmp_path):
    out = _mk(tmp_path / "out.json", b"{}")
    cache.sidecar_path(out).write_text("not json{")
    assert cache.read_sidecar(out) is None
