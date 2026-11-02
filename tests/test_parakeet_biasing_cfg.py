from pathlib import Path

from modular_transcriptor.parakeet_transcriber import (
    build_decoding_cfg,
    parakeet_cache_extra,
)


def test_build_decoding_cfg_none_when_disabled():
    assert build_decoding_cfg(None, 4.0) is None
    assert build_decoding_cfg(Path("x.txt"), 0.0) is None


def test_build_decoding_cfg_sets_boosting_fields(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("Shondo\n")
    cfg = build_decoding_cfg(f, 4.0)
    assert cfg["strategy"] == "greedy_batch"
    assert cfg["greedy"]["boosting_tree"]["key_phrases_file"] == str(f)
    # gentleness defaults (full-stream knob sweep): gentle per-token push,
    # strong depth reward so the boost breaks ties for phrases, not single tokens
    assert cfg["greedy"]["boosting_tree"]["context_score"] == 0.3
    assert cfg["greedy"]["boosting_tree"]["depth_scaling"] == 4.0
    assert cfg["greedy"]["boosting_tree_alpha"] == 4.0
    # explicit overrides are honored
    cfg2 = build_decoding_cfg(f, 2.0, context_score=1.0, depth_scaling=2.0)
    assert cfg2["greedy"]["boosting_tree"]["context_score"] == 1.0
    assert cfg2["greedy"]["boosting_tree"]["depth_scaling"] == 2.0


def test_cache_extra_alpha_inert_without_boost_file():
    assert parakeet_cache_extra(None, 4.0) == parakeet_cache_extra(None, 0.0)
