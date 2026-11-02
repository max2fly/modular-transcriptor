from __future__ import annotations

from pathlib import Path

from modular_transcriptor.runner import build_argparser


def test_runner_parses_boost_args():
    ns = build_argparser().parse_args(
        [
            "--stage",
            "parakeet",
            "--work",
            "/w",
            "--audio",
            "/a.wav",
            "--boost-file",
            "/w/boost.txt",
            "--boost-alpha",
            "4.0",
        ]
    )
    assert ns.boost_file == Path("/w/boost.txt") and ns.boost_alpha == 4.0


def test_runner_boost_args_default_none():
    ns = build_argparser().parse_args(
        ["--stage", "parakeet", "--work", "/w", "--audio", "/a.wav"]
    )
    assert ns.boost_file is None and ns.boost_alpha is None
