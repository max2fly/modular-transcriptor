from __future__ import annotations

import sys
import types
from contextlib import contextmanager


@contextmanager
def fake_modules(**modules):
    """Temporarily install fake modules into sys.modules so production
    modules with lazy heavy imports (nemo/torch/transformers/whisperx)
    are importable without the real packages."""
    saved = {name: sys.modules.get(name) for name in modules}
    try:
        for name, mod in modules.items():
            sys.modules[name] = mod
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def make_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m
