"""Runtime dependency helpers for optional RAG integrations."""

from __future__ import annotations

import importlib


INSTALL_HINT = "Install the RAG stack with `pip install -r requirements.txt`."


def require_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing optional dependency `{module_name}`. {INSTALL_HINT}"
        ) from exc

