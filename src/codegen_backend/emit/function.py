from __future__ import annotations

from collections.abc import Sequence


def emit_function_header(
    name: str, args: Sequence[str], qualifiers: str | None = None
) -> str:
    qualifier_prefix = f"{qualifiers} " if qualifiers else ""
    return f"{qualifier_prefix}{name}({', '.join(args)}) {{"
