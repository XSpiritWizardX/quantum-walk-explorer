"""Simple smoke check for the web app and core modules."""

from __future__ import annotations

import importlib
import pathlib
import py_compile
import sys


TARGETS = [
    "webapp.py",
    "src/quantum_walk_explorer/cli.py",
    "src/quantum_walk_explorer/maze.py",
    "src/quantum_walk_explorer/visualize.py",
]

MODULES = [
    "quantum_walk_explorer",
    "quantum_walk_explorer.walk",
    "quantum_walk_explorer.maze",
    "quantum_walk_explorer.visualize",
]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    errors = []

    for target in TARGETS:
        path = root / target
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"Compile failed: {path} ({exc.msg})")

    for module in MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"Import failed: {module} ({exc})")

    if errors:
        print("Smoke check failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Smoke check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
