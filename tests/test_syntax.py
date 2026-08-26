"""
Every module parses, and the CPU-only ones import. Runs anywhere, in a second.

This exists because a broken module reached a commit three times in one
session, each time from the same cause: multi-line Python written through a
shell heredoc, where an escaped \\n inside a string became a real newline and
split an f-string across two lines. It parses fine in the patch script and
dies on the VM after a `git pull` -- which is the worst possible place to find
it, because the GPU stages cannot be run here at all.

Import is checked separately from parsing, and only for modules that have no
GPU dependency: `stage1_edit` and `stage3_judge` are parsed but not imported,
since importing them on the laptop is meaningless (their torch/vllm imports are
deliberately deferred into functions, so a bad import would not show up here
anyway).

Run:  ./.venv/Scripts/python.exe tests/test_syntax.py
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Imported as well as parsed: no torch, no vllm, no diffusers.
CPU_MODULES = [
    "src.schema",
    "src.corruptions",
    "src.judge_prompt",
    "src.build_manifest",
    "src.stage0_coco",
    "src.stage2_corrupt",
    "src.stage4_analyze",
]


def main() -> int:
    ok = True

    files = sorted(
        p for d in ("src", "scripts", "tests")
        for p in (ROOT / d).glob("*.py")
    )
    print(f"=== parsing {len(files)} files ===")
    for p in files:
        try:
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except SyntaxError as e:
            ok = False
            print(f"  FAIL {p.relative_to(ROOT)}:{e.lineno}: {e.msg}")
    if ok:
        print("  PASS all files parse")

    print(f"\n=== importing {len(CPU_MODULES)} CPU-only modules ===")
    for name in CPU_MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:
            ok = False
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    if ok:
        print("  PASS all CPU modules import")

    # The GPU modules must stay importable without torch/vllm present, because
    # --dry-run and --preflight are the whole point of being able to check them
    # from the laptop. A module-level `import torch` would break that silently.
    print("\n=== GPU modules import without torch/vllm ===")
    for name in ("src.stage1_edit", "src.stage3_judge"):
        try:
            importlib.import_module(name)
            print(f"  PASS {name}")
        except Exception as e:
            ok = False
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    for banned in ("torch", "vllm", "diffusers"):
        if banned in sys.modules:
            ok = False
            print(f"  FAIL importing them pulled in {banned!r}")

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
