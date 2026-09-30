#!/usr/bin/env python3
"""Stop the NVIDIA TRELLIS.2 checkout loading BRIA RMBG-2.0, before it ever downloads it.

Upstream `microsoft/TRELLIS.2` builds its background remover inside `from_pretrained`, and
the TRELLIS.2-4B config names `briaai/RMBG-2.0`. That model's licence is not one this repo's
generation pipeline may use (see AGENTS.md), so the constructor call is replaced with
`None` in both pipelines that have it. Images are cut out by our own remover
(`image_to_3dlab/matte.py`) before they reach TRELLIS.

`scripts/trellis_cuda_generate.py` refuses to run on a checkout this has not patched, so
this is a guardrail, not a suggestion. Safe to run twice.

    python scripts/patch_trellis_cuda_no_bria.py                 # vendor/trellis-cuda
    python scripts/patch_trellis_cuda_no_bria.py --root <TRELLIS.2 checkout>
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "trellis-cuda" / "TRELLIS.2"

# The two pipelines whose `from_pretrained` constructs the remover.
PIPELINES = ("trellis2_image_to_3d.py", "trellis2_texturing.py")

NEEDLE = (
    "        pipeline.rembg_model = getattr(rembg, "
    "args['rembg_model']['name'])(**args['rembg_model']['args'])"
)
MARKER = "# image-to-3dlab: BRIA RMBG-2.0 is never loaded"
REPLACEMENT = (
    f"        {MARKER} (licence guardrail).\n"
    "        # Inputs are cut out by image_to_3dlab/matte.py before they get here.\n"
    "        pipeline.rembg_model = None"
)


def pipeline_files(root: Path) -> list[Path]:
    return [root / "trellis2" / "pipelines" / name for name in PIPELINES]


def patch_source(source: str, label: str = "source") -> tuple[str, bool]:
    """Return the patched text and whether it changed. Raises if the anchor is missing."""
    if NEEDLE in source:
        return source.replace(NEEDLE, REPLACEMENT), True
    if REPLACEMENT in source:
        return source, False
    raise RuntimeError(f"TRELLIS background-model hook not found in {label}; "
                       "upstream changed, re-check the patch before running")


def is_patched(root: Path) -> bool:
    """True when every pipeline file exists, carries the patch and has no live BRIA load."""
    for path in pipeline_files(root):
        try:
            text = path.read_text()
        except OSError:
            return False
        if NEEDLE in text or REPLACEMENT not in text:
            return False
    return True


def apply(root: Path) -> list[Path]:
    """Patch every pipeline file under a TRELLIS.2 checkout. Returns the files changed."""
    changed = []
    for path in pipeline_files(root):
        if not path.is_file():
            raise RuntimeError(f"not a TRELLIS.2 checkout: missing {path}")
        text, did = patch_source(path.read_text(), str(path))
        if did:
            path.write_text(text)
            changed.append(path)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="the TRELLIS.2 checkout (default: %(default)s)")
    args = parser.parse_args(argv)
    changed = apply(args.root.resolve())
    print(f"patched {len(changed)} file(s)" if changed else "already patched",
          "- BRIA RMBG-2.0 will not load.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
