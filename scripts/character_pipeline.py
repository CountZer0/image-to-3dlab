#!/usr/bin/env python3
"""Turn one picture of a character into a finished, game-ready GLB, in one command.

    python scripts/character_pipeline.py hero.png
    python scripts/character_pipeline.py hero.png --name hero --faces 20000 --seed 7

**Why this exists.** The route already works, but it is three tools run by hand: cut the
background out, generate with Pixal3D, then Finish (retopologise, Pixel Match, compress).
Each has its own flags and its own record, and the Pixel Match camera has to be found and
passed along. This chains them, keeps every in-between file, and writes one provenance
record for the finished model.

**One folder per character**, under `output/characters/<name>/` by default:

    input/<picture>                the picture exactly as given
    steps/0_generated.glb (.json)  Pixal3D's raw model and its run record
    steps/1_retopo.glb ...         Finish's in-between models
    <name>_20k.glb                 the finished model
    <name>_20k.provenance.json     how it was made, under which licences

No rig: the finished model is static on purpose, ready for whatever rigging route comes
next. `--resume` reuses any stage already on disk, so a run that died in Finish does not
pay for generation twice. `--dry-run` prints the commands and stops.

Licence: Pixal3D's code and flow weights are MIT, but it bundles the DINOv3 image encoder
under its own licence, so the result is classed `commercial-conditional`, as TRELLIS's is.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))

from pixal3d_generate import LICENSE_NAME, LICENSE_URL, readiness

from image_to_3dlab.blender import find_blender
from image_to_3dlab.provenance import _pipeline_revision, sha256_file

DEFAULT_ROOT = REPO / "output" / "characters"
DEFAULT_FACES = 20000
DEFAULT_TEXTURE = 2048
DEFAULT_SEED = 42

CLASSIFICATION = "commercial-conditional"
LICENSE = {
    "name": LICENSE_NAME,
    "url": LICENSE_URL,
    "conditions": [
        "Pixal3D code and flow weights are MIT licensed.",
        "The bundled DINOv3 image encoder is governed by the separate DINOv3 License.",
        "The input picture's own rights still apply to the model made from it.",
    ],
}
DINOV3 = {
    "component": "facebook/dinov3 (bundled in raven38/pixal3d-sv-q8_0-v1)",
    "purpose": "image conditioning",
    "license": "DINOv3 License",
    "commercial_status": "commercial-conditional",
    "license_url": "https://ai.meta.com/resources/models-and-libraries/dinov3-license/",
}


def slug(text: str) -> str:
    """A folder- and file-safe name: `Meebit #3823 (full)` -> `meebit_3823_full`."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return cleaned or "character"


def faces_label(faces: int) -> str:
    """`20000` -> `20k`, `1500` -> `1.5k`, `800` -> `800`; matches Finish's own naming."""
    return str(faces) if faces < 1000 else f"{faces / 1000:g}k"


def run_paths(root: Path, name: str, image: Path, faces: int) -> dict[str, Path]:
    """Every path a run reads or writes. Fixed names, so `--resume` knows what to look for."""
    directory = root / name
    steps = directory / "steps"
    generated = steps / "0_generated.glb"
    final = directory / f"{name}_{faces_label(faces)}.glb"
    return {
        "dir": directory,
        "input": directory / "input" / image.name,
        "steps": steps,
        "generated": generated,
        # pixal3d_generate.py writes its record as `<output stem>.json`, and trellis-cli
        # stages the camera it used as `<output>.svviews/`, which Pixel Match needs.
        "generated_record": generated.with_suffix(".json"),
        "views": generated.with_suffix(".svviews"),
        "final": final,
        "finish_record": final.with_name(f"{final.stem}.retopo-repaint.json"),
        "provenance": final.with_suffix(".provenance.json"),
        # The settings the in-between files were made with. Finish's step files carry no
        # face count, so without this a resume at new settings would reuse old ones.
        "lock": steps / "run.json",
    }


def lock_settings(settings: dict[str, Any], image_sha256: str) -> dict[str, Any]:
    """What every in-between file depends on: the settings and the exact input picture."""
    return {**{k: v for k, v in settings.items() if k != "name"}, "input_sha256": image_sha256}


def resume_problem(lock: Path, wanted: dict[str, Any]) -> str | None:
    """Why `--resume` would mix old and new settings, or None when it is safe."""
    if not lock.is_file():
        return (f"{lock} is missing, so the files there cannot be matched to these "
                "settings. Run without --resume to start over.")
    stored = json.loads(lock.read_text(encoding="utf-8"))
    changed = sorted(k for k in set(stored) | set(wanted) if stored.get(k) != wanted.get(k))
    if changed:
        return (f"--resume refused: {', '.join(changed)} changed since this run started. "
                "Run without --resume, or use --name for a separate run.")
    return None


def generate_command(python: str, image: Path, output: Path, seed: int,
                     steps: int | None) -> list[str]:
    command = [python, str(SCRIPTS / "pixal3d_generate.py"), str(image), str(output),
               "--seed", str(seed)]
    if steps is not None:
        command += ["--steps", str(steps)]
    return command


def finish_command(python: str, generated: Path, image: Path, output: Path, *,
                   faces: int, texture_size: int, steps_dir: Path, views: Path | None,
                   resume: bool) -> list[str]:
    """Finish without the repaint: it needs Apple Silicon, and Pixal3D arrives painted."""
    command = [python, str(SCRIPTS / "retopo_repaint.py"), str(generated), str(image),
               str(output), "--faces", str(faces), "--texture-size", str(texture_size),
               "--skip-paint", "--steps-dir", str(steps_dir)]
    if views is not None:
        command += ["--views", str(views)]
    if resume:
        command.append("--resume")
    return command


def usable_views(views: Path) -> Path | None:
    """The Pixel Match camera folder, or None when the generator did not leave one."""
    return views if (views / "transforms.json").is_file() else None


def provenance_record(image: Path, final: Path, settings: dict[str, Any],
                      generated: dict[str, Any], finished: dict[str, Any],
                      revision: dict[str, Any] | None = None) -> dict[str, Any]:
    """One record for the finished model: input, output, licence and both stages' records.

    The licence is the generator's: Finish only reshapes and re-encodes, and Pixel Match
    copies the input picture's own pixels, so neither adds a licence of its own.
    """
    components = [DINOV3] + list(generated.get("components") or [])
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "character",
        "input": {"path": str(image), "sha256": sha256_file(image),
                  "source": "user-provided"},
        "output": {"path": str(final), "sha256": sha256_file(final),
                   "classification": CLASSIFICATION},
        "license": LICENSE,
        "components": components,
        "settings": settings,
        "stages": {"generate": generated, "finish": finished},
        "software": {"pipeline_revision": revision},
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str], label: str) -> None:
    print(f"[character] {label}: {shlex.join(command)}", flush=True)
    code = subprocess.run(command, cwd=str(REPO), check=False).returncode
    if code != 0:
        raise SystemExit(f"[character] {label} failed with exit code {code}")


def preflight() -> str | None:
    """Why the run cannot start, checked in seconds before a minutes-long generation."""
    state = readiness()
    if not state["ready"]:
        return ("Pixal3D is not installed. Run `python scripts/bootstrap_pixal3d.py` "
                f"(it states the 8.4 GB download and asks first). Detail: {state}")
    if find_blender() is None:
        return ("Blender was not found, and Finish needs it. Install Blender 4.2+ or set "
                "I2L_BLENDER to its executable.")
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="picture of the character")
    parser.add_argument("--name", default=None,
                        help="run and file name (default: from the picture's file name)")
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT,
                        help=f"parent folder for runs (default: {DEFAULT_ROOT})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=None,
                        help="Pixal3D sampling steps (default: its own choice, 8 or 12)")
    parser.add_argument("--faces", type=int, default=DEFAULT_FACES,
                        help="finished face count")
    parser.add_argument("--texture-size", type=int, default=DEFAULT_TEXTURE,
                        help="finished texture size in pixels")
    parser.add_argument("--no-pixel-match", action="store_true",
                        help="keep the generator's paint on the front too")
    parser.add_argument("--resume", action="store_true",
                        help="reuse stages already on disk; only safe with the same settings")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands and stop")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.image.is_file():
        raise SystemExit(f"not found: {args.image}")
    if args.faces < 1000:
        raise SystemExit("--faces below 1000 shreds a character; Finish refuses it too")

    name = slug(args.name or args.image.stem)
    paths = run_paths(args.out.resolve(), name, args.image, args.faces)
    python = sys.executable
    settings = {"name": name, "seed": args.seed, "steps": args.steps,
                "faces": args.faces, "texture_size": args.texture_size,
                "pixel_match": not args.no_pixel_match}

    if args.dry_run:
        views = None if args.no_pixel_match else usable_views(paths["views"])
        print(shlex.join(generate_command(python, paths["input"], paths["generated"],
                                          args.seed, args.steps)))
        print(shlex.join(finish_command(
            python, paths["generated"], paths["input"], paths["final"],
            faces=args.faces, texture_size=args.texture_size, steps_dir=paths["steps"],
            views=views, resume=args.resume)))
        return 0

    problem = preflight()
    if problem:
        raise SystemExit(problem)

    wanted = lock_settings(settings, sha256_file(args.image))
    if args.resume:
        problem = resume_problem(paths["lock"], wanted)
        if problem:
            raise SystemExit(problem)
        if paths["final"].is_file() and paths["provenance"].is_file():
            print(f"[character] already finished: {paths['final']}", flush=True)
            return 0
    elif paths["final"].exists():
        raise SystemExit(f"{paths['dir']} already has a finished model. Use --name to "
                         "start a new run, or delete that folder to redo it.")
    paths["steps"].mkdir(parents=True, exist_ok=True)
    paths["input"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.image, paths["input"])
    paths["lock"].write_text(json.dumps(wanted, indent=2, sort_keys=True) + "\n")

    if args.resume and paths["generated"].is_file() and paths["generated_record"].is_file():
        print(f"[character] generate: reusing {paths['generated'].name}", flush=True)
    else:
        _run(generate_command(python, paths["input"], paths["generated"], args.seed,
                              args.steps), "generate")

    views = None if args.no_pixel_match else usable_views(paths["views"])
    if not args.no_pixel_match and views is None:
        print("[character] note: no Pixal3D camera found, so no Pixel Match", flush=True)
    settings["pixel_match"] = views is not None
    _run(finish_command(python, paths["generated"], paths["input"], paths["final"],
                        faces=args.faces, texture_size=args.texture_size,
                        steps_dir=paths["steps"], views=views, resume=args.resume),
         "finish")

    record = provenance_record(
        paths["input"], paths["final"], settings,
        _read_json(paths["generated_record"]), _read_json(paths["finish_record"]),
        _pipeline_revision(),
    )
    paths["provenance"].write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    size = paths["final"].stat().st_size / 1048576
    print(f"[character] done -> {paths['final']} ({size:.1f} MB); "
          f"provenance {paths['provenance'].name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
