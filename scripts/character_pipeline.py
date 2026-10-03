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

**A quality check, because generators fail quietly.** A flat pixel-art bust once came back
as nine floating fragments with two bars through the head, and every stage reported
success. So the finished model is measured against its own picture: how well its outline,
seen from Pixal3D's camera, overlaps the cut-out (good models measure ~0.98, that one
0.44), and how much of its surface sits in the largest connected piece. Below the bar the
files are kept, the numbers go in the provenance record, and the run exits non-zero so a
batch can tell. Regenerate (another `--seed`, or a better picture) rather than repair.

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

import numpy as np
import trimesh
from mesh_health import weld_by_position
from pixal3d_generate import LICENSE_NAME, LICENSE_URL, has_alpha, readiness

from image_to_3dlab import photo_paint
from image_to_3dlab.blender import find_blender
from image_to_3dlab.matte import LITE_MODEL, matte_model
from image_to_3dlab.provenance import _pipeline_revision, sha256_file

DEFAULT_ROOT = REPO / "output" / "characters"
DEFAULT_FACES = 20000
DEFAULT_TEXTURE = 2048
DEFAULT_SEED = 42

# Quality bars, set between the two models measured so far: a good one at 0.976 / 100%,
# a shattered one at 0.442 / 70.5%. Directional, not precise: tighten with more runs.
MIN_SILHOUETTE_IOU = 0.85
MIN_LARGEST_PART = 0.90

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


def silhouette_iou(model: np.ndarray, matte: np.ndarray) -> float:
    """Overlap of two boolean masks: shared pixels over pixels in either. 0 when both empty."""
    union = np.logical_or(model, matte).sum()
    return float(np.logical_and(model, matte).sum() / union) if union else 0.0


def largest_part_share(mesh: trimesh.Trimesh) -> tuple[int, float]:
    """(connected pieces, largest piece's share of the surface area), welded by position.

    Welding is required: glTF splits vertices at every UV seam, and without it every UV
    chart would count as a separate piece (see `mesh_health.weld_by_position`).
    """
    parts = weld_by_position(mesh).split(only_watertight=False)
    areas = [part.area for part in parts]
    total = sum(areas)
    return len(parts), (max(areas) / total if total else 0.0)


def quality_verdict(iou: float | None, largest: float) -> list[str]:
    """What is wrong with the model, in words; empty means it passed."""
    problems = []
    if iou is not None and iou < MIN_SILHOUETTE_IOU:
        problems.append(f"its outline overlaps the picture by only {iou:.2f} "
                        f"(needs {MIN_SILHOUETTE_IOU})")
    if largest < MIN_LARGEST_PART:
        problems.append(f"its largest piece is only {largest:.0%} of the surface "
                        f"(needs {MIN_LARGEST_PART:.0%}): it is in fragments")
    return problems


def measure_quality(model: Path, views: Path | None) -> dict[str, Any]:
    """Measure a finished GLB against the picture Pixal3D saw. Seconds, numpy only."""
    positions, _, faces, _ = photo_paint.read_glb(model)
    pieces, largest = largest_part_share(trimesh.Trimesh(positions, faces, process=False))
    iou = None
    if views is not None:
        loaded, mesh_scale = photo_paint.load_views(views)
        view = loaded[0]
        x, y, depth = photo_paint.project(
            photo_paint.to_view_space(positions, mesh_scale=mesh_scale), view)
        face_of, _ = photo_paint.rasterize(np.stack([x, y], axis=1)[faces], depth[faces],
                                           view.image.shape[:2])
        drawn = face_of.reshape(view.image.shape[:2]) >= 0
        iou = round(silhouette_iou(drawn, view.image[..., 3] > 127), 3)
    problems = quality_verdict(iou, largest)
    return {"passed": not problems, "problems": problems, "silhouette_iou": iou,
            "pieces": pieces, "largest_part_share": round(largest, 3),
            "bars": {"silhouette_iou": MIN_SILHOUETTE_IOU,
                     "largest_part_share": MIN_LARGEST_PART}}


def quality_problem(quality: dict[str, Any] | None) -> str | None:
    """Why a finished model fails its quality check, or None when it passed."""
    if quality is None:
        return "no quality check is recorded for it"
    if quality.get("passed"):
        return None
    return "; ".join(quality.get("problems") or ["it did not pass"])


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


def preflight(image: Path) -> str | None:
    """Why the run cannot start, checked in seconds before a minutes-long generation."""
    state = readiness()
    if not state["ready"]:
        return ("Pixal3D is not installed. Run `python scripts/bootstrap_pixal3d.py` "
                f"(it states the 8.4 GB download and asks first). Detail: {state}")
    if not has_alpha(image) and matte_model() != LITE_MODEL:
        return ("BiRefNet-lite is not installed, and this picture needs its background cut "
                "out. Run `python scripts/bootstrap_matte.py` (224 MB, it asks first), or "
                "pass a picture that is already cut out.")
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

    problem = preflight(args.image)
    if problem:
        raise SystemExit(problem)

    wanted = lock_settings(settings, sha256_file(args.image))
    if args.resume:
        problem = resume_problem(paths["lock"], wanted)
        if problem:
            raise SystemExit(problem)
        if paths["final"].is_file() and paths["provenance"].is_file():
            problem = quality_problem(_read_json(paths["provenance"]).get("quality"))
            if problem:
                raise SystemExit(f"[character] already finished, but FAILED quality check: "
                                 f"{problem}. Files kept. Try another --seed, or a fuller, "
                                 "shaded picture.")
            print(f"[character] already finished: {paths['final']}", flush=True)
            return 0
    elif (paths["final"].exists() and paths["lock"].is_file()
          and not resume_problem(paths["lock"], wanted)):
        raise SystemExit(f"{paths['dir']} already holds this run. Add --resume to finish "
                         "or reuse it, or delete that folder to redo it.")
    elif paths["final"].exists() or (paths["lock"].is_file()
                                     and resume_problem(paths["lock"], wanted)):
        # A fresh run at other settings would overwrite steps/ that another finished
        # model in this folder was made from.
        raise SystemExit(f"{paths['dir']} already holds a run at other settings. Use "
                         "--name to start a new run, or delete that folder to redo it.")
    paths["steps"].mkdir(parents=True, exist_ok=True)
    paths["input"].parent.mkdir(parents=True, exist_ok=True)
    # Skipped when the stored copy is already this picture: resuming from it would
    # otherwise copy a file onto itself, which shutil refuses.
    if not (paths["input"].is_file()
            and sha256_file(paths["input"]) == wanted["input_sha256"]):
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
    try:
        quality = measure_quality(paths["final"], usable_views(paths["views"]))
    except (ValueError, KeyError, IndexError, OSError) as error:
        quality = {"passed": False, "error": f"{type(error).__name__}: {error}",
                   "problems": [f"the model could not be measured ({error})"]}
    record["quality"] = quality
    paths["provenance"].write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    size = paths["final"].stat().st_size / 1048576
    if "error" not in quality:
        iou = quality["silhouette_iou"]
        print(f"[character] quality: outline match {'n/a' if iou is None else iou}, "
              f"{quality['pieces']} piece(s), largest {quality['largest_part_share']:.0%}",
              flush=True)
    print(f"[character] done -> {paths['final']} ({size:.1f} MB); "
          f"provenance {paths['provenance'].name}", flush=True)
    problem = quality_problem(quality)
    if problem:
        raise SystemExit("[character] FAILED quality check: " + problem
                         + ". Files kept. Try another --seed, or a fuller, shaded picture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
