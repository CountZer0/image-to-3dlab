"""Tests for the one-command character pipeline.

Generation costs minutes, so what is worth testing here is what would go wrong silently
after those minutes: the paths a resume looks for, the flags handed to Pixal3D and Finish
(the repaint must stay off; Pixel Match needs the camera Pixal3D left), and the licence
the finished model carries.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "character_pipeline.py"


def _load():
    spec = importlib.util.spec_from_file_location("character_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cp = _load()


@pytest.mark.parametrize("text, expected", [
    ("meebit_03823_full", "meebit_03823_full"),
    ("Meebit #3823 (full)", "meebit_3823_full"),
    ("  ", "character"),
])
def test_names_are_safe_for_folders_and_files(text, expected):
    assert cp.slug(text) == expected


@pytest.mark.parametrize("faces, label", [(20000, "20k"), (1500, "1.5k"), (800, "800")])
def test_face_labels_match_finish_naming(faces, label):
    assert cp.faces_label(faces) == label


def test_run_paths_keep_one_character_in_one_folder(tmp_path):
    paths = cp.run_paths(tmp_path, "hero", Path("/pics/hero.png"), 20000)
    assert paths["dir"] == tmp_path / "hero"
    assert paths["input"] == tmp_path / "hero" / "input" / "hero.png"
    assert paths["final"] == tmp_path / "hero" / "hero_20k.glb"
    assert paths["provenance"] == tmp_path / "hero" / "hero_20k.provenance.json"
    # Where pixal3d_generate.py and retopo_repaint.py actually write their records.
    assert paths["generated_record"] == tmp_path / "hero" / "steps" / "0_generated.json"
    assert paths["finish_record"] == tmp_path / "hero" / "hero_20k.retopo-repaint.json"
    assert paths["views"] == tmp_path / "hero" / "steps" / "0_generated.svviews"


def test_generate_passes_steps_only_when_asked():
    base = cp.generate_command("py", Path("in.png"), Path("out.glb"), 7, None)
    assert base[1].endswith("pixal3d_generate.py")
    assert base[2:] == ["in.png", "out.glb", "--seed", "7"]
    assert cp.generate_command("py", Path("in.png"), Path("out.glb"), 7, 12)[-2:] == [
        "--steps", "12"]


def test_finish_never_repaints_and_uses_the_camera_when_there_is_one(tmp_path):
    command = cp.finish_command(
        "py", Path("raw.glb"), Path("in.png"), Path("hero_20k.glb"), faces=20000,
        texture_size=2048, steps_dir=tmp_path, views=Path("raw.svviews"), resume=False)
    assert command[1].endswith("retopo_repaint.py")
    assert command[2:5] == ["raw.glb", "in.png", "hero_20k.glb"]
    assert "--skip-paint" in command
    assert command[command.index("--faces") + 1] == "20000"
    assert command[command.index("--views") + 1] == "raw.svviews"
    assert "--resume" not in command


def test_finish_without_a_camera_skips_pixel_match_and_can_resume(tmp_path):
    command = cp.finish_command(
        "py", Path("raw.glb"), Path("in.png"), Path("out.glb"), faces=20000,
        texture_size=2048, steps_dir=tmp_path, views=None, resume=True)
    assert "--views" not in command
    assert command[-1] == "--resume"


def test_a_views_folder_counts_only_with_its_transforms(tmp_path):
    views = tmp_path / "raw.svviews"
    views.mkdir()
    assert cp.usable_views(views) is None
    (views / "transforms.json").write_text("{}")
    assert cp.usable_views(views) == views


def test_provenance_carries_the_dinov3_licence_and_both_stage_records(tmp_path):
    image = tmp_path / "hero.png"
    image.write_bytes(b"picture")
    final = tmp_path / "hero_20k.glb"
    final.write_bytes(b"glb")
    generated = {"backend": "pixal3d", "components": [{"component": "rembg/x"}]}
    finished = {"stages": ["retopologise", "photo", "bake", "compress"]}
    record = cp.provenance_record(image, final, {"faces": 20000}, generated, finished)

    assert record["output"]["classification"] == "commercial-conditional"
    assert "DINOv3" in record["license"]["name"]
    assert [c["component"] for c in record["components"]][1] == "rembg/x"
    assert record["components"][0]["license"] == "DINOv3 License"
    assert record["stages"] == {"generate": generated, "finish": finished}
    assert len(record["input"]["sha256"]) == 64
    json.dumps(record)  # must be writable as-is


def test_dry_run_prints_both_commands_and_touches_nothing(tmp_path, capsys):
    image = tmp_path / "hero.png"
    image.write_bytes(b"picture")
    assert cp.main([str(image), "--out", str(tmp_path / "runs"), "--dry-run"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert "pixal3d_generate.py" in lines[0]
    assert "retopo_repaint.py" in lines[1] and "--skip-paint" in lines[1]
    assert not (tmp_path / "runs").exists()


def test_a_tiny_face_count_is_refused_before_any_work(tmp_path):
    image = tmp_path / "hero.png"
    image.write_bytes(b"picture")
    with pytest.raises(SystemExit, match="faces"):
        cp.main([str(image), "--faces", "500", "--dry-run"])
