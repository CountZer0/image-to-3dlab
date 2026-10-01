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


def test_the_lock_holds_every_setting_and_the_exact_picture():
    lock = cp.lock_settings({"name": "hero", "faces": 20000, "seed": 7}, "abc")
    assert lock == {"faces": 20000, "seed": 7, "input_sha256": "abc"}


def test_resume_is_refused_without_a_lock(tmp_path):
    assert "missing" in cp.resume_problem(tmp_path / "run.json", {"faces": 20000})


def test_resume_names_every_setting_that_changed(tmp_path):
    lock = tmp_path / "run.json"
    lock.write_text(json.dumps({"faces": 20000, "seed": 7, "input_sha256": "a"}))
    assert cp.resume_problem(lock, {"faces": 20000, "seed": 7, "input_sha256": "a"}) is None
    problem = cp.resume_problem(lock, {"faces": 10000, "seed": 7, "input_sha256": "b"})
    assert "faces" in problem and "input_sha256" in problem and "seed" not in problem


def _stage_a_run(tmp_path, monkeypatch, **overrides):
    """A picture plus a run folder whose lock was written with default settings."""
    monkeypatch.setattr(cp, "preflight", lambda image: None)
    image = tmp_path / "hero.png"
    image.write_bytes(b"picture")
    root = tmp_path / "runs"
    paths = cp.run_paths(root.resolve(), "hero", image, cp.DEFAULT_FACES)
    paths["steps"].mkdir(parents=True)
    settings = {"name": "hero", "seed": cp.DEFAULT_SEED, "steps": None,
                "faces": cp.DEFAULT_FACES, "texture_size": cp.DEFAULT_TEXTURE,
                "pixel_match": True, **overrides}
    paths["lock"].write_text(json.dumps(cp.lock_settings(settings, cp.sha256_file(image))))
    return image, root, paths


def test_resume_at_a_new_face_count_is_refused_before_any_work(tmp_path, monkeypatch):
    image, root, _ = _stage_a_run(tmp_path, monkeypatch)
    monkeypatch.setattr(cp, "_run", lambda *a: pytest.fail("must not run a stage"))
    with pytest.raises(SystemExit, match="faces"):
        cp.main([str(image), "--out", str(root), "--resume", "--faces", "10000"])


def test_resume_with_an_edited_picture_is_refused(tmp_path, monkeypatch):
    image, root, _ = _stage_a_run(tmp_path, monkeypatch)
    image.write_bytes(b"edited picture")
    monkeypatch.setattr(cp, "_run", lambda *a: pytest.fail("must not run a stage"))
    with pytest.raises(SystemExit, match="input_sha256"):
        cp.main([str(image), "--out", str(root), "--resume"])


def test_resume_of_a_finished_run_changes_nothing(tmp_path, monkeypatch):
    image, root, paths = _stage_a_run(tmp_path, monkeypatch)
    paths["final"].write_bytes(b"glb")
    paths["provenance"].write_text("{}")
    monkeypatch.setattr(cp, "_run", lambda *a: pytest.fail("must not run a stage"))
    assert cp.main([str(image), "--out", str(root), "--resume"]) == 0
    assert paths["provenance"].read_text() == "{}"


def test_a_fresh_run_over_the_same_finished_run_points_to_resume(tmp_path, monkeypatch):
    image, root, paths = _stage_a_run(tmp_path, monkeypatch)
    paths["final"].write_bytes(b"glb")
    with pytest.raises(SystemExit, match="already holds this run.*--resume"):
        cp.main([str(image), "--out", str(root)])


def test_a_fresh_run_over_a_finished_run_at_other_settings_is_refused(tmp_path, monkeypatch):
    image, root, paths = _stage_a_run(tmp_path, monkeypatch, seed=7)
    paths["final"].write_bytes(b"glb")
    with pytest.raises(SystemExit, match="other settings"):
        cp.main([str(image), "--out", str(root)])


def test_printed_commands_survive_spaces_in_paths(tmp_path, capsys):
    folder = tmp_path / "my pics"
    folder.mkdir()
    image = folder / "Hero One.png"
    image.write_bytes(b"picture")
    cp.main([str(image), "--out", str(tmp_path / "runs"), "--dry-run"])
    assert "'" in capsys.readouterr().out


def test_resume_from_the_runs_own_copy_of_the_picture_does_not_crash(tmp_path, monkeypatch):
    image, root, paths = _stage_a_run(tmp_path, monkeypatch)
    paths["input"].parent.mkdir(parents=True)
    paths["input"].write_bytes(image.read_bytes())
    ran = []
    monkeypatch.setattr(cp, "_run", lambda command, label: ran.append(label))
    with pytest.raises(FileNotFoundError):  # stops at reading records the stub never wrote
        cp.main([str(paths["input"]), "--out", str(root), "--name", "hero", "--resume"])
    assert ran == ["generate", "finish"]


def test_a_fresh_run_at_other_settings_keeps_the_old_runs_steps(tmp_path, monkeypatch):
    image, root, _ = _stage_a_run(tmp_path, monkeypatch)
    monkeypatch.setattr(cp, "_run", lambda *a: pytest.fail("must not run a stage"))
    with pytest.raises(SystemExit, match="other settings"):
        cp.main([str(image), "--out", str(root), "--faces", "10000"])


def _ready_machine(tmp_path, monkeypatch, *, lite: bool):
    monkeypatch.setattr(cp, "readiness", lambda: {"ready": True})
    monkeypatch.setattr(cp, "find_blender", lambda: Path("/usr/bin/blender"))
    home = tmp_path / "u2net"
    home.mkdir()
    monkeypatch.setenv("U2NET_HOME", str(home))
    if lite:
        (home / f"{cp.LITE_MODEL}.onnx").write_bytes(b"onnx")


def _picture(tmp_path, *, matted: bool):
    from PIL import Image

    image = Image.new("RGBA", (8, 8), (200, 50, 50, 255))
    if matted:
        for x in range(8):
            image.putpixel((x, 0), (0, 0, 0, 0))
    path = tmp_path / ("matted.png" if matted else "plain.png")
    image.save(path)
    return path


def test_preflight_refuses_a_plain_picture_without_birefnet_lite(tmp_path, monkeypatch):
    _ready_machine(tmp_path, monkeypatch, lite=False)
    problem = cp.preflight(_picture(tmp_path, matted=False))
    assert "scripts/bootstrap_matte.py" in problem and "224 MB" in problem
    assert not list((tmp_path / "u2net").iterdir())


def test_preflight_lets_an_already_cut_out_picture_through_without_lite(tmp_path, monkeypatch):
    _ready_machine(tmp_path, monkeypatch, lite=False)
    assert cp.preflight(_picture(tmp_path, matted=True)) is None


def test_preflight_lets_a_plain_picture_through_once_lite_is_installed(tmp_path, monkeypatch):
    _ready_machine(tmp_path, monkeypatch, lite=True)
    assert cp.preflight(_picture(tmp_path, matted=False)) is None


def test_a_run_without_lite_stops_before_any_stage(tmp_path, monkeypatch):
    _ready_machine(tmp_path, monkeypatch, lite=False)
    monkeypatch.setattr(cp, "_run", lambda *a: pytest.fail("must not run a stage"))
    root = tmp_path / "runs"
    with pytest.raises(SystemExit, match="bootstrap_matte"):
        cp.main([str(_picture(tmp_path, matted=False)), "--out", str(root)])
    assert not root.exists()
