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

import numpy as np
import pytest
import trimesh

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
    stored = json.dumps({"quality": {"passed": True, "problems": []}})
    paths["provenance"].write_text(stored)
    monkeypatch.setattr(cp, "_run", lambda *a: pytest.fail("must not run a stage"))
    assert cp.main([str(image), "--out", str(root), "--resume"]) == 0
    assert paths["provenance"].read_text() == stored


@pytest.mark.parametrize("record, reason", [
    ({"quality": {"passed": False, "problems": ["it is in fragments"]}}, "it is in fragments"),
    ({}, "no quality check"),
])
def test_resume_of_a_finished_run_that_failed_its_check_still_fails(
        tmp_path, monkeypatch, record, reason):
    image, root, paths = _stage_a_run(tmp_path, monkeypatch)
    paths["final"].write_bytes(b"glb")
    paths["provenance"].write_text(json.dumps(record))
    monkeypatch.setattr(cp, "_run", lambda *a: pytest.fail("must not run a stage"))
    with pytest.raises(SystemExit, match=f"FAILED quality check: {reason}"):
        cp.main([str(image), "--out", str(root), "--resume"])
    assert paths["final"].read_bytes() == b"glb"


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


def test_silhouette_iou_is_overlap_over_union():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2] = True          # 8 pixels
    b[1:3] = True         # 8 pixels, 4 shared
    assert cp.silhouette_iou(a, b) == pytest.approx(4 / 12)
    assert cp.silhouette_iou(a, a) == 1.0
    assert cp.silhouette_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 0.0


def test_one_body_counts_as_one_piece_even_when_split_at_uv_seams():
    box = trimesh.creation.box()
    # glTF-style: every face gets its own vertices, as a UV split would do.
    split = trimesh.Trimesh(box.vertices[box.faces].reshape(-1, 3),
                            np.arange(len(box.faces) * 3).reshape(-1, 3), process=False)
    assert cp.largest_part_share(split) == (1, pytest.approx(1.0))


def test_fragments_are_counted_by_share_of_surface():
    big = trimesh.creation.box(extents=(3, 3, 3))
    small = trimesh.creation.box(extents=(1, 1, 1))
    small.apply_translation((10, 0, 0))
    pieces, largest = cp.largest_part_share(trimesh.util.concatenate([big, small]))
    assert pieces == 2
    assert largest == pytest.approx(54 / 60)


@pytest.mark.parametrize("iou, largest, failed", [
    (0.976, 1.0, []),                         # the Meebit that worked
    (0.442, 0.705, ["outline", "fragments"]),  # the pixel-art bust that shattered
    (None, 0.95, []),                         # no camera: only the fragment check runs
    (0.90, 0.80, ["fragments"]),
])
def test_quality_verdict_names_each_problem(iou, largest, failed):
    problems = cp.quality_verdict(iou, largest)
    assert len(problems) == len(failed)
    for word, problem in zip(failed, problems):
        assert word in problem


# --- the quality check end to end ---------------------------------------------------------

# A camera far away with a narrow lens, so a box seen face-on projects to a rectangle whose
# size follows from the pinhole formula alone. Pixal3D's frame maps GLB (x, y, z) to view
# (-x, z, y): GLB y is depth, GLB z is up.
SIZE, FOCAL, DISTANCE, MESH_SCALE = 64, 1600.0, 50.0, 2.0


def _box_glb(path, extents, offset=(0.0, 0.0, 0.0), extra=None):
    from PIL import Image

    box = trimesh.creation.box(extents=extents)
    box.apply_translation(offset)
    if extra is not None:
        box = trimesh.util.concatenate([box, extra])
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", (4, 4), (128, 128, 128)))
    uv = (box.vertices[:, :2] - box.vertices[:, :2].min(0)) / np.ptp(box.vertices[:, :2], 0)
    box.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    path.write_bytes(box.export(file_type="glb"))
    return path


def _views_of(folder, extents):
    """A .svviews folder whose matte is the box's front face, drawn without photo_paint."""
    from PIL import Image

    width, depth, height = (e / MESH_SCALE for e in extents)
    near = DISTANCE - depth / 2
    half_w, half_h = FOCAL * width / 2 / near, FOCAL * height / 2 / near
    centre = np.arange(SIZE) + 0.5 - SIZE / 2
    inside = (np.abs(centre)[None, :] < half_w) & (np.abs(centre)[:, None] < half_h)
    rgba = np.zeros((SIZE, SIZE, 4), np.uint8)
    rgba[inside] = (200, 50, 50, 255)
    folder.mkdir()
    Image.fromarray(rgba).save(folder / "input.png")
    camera = np.eye(4)
    camera[2, 3] = DISTANCE
    (folder / "transforms.json").write_text(json.dumps({
        "camera_angle_x": 2 * np.arctan(SIZE / 2 / FOCAL), "mesh_scale": MESH_SCALE,
        "frames": [{"file_path": "input.png", "transform_matrix": camera.tolist()}],
    }))
    return folder, inside


# Tall and narrow, and thin in depth, so a swapped axis or a lost scale changes the outline.
EXTENTS = (1.0, 0.4, 1.6)


def test_a_model_that_matches_its_picture_passes(tmp_path):
    views, inside = _views_of(tmp_path / "raw.svviews", EXTENTS)
    assert 200 < inside.sum() < SIZE * SIZE / 2  # a real outline, not empty or full
    quality = cp.measure_quality(_box_glb(tmp_path / "box.glb", EXTENTS), views)
    assert quality["silhouette_iou"] >= 0.95
    assert quality["pieces"] == 1 and quality["largest_part_share"] == 1.0
    assert quality["passed"] and quality["problems"] == []


@pytest.mark.parametrize("extents, offset", [
    (EXTENTS, (0.5, 0.0, 0.0)),            # moved sideways
    ((2.0, 0.8, 3.2), (0.0, 0.0, 0.0)),    # twice the size
    ((1.6, 0.4, 1.0), (0.0, 0.0, 0.0)),    # lying on its side
])
def test_a_model_that_misses_its_picture_fails_on_outline(tmp_path, extents, offset):
    views, _ = _views_of(tmp_path / "raw.svviews", EXTENTS)
    quality = cp.measure_quality(_box_glb(tmp_path / "box.glb", extents, offset), views)
    assert quality["silhouette_iou"] < cp.MIN_SILHOUETTE_IOU
    assert not quality["passed"] and "outline" in quality["problems"][0]


def test_without_a_camera_only_the_fragment_check_runs(tmp_path):
    quality = cp.measure_quality(_box_glb(tmp_path / "box.glb", EXTENTS), None)
    assert quality["silhouette_iou"] is None and quality["passed"]


def _fragments():
    small = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    small.apply_translation((5.0, 0.0, 0.0))
    return small


def _stub_stages(monkeypatch, paths, write_final):
    def run(command, label):
        if label == "generate":
            paths["generated_record"].write_text("{}")
        else:
            write_final(paths["final"])
            paths["finish_record"].write_text("{}")
    monkeypatch.setattr(cp, "_run", run)


def test_a_failed_check_keeps_the_files_records_the_numbers_and_exits_non_zero(
        tmp_path, monkeypatch):
    image, root, paths = _stage_a_run(tmp_path, monkeypatch)
    _stub_stages(monkeypatch, paths,
                 lambda final: _box_glb(final, (1, 1, 1), extra=_fragments()))
    with pytest.raises(SystemExit, match="FAILED quality check.*fragments"):
        cp.main([str(image), "--out", str(root)])
    assert paths["final"].is_file()
    quality = json.loads(paths["provenance"].read_text())["quality"]
    assert quality["passed"] is False and quality["pieces"] == 2
    assert quality["largest_part_share"] < cp.MIN_LARGEST_PART


def test_a_passing_check_is_recorded_and_exits_zero(tmp_path, monkeypatch):
    image, root, paths = _stage_a_run(tmp_path, monkeypatch)
    _stub_stages(monkeypatch, paths, lambda final: _box_glb(final, (1, 1, 1)))
    assert cp.main([str(image), "--out", str(root)]) == 0
    assert json.loads(paths["provenance"].read_text())["quality"]["passed"] is True


def test_a_model_that_cannot_be_measured_still_gets_its_provenance(tmp_path, monkeypatch):
    image, root, paths = _stage_a_run(tmp_path, monkeypatch)
    _stub_stages(monkeypatch, paths, lambda final: final.write_bytes(b"not a glb"))
    with pytest.raises(SystemExit, match="FAILED quality check.*could not be measured"):
        cp.main([str(image), "--out", str(root)])
    record = json.loads(paths["provenance"].read_text())
    assert record["license"] == cp.LICENSE
    assert record["quality"]["passed"] is False and record["quality"]["error"]
