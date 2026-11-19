import json

import cv2
import numpy as np

from src.runtime.run_manager import RunContext


def _video(path, color):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for _ in range(12):
        writer.write(np.full((24, 32, 3), color, dtype=np.uint8))
    writer.release()


def test_two_videos_and_repeat_are_isolated(tmp_path):
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    _video(first, (0, 0, 255))
    _video(second, (0, 255, 0))
    run_a = RunContext.create(first, output_root=tmp_path / "runs", category="bottle")
    run_b = RunContext.create(second, output_root=tmp_path / "runs", category="cup")
    run_a_repeat = RunContext.create(first, output_root=tmp_path / "runs", category="bottle")

    assert len({run_a.run_id, run_b.run_id, run_a_repeat.run_id}) == 3
    manifest_a = json.loads(run_a.manifest_path.read_text())
    manifest_b = json.loads(run_b.manifest_path.read_text())
    assert manifest_a["input"]["sha256"] != manifest_b["input"]["sha256"]
    assert run_a.video_path.parent == run_a.root / "input"
    assert run_b.video_path.parent == run_b.root / "input"
    assert run_a.video_path.read_bytes() == first.read_bytes()
    assert run_a_repeat.video_path.read_bytes() == first.read_bytes()
