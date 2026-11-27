import cv2
import numpy as np
import pytest

from video_real2sim.config import SamplingSettings
from video_real2sim.errors import IntegrityError
from video_real2sim.video import preprocess


def synthetic_video(path, count=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 90))
    assert writer.isOpened()
    for i in range(count):
        frame = np.full((90, 160, 3), 20, np.uint8)
        # Hidden yellow content becomes visible only in the second half.
        if i >= count // 2:
            frame[20:70, 40:120] = [0, 220, 220]
        writer.write(frame)
    writer.release()


def test_full_decode_first_last_and_reveal(tmp_path):
    video = tmp_path / "source.mp4"
    synthetic_video(video)
    settings = SamplingSettings(uniform_frames=4, event_frames=2)
    result = preprocess(video, tmp_path / "evidence", settings)
    frames = [r["frame"] for r in result["selected_frames"]]
    assert result["decoded_frames"] == 30
    assert frames[0] == 0 and frames[-1] == 29 and 15 in frames
    assert len(frames) == 6
    assert preprocess(video, tmp_path / "evidence", settings) == result


def test_zero_events_and_changed_config(tmp_path):
    video = tmp_path / "source.mp4"
    synthetic_video(video)
    result = preprocess(
        video, tmp_path / "evidence", SamplingSettings(uniform_frames=4, event_frames=0)
    )
    assert len(result["selected_frames"]) == 4
    with pytest.raises(IntegrityError):
        preprocess(video, tmp_path / "evidence", SamplingSettings(uniform_frames=8))
