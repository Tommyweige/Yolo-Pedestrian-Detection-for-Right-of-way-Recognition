"""Event times use the source reader's one-based video frame number."""
import math

import cv2


def event_timestamp(frame, dataset, webcam=False, index=0):
    fps = dataset.fps[index] if webcam else dataset.cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0 or frame < 1:
        raise ValueError("事件時間需要有效的來源幀率與幀號。")
    return f"{(frame - 1) / fps:.3f} sec, 第{frame}幀,"


def discard_expired_tracks(active_ids, *states):
    for state in states:
        for identity in state.keys() - active_ids:
            del state[identity]
