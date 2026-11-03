from pathlib import Path
import cv2


def extract_frames(video_path: Path, output_dir: Path, interval_seconds: float = 1.0,
                   max_frames: int = 12, size: int = 224):
    """Sample unique frames through VideoCapture; returns saved frame paths and error."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.jpg"))
    if existing:
        return existing[:max_frames], None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], "OpenCV could not open video"
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not fps or fps <= 0 or not count or count <= 0:
        cap.release()
        return [], "Video has invalid FPS or frame count"
    duration = count / fps
    step = max(1, int(round(interval_seconds * fps)))
    if duration / interval_seconds > max_frames:
        step = max(step, int(count / max_frames))
    paths, last = [], -1
    index = 0
    while len(paths) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0 and index != last:
            h, w = frame.shape[:2]
            scale = size / max(h, w)
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = cv2.copyMakeBorder(frame, (size-nh)//2, size-nh-(size-nh)//2,
                                        (size-nw)//2, size-nw-(size-nw)//2,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            dest = output_dir / f"frame_{len(paths):04d}.jpg"
            if cv2.imwrite(str(dest), canvas):
                paths.append(dest)
                last = index
        index += 1
    cap.release()
    if not paths:
        return [], "No readable frames extracted"
    return paths, None
