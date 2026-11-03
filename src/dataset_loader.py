from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = {"REAL": 0, "AI_GENERATED": 1}
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg")


def scan_dataset(class_dirs, extensions=VIDEO_EXTENSIONS):
    rows = []
    for label, directories in class_dirs.items():
        if label not in CLASSES:
            raise ValueError(f"Unsupported label {label}; use REAL or AI_GENERATED")
        roots = directories if isinstance(directories, (list, tuple)) else [directories]
        for directory in roots:
            root = Path(directory)
            if not root.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {root}")
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in extensions:
                    # Keep camera views and numbered-folder source clips grouped.
                    if label == "REAL" and path.parent.name.lower().startswith("scene_"):
                        group = f"{root.name}:{path.parent.name}"
                    elif label == "REAL" and root.name.lower() == "files":
                        group = f"{root.name}:{path.parent.name}"
                    else:
                        group = path.stem
                    rows.append({"video_path": str(path.resolve()), "label": label,
                                 "label_id": CLASSES[label], "video_id": path.stem, "group_id": group})
    if not rows:
        raise ValueError("No supported videos found in configured class directories")
    return pd.DataFrame(rows).drop_duplicates("video_path").reset_index(drop=True)


def load_video_manifest(manifest_path, extensions=VIDEO_EXTENSIONS):
    """Load manually verified labels. Optional group_id keeps related clips together."""
    manifest_path = Path(manifest_path).resolve()
    table = pd.read_csv(manifest_path)
    required = {"video_path", "label"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing))}")
    rows = []
    for row in table.itertuples(index=False):
        label = str(row.label).strip().upper()
        if label not in CLASSES:
            raise ValueError(f"Invalid label {label!r}; labels must be REAL or AI_GENERATED")
        video_path = Path(str(row.video_path)).expanduser()
        if not video_path.is_absolute():
            video_path = manifest_path.parent / video_path
        video_path = video_path.resolve()
        if video_path.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported video extension: {video_path}")
        if not video_path.is_file():
            raise FileNotFoundError(f"Manifest video does not exist: {video_path}")
        if "group_id" in table.columns and pd.notna(row.group_id) and str(row.group_id).strip():
            group = str(row.group_id).strip()
        else:
            group = f"manifest_{video_path.parent.name}"
        rows.append({"video_path": str(video_path), "label": label,
                     "label_id": CLASSES[label], "video_id": video_path.stem,
                     "group_id": group})
    return pd.DataFrame(rows).drop_duplicates("video_path").reset_index(drop=True)


class FrameDataset(Dataset):
    def __init__(self, frame_table, transform=None):
        self.frames = frame_table.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self): return len(self.frames)

    def __getitem__(self, index):
        row = self.frames.iloc[index]
        with Image.open(row.frame_path) as image:
            x = self.transform(image.convert("RGB"))
        return x, int(row.label_id), str(row.video_id)
