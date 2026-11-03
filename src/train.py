import argparse
from pathlib import Path
import hashlib
import yaml
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from .dataset_loader import scan_dataset, load_video_manifest, FrameDataset
from .model import build_model
from .utils import seed_everything, resolve_path, ensure_dirs
from .video_processor import extract_frames


def read_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f: cfg = yaml.safe_load(f)
    cfg["_base"] = path.parent
    cfg["class_dirs"] = {
        k: [resolve_path(item, path.parent) for item in v] if isinstance(v, list) else resolve_path(v, path.parent)
        for k, v in cfg["class_dirs"].items()
    }
    if cfg.get("additional_manifest_csv"):
        cfg["additional_manifest_csv"] = resolve_path(cfg["additional_manifest_csv"], path.parent)
    cfg["output_dir"] = resolve_path(cfg["output_dir"], path.parent)
    return cfg


def split_videos(videos, seed, ratios=(.70, .15, .15)):
    # Split groups separately per class; REAL multi-camera scene clips remain together.
    try:
        group_splits = {"train": set(), "validation": set(), "test": set()}
        for label, subset in videos.groupby("label"):
            groups = sorted(subset.group_id.unique().tolist())
            if len(groups) < 3:
                raise ValueError(f"Class {label} has only {len(groups)} independent groups; at least 3 are needed.")
            train_groups, hold_groups = train_test_split(
                groups, test_size=ratios[1]+ratios[2], random_state=seed)
            val_groups, test_groups = train_test_split(
                hold_groups, test_size=ratios[2]/(ratios[1]+ratios[2]), random_state=seed)
            group_splits["train"].update((label, group) for group in train_groups)
            group_splits["validation"].update((label, group) for group in val_groups)
            group_splits["test"].update((label, group) for group in test_groups)
        return tuple(videos[[ (row.label, row.group_id) in group_splits[name]
                              for row in videos.itertuples(index=False) ]].copy()
                     for name in ("train", "validation", "test"))
    except ValueError as exc:
        raise ValueError("Cannot make group-separated train/validation/test splits. Each class needs at least three independent groups.") from exc


def expand_frames(videos, cfg):
    rows, errors = [], []
    root = cfg["output_dir"] / "frames"
    for row in tqdm(videos.itertuples(index=False), total=len(videos), desc="Extracting frames"):
        key = hashlib.sha1(row.video_path.encode("utf-8")).hexdigest()[:16]
        paths, error = extract_frames(Path(row.video_path), root / f"{row.label}_{key}",
                                     cfg["frame_interval_seconds"], cfg["max_frames"], cfg["image_size"])
        if error: errors.append({"video_path": row.video_path, "label": row.label, "error": error})
        for p in paths:
            rows.append({"video_id": row.video_path, "video_path": row.video_path,
                         "label": row.label, "label_id": row.label_id, "frame_path": str(p)})
    pd.DataFrame(errors, columns=["video_path", "label", "error"]).to_csv(cfg["output_dir"] / "extraction_errors.csv", index=False)
    return pd.DataFrame(rows)


def set_phase(model, training):
    model.train(training)
    if training:
        # The feature extractor is frozen; keep its BatchNorm running stats fixed too.
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def main():
    ap = argparse.ArgumentParser(description="Train a video-level AI-generated video detector")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    cfg = read_config(args.config)
    for key, value in [("epochs", args.epochs), ("batch_size", args.batch_size),
                       ("learning_rate", args.learning_rate), ("seed", args.seed)]:
        if value is not None: cfg[key] = value
    ensure_dirs(cfg["output_dir"], cfg["output_dir"] / "models")
    seed_everything(cfg["seed"])
    sources = [scan_dataset(cfg["class_dirs"])]
    if cfg.get("additional_manifest_csv"):
        sources.append(load_video_manifest(cfg["additional_manifest_csv"]))
    videos = pd.concat(sources, ignore_index=True)
    conflicts = videos.groupby("video_path").label.nunique()
    if (conflicts > 1).any():
        raise ValueError("The same video path has conflicting labels across configured data sources.")
    videos = videos.drop_duplicates("video_path").reset_index(drop=True)
    present = set(videos.label)
    if present != {"REAL", "AI_GENERATED"}:
        missing = sorted({"REAL", "AI_GENERATED"} - present)
        raise ValueError("Binary training requires labeled examples from both classes. Missing: " + ", ".join(missing) +
                         ". Configure verified class directories in config.yaml; I2V/T2V are not REAL data.")
    # Extract first so zero-byte/corrupt videos do not consume split slots.
    frames = expand_frames(videos, cfg)
    if frames.empty: raise RuntimeError("No frames extracted; see extraction_errors.csv")
    videos = videos[videos.video_path.isin(frames.video_path.unique())].reset_index(drop=True)
    present = set(videos.label)
    if present != {"REAL", "AI_GENERATED"}:
        raise ValueError("After skipping unreadable files, both REAL and AI_GENERATED videos must remain.")
    train_v, val_v, test_v = split_videos(videos, cfg["seed"])
    for name, subset in [("train", train_v), ("validation", val_v), ("test", test_v)]:
        subset.to_csv(cfg["output_dir"] / f"{name}_videos.csv", index=False)
        print(f"{name}: {len(subset)} videos; " + ", ".join(f"{k}={v}" for k,v in subset.label.value_counts().to_dict().items()))
    # Video-level membership is assigned after extraction; frames never cross splits.
    split_of = {p: name for name, subset in [("train", train_v), ("validation", val_v), ("test", test_v)] for p in subset.video_path}
    frames["split"] = frames.video_path.map(split_of)
    frames.to_csv(cfg["output_dir"] / "sampled_frames.csv", index=False)
    for name in ["train", "validation", "test"]:
        subset = frames[frames.split == name]
        print(f"{name}: {subset.video_path.nunique()} videos, {len(subset)} frames")
    summary = (frames.groupby(["split", "label"], as_index=False)
               .agg(videos=("video_path", "nunique"), frames=("frame_path", "count")))
    summary.to_csv(cfg["output_dir"] / "dataset_summary.csv", index=False)
    mean, std = [0.485, .456, .406], [.229, .224, .225]
    transform = transforms.Compose([transforms.Resize((cfg["image_size"], cfg["image_size"])), transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_frames = frames[frames.split == "train"].reset_index(drop=True)
    per_video_frames = train_frames.groupby("video_id").size().to_dict()
    videos_per_class = train_frames.groupby("label_id").video_id.nunique().to_dict()
    sample_weights = [1.0 / (videos_per_class[int(row.label_id)] * per_video_frames[row.video_id])
                      for row in train_frames.itertuples(index=False)]
    min_class_frames = int(train_frames.label_id.value_counts().min())
    samples_per_class = min(min_class_frames, int(cfg["balanced_samples_per_class"]))
    sampler = WeightedRandomSampler(sample_weights, num_samples=2 * samples_per_class, replacement=True)
    print(f"Balanced training epoch: {samples_per_class} sampled frames per class")
    validation_frames = frames[frames.split == "validation"]
    val_video_labels = validation_frames[["video_path", "label_id"]].drop_duplicates()
    videos_per_val_class = val_video_labels.label_id.value_counts()
    monitor_videos_per_class = int(videos_per_val_class.min())
    monitor_paths = set()
    for label_id, group in val_video_labels.groupby("label_id"):
        monitor_paths.update(group.sample(n=monitor_videos_per_class, random_state=cfg["seed"]).video_path)
    validation_monitor = validation_frames[validation_frames.video_path.isin(monitor_paths)].reset_index(drop=True)
    pd.DataFrame({"video_path": sorted(monitor_paths)}).to_csv(cfg["output_dir"] / "validation_monitor_videos.csv", index=False)
    print(f"Validation model-selection monitor: {monitor_videos_per_class} videos per class, {len(validation_monitor)} frames")
    loaders = {
        "train": DataLoader(FrameDataset(train_frames, transform), batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["num_workers"]),
        "validation": DataLoader(FrameDataset(validation_monitor, transform), batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"]),
    }
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] != "cpu" else "cpu")
    model = build_model(pretrained=True, freeze_backbone=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["learning_rate"])
    best, history = -1, []
    for epoch in range(cfg["epochs"]):
        record = {"epoch": epoch+1}
        for phase in ["train", "validation"]:
            set_phase(model, phase == "train"); loss_sum=0; labels=[]; preds=[]
            for x,y,_ in tqdm(loaders[phase], desc=f"{phase} {epoch+1}", leave=False):
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    logits=model(x); loss=criterion(logits,y)
                    if phase == "train": loss.backward(); optimizer.step()
                loss_sum += loss.item()*len(y); labels.extend(y.cpu().tolist()); preds.extend(logits.argmax(1).cpu().tolist())
            record[f"{phase}_loss"] = loss_sum/max(1,len(loaders[phase].dataset))
            record[f"{phase}_accuracy"] = accuracy_score(labels,preds)
            record[f"{phase}_macro_f1"] = f1_score(labels,preds,average="macro",zero_division=0)
        print(record); history.append(record)
        if record["validation_macro_f1"] > best:
            best=record["validation_macro_f1"]
            torch.save({"state_dict":model.state_dict(),"classes":{"REAL":0,"AI_GENERATED":1},
                        "image_size":cfg["image_size"],"normalization":{"mean":mean,"std":std},
                        "architecture":"resnet50","config":{k:v for k,v in cfg.items() if not k.startswith("_")}},
                       cfg["output_dir"] / "models" / "best_model.pt")
    pd.DataFrame(history).to_csv(cfg["output_dir"] / "training_history.csv", index=False)
    history_df = pd.DataFrame(history)
    plt.plot(history_df.epoch, history_df.train_loss, label="Training loss")
    plt.plot(history_df.epoch, history_df.validation_loss, label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss"); plt.legend(); plt.tight_layout()
    plt.savefig(cfg["output_dir"] / "training_loss.png", dpi=160); plt.close()
    print(f"Best validation-monitor frame-level macro F1: {best:.4f}. Test remains held out for evaluate.py.")


if __name__ == "__main__": main()
