import argparse
from pathlib import Path
import hashlib
import statistics
import torch
from PIL import Image
from torchvision import transforms
from .model import build_model
from .video_processor import extract_frames


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"]); model.to(device).eval()
    return model, ckpt


def predict(video, checkpoint, interval=1.0, max_frames=12, device=None, batch_size=32, aggregation="mean", threshold=None):
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be 'mean' or 'median'")
    device = torch.device((device if device and device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")))
    model, ckpt = load_checkpoint(checkpoint, device)
    if threshold is None:
        threshold = ckpt.get("decision_thresholds", {}).get(aggregation, 0.5)
    size = ckpt["image_size"]; norm = ckpt["normalization"]
    transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(norm["mean"], norm["std"])])
    video_path = Path(video).resolve()
    key = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:16]
    temp = Path(checkpoint).parent / "prediction_frames" / f"{video_path.stem}_{key}"
    paths, err = extract_frames(video_path, temp, interval, max_frames, size)
    if err: raise RuntimeError(err)
    tensors=[]
    for path in paths:
        with Image.open(path) as im: tensors.append(transform(im.convert("RGB")))
    scores=[]
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            x=torch.stack(tensors[start:start+batch_size]).to(device)
            scores.extend(torch.softmax(model(x),1)[:,1].cpu().tolist())
    score = sum(scores)/len(scores) if aggregation == "mean" else statistics.median(scores)
    return ("AI_GENERATED" if score >= threshold else "REAL"), score, len(scores)


def main():
    ap=argparse.ArgumentParser(description="Predict a video's class using mean sampled-frame score")
    ap.add_argument("--video",required=True); ap.add_argument("--checkpoint",default="results/models/best_model.pt")
    ap.add_argument("--interval",type=float,default=1.0); ap.add_argument("--max-frames",type=int,default=12); ap.add_argument("--device",choices=["cpu","cuda"])
    ap.add_argument("--aggregation",choices=["mean","median"],default="mean",help="How sampled-frame AI scores are combined")
    ap.add_argument("--threshold",type=float,help="Override the validation-calibrated decision threshold")
    args=ap.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    threshold=args.threshold if args.threshold is not None else checkpoint.get("decision_thresholds",{}).get(args.aggregation,0.5)
    label,score,count=predict(args.video,args.checkpoint,args.interval,args.max_frames,args.device,aggregation=args.aggregation,threshold=threshold)
    warning="\nWarning: score is close to the decision threshold; treat this result as uncertain." if abs(score-threshold)<0.05 else ""
    print(f"Video: {Path(args.video).name}\nPrediction: {label}\nAI-generated score ({args.aggregation} frame softmax; uncalibrated): {score:.4f}\nDecision threshold: {threshold:.4f}\nFrames analyzed: {count}{warning}")

if __name__=="__main__": main()
