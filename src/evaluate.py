import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, roc_curve)
from .train import read_config
from .dataset_loader import FrameDataset
from .predict_video import load_checkpoint
from .utils import ensure_dirs


def score_frames(model, frame_table, transform, device, batch_size, num_workers):
    loader=DataLoader(FrameDataset(frame_table,transform),batch_size=batch_size,shuffle=False,num_workers=num_workers)
    per_video={}
    with torch.inference_mode():
        for x,_,video_ids in loader:
            frame_scores=torch.softmax(model(x.to(device)),dim=1)[:,1].cpu().tolist()
            for video_id,score in zip(video_ids,frame_scores):
                per_video.setdefault(video_id,[]).append(score)
    return per_video


def select_threshold(scores, truth):
    # Search score boundaries and choose the center of the best macro-F1 plateau.
    unique=np.unique(np.asarray(scores,dtype=float))
    candidates=np.concatenate(([0.0],(unique[:-1]+unique[1:])/2,[1.0]))
    f1_values=np.asarray([f1_score(truth,np.asarray(scores)>=t,average="macro",zero_division=0)
                          for t in candidates])
    best=f1_values.max()
    plateau=candidates[np.isclose(f1_values,best)]
    return float(np.median(plateau)), float(best)


def main():
    ap=argparse.ArgumentParser(description="Evaluate best checkpoint on video-level held-out test split")
    ap.add_argument("--config",default="config.yaml"); ap.add_argument("--checkpoint")
    args=ap.parse_args(); cfg=read_config(args.config); out=cfg["output_dir"]
    ckpt=args.checkpoint or str(out/"models"/"best_model.pt")
    manifest=pd.read_csv(out/"test_videos.csv")
    device = "cuda" if torch.cuda.is_available() and cfg["device"] != "cpu" else "cpu"
    model, checkpoint = load_checkpoint(ckpt, device)
    transform=transforms.Compose([
        transforms.Resize((checkpoint["image_size"],checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["normalization"]["mean"],checkpoint["normalization"]["std"])])
    frame_table=pd.read_csv(out/"sampled_frames.csv")
    monitor=pd.read_csv(out/"validation_monitor_videos.csv").video_path.tolist()
    val_frames=frame_table[(frame_table.split == "validation") & frame_table.video_path.isin(monitor)]
    test_frames=frame_table[frame_table.split == "test"]
    batch_size=max(16,cfg["batch_size"]*2)
    val_scores=score_frames(model,val_frames,transform,device,batch_size,cfg["num_workers"])
    per_video=score_frames(model,test_frames,transform,device,batch_size,cfg["num_workers"])
    val_manifest=pd.read_csv(out/"validation_videos.csv")
    val_truth=[]; val_mean=[]; val_median=[]
    for row in val_manifest.itertuples(index=False):
        values=val_scores.get(row.video_path,[])
        if row.video_path in monitor and values:
            val_truth.append(int(row.label_id)); val_mean.append(float(np.mean(values))); val_median.append(float(np.median(values)))
    thresholds={}
    for method,values in [("mean",val_mean),("median",val_median)]:
        threshold,val_macro_f1=select_threshold(values,val_truth)
        thresholds[method]={"threshold":threshold,"validation_macro_f1":val_macro_f1,
                            "validation_balanced_accuracy":float(balanced_accuracy_score(val_truth,np.asarray(values)>=threshold))}
    checkpoint["decision_thresholds"]={key:value["threshold"] for key,value in thresholds.items()}
    checkpoint["threshold_calibration"]={"method":"maximize video-level macro F1 on fixed balanced validation monitor; choose center of best threshold plateau",
                                         "videos_per_class":len(val_truth)//2,"metrics":thresholds}
    torch.save(checkpoint,ckpt)
    (out/"decision_thresholds.json").write_text(json.dumps(checkpoint["threshold_calibration"],indent=2),encoding="utf-8")
    selected_threshold=thresholds["mean"]["threshold"]
    y_true=[]; scores=[]; frame_counts=[]; names=[]
    for row in manifest.itertuples(index=False):
        video_scores=per_video.get(row.video_path,[])
        if not video_scores: continue
        y_true.append(int(row.label_id)); scores.append(float(sum(video_scores)/len(video_scores)))
        frame_counts.append(len(video_scores)); names.append(row.video_path)
    if not y_true: raise RuntimeError("No test frames were found in sampled_frames.csv")
    y_pred=[int(s>=selected_threshold) for s in scores]
    acc=accuracy_score(y_true,y_pred)
    balanced_acc=balanced_accuracy_score(y_true,y_pred)
    p,r,f,support=precision_recall_fscore_support(y_true,y_pred,labels=[0,1],zero_division=0)
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    result={"videos_evaluated":len(y_true),"frames_evaluated":sum(frame_counts),"decision_threshold":selected_threshold,
            "validation_monitor_macro_f1":thresholds["mean"]["validation_macro_f1"],
            "accuracy":acc,"balanced_accuracy":balanced_acc,
            "REAL_precision":p[0],"REAL_recall":r[0],"REAL_f1":f[0],
            "AI_GENERATED_precision":p[1],"AI_GENERATED_recall":r[1],"AI_GENERATED_f1":f[1],
            "REAL_support":support[0],"AI_GENERATED_support":support[1],
            "macro_f1":(f[0]+f[1])/2,"confusion_matrix_tn_fp_fn_tp":cm.ravel().tolist()}
    if len(set(y_true))==2: result["roc_auc"]=roc_auc_score(y_true,scores)
    ensure_dirs(out)
    pd.DataFrame([result]).to_csv(out/"evaluation_metrics.csv",index=False)
    pd.DataFrame({"video_path":names,"true_label":y_true,"ai_generated_score":scores,"predicted_label":["AI_GENERATED" if v else "REAL" for v in y_pred],"frames":frame_counts}).to_csv(out/"video_predictions.csv",index=False)
    disp=ConfusionMatrixDisplay(cm,display_labels=["REAL","AI_GENERATED"]); disp.plot(cmap="Blues",values_format="d"); plt.tight_layout(); plt.savefig(out/"confusion_matrix.png",dpi=160); plt.close()
    if "roc_auc" in result:
        fpr,tpr,_=roc_curve(y_true,scores); plt.plot(fpr,tpr,label=f"ROC-AUC={result['roc_auc']:.3f}"); plt.plot([0,1],[0,1],"--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.tight_layout(); plt.savefig(out/"roc_curve.png",dpi=160); plt.close()
    print(pd.Series(result).to_string())

if __name__=="__main__": main()
