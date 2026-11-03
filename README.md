# AI-Generated Video Detection with Transfer Learning

Python project for the research question: **How effectively can transfer learning distinguish real camera-recorded videos from AI-generated videos using extracted visual frames?** It includes command-line training and evaluation plus a browser upload interface backed by a FastAPI prediction endpoint. The model samples video frames, classifies them with a pretrained ResNet-50, and aggregates frame scores into a video estimate.

## Current data and experiment status

The supplied `I2V/` directory contains 3,000 MP4 candidates in three generator-named nested folders. The user confirmed that videos in both `archive/` and `files/` are authentic camera recordings. The archive has 20 MPEG files (one zero-byte); `files` has 40 MP4 videos and 40 JPG stills. The JPGs are not used. The configuration uses the 60 MP4/MPEG paths as `REAL` and I2V as `AI_GENERATED`; there are 59 usable REAL videos after the zero-byte file is skipped. I2V has no accompanying label documentation in this workspace, so verify its mapping against the original source before using results in a paper. No dataset was downloaded.

Training and held-out evaluation were run on Python 3.11.9, PyTorch 2.14.1 CPU, torchvision 0.29.1, and OpenCV 4.10.0. Source compilation, CLI help, video scanning/extraction, checkpoint loading, and real-checkpoint prediction checks passed. One zero-byte real video was excluded and recorded in `results/extraction_errors.csv`.

## Dataset layout

```text
I2V/                    # AI_GENERATED candidates (place locally; not committed)
archive/                # REAL videos (place locally; not committed)
files/                  # additional REAL videos (place locally; not committed)
dataset/
  extra_dataset_manifest_template.csv
```

Place your labeled collections in the folders shown above or update `class_dirs` in `config.yaml`. Dataset files are excluded from Git. Nested directories are scanned for MP4, MOV, AVI, MKV, WEBM, MPEG, and MPG files. Splitting happens before frames are assigned, at video level; clips under a real `scene_*` folder are grouped so camera views of one scene remain in the same split. For extra labeled data in mixed folders, set `additional_manifest_csv`; see [dataset/README.md](dataset/README.md) and the manifest template. Related files need a shared `group_id` in that manifest.

The project does not fetch datasets. Obtain permitted data from its official source and retain its label documentation. [FaceForensics++](https://github.com/ondyari/FaceForensics) provides original YouTube and manipulated videos; [Celeb-DF](https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics.html) provides real and DeepFake videos. Their manipulated examples may not represent modern text-to-video or image-to-video generators. Record the chosen subset, license/access terms, and domain limitation in research notes. Do not map a folder to a class based only on convenient filenames.

## Setup

Use Python 3.10–3.12 and install a PyTorch/torchvision pair compatible with the target CPU or CUDA installation, then install the remaining pinned dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA, follow the current PyTorch installation selector for the machine's driver, then install the matching torchvision; use the CPU wheels on machines without a supported GPU. The code automatically selects CUDA if available, and `device: cpu` forces CPU. CPU training may take substantially longer. The first training run also obtains the torchvision pretrained ResNet50 weights if they are not already cached; network access is needed for that one-time weight download.

## Run

From the repository root, after configuring both labeled directories:

```powershell
python -m src.train --config config.yaml
python -m src.evaluate --config config.yaml
python -m src.predict_video --video "path/to/video.mp4" --checkpoint results/models/best_model.pt
```

## Web upload and Render deployment

The project includes a browser upload page and a FastAPI prediction endpoint. To deploy it, connect this GitHub repository's `main` branch to Render as a Blueprint using the root `render.yaml` file. The Blueprint uses the free plan and a single CPU worker. Render requires web services to listen on `0.0.0.0` and its assigned `$PORT`; the supplied start command does both.

The trained checkpoint is not stored in Git. Before the web page can predict, upload `results/models/best_model.pt` to a trusted HTTPS artifact host and set the Render `MODEL_URL` environment variable to its direct download URL. Treat the checkpoint as trusted input because it is a PyTorch pickle. The app downloads it at startup, holds the model in memory, limits video uploads to 100 MB, serializes inference requests, and removes uploaded videos and extracted frames after each request. The service needs outbound HTTPS access to download the checkpoint.

The free Render instance has 512 MB RAM and 0.1 CPU, so ResNet-50 inference may be slow or exceed available memory. Upgrade the service only if you choose a paid plan; do not assume free-tier deployment is suitable for production. The free service may spin down when idle.

Training options: `--epochs`, `--batch-size`, `--learning-rate`, and `--seed`. The config controls paths, frame sampling, image size, batch size, epochs, and worker count. The pipeline samples up to 12 frames approximately once per second by default, caches resized frames, records extraction errors, and writes per-video split manifests. Group-wise splitting requires at least three independent groups per class; each group remains in one split. Split targets are approximately 70/15/15 by video.


The model starts from ImageNet pretrained ResNet50 with a frozen feature backbone (including frozen BatchNorm running statistics) and a newly initialized two-class head. AdamW optimizes the head. Training uses replacement sampling balanced by class and video, capped at 128 sampled frames per class per epoch by default for CPU practicality. Model selection uses a fixed, seeded, class-balanced subset of validation videos (listed in `validation_monitor_videos.csv`); the full validation and test splits remain intact. The checkpoint with best validation macro F1 is saved. Evaluation calibrates a decision threshold per aggregation method on the balanced validation monitor by maximizing video-level macro F1, then evaluates the untouched test split. Prediction defaults to mean aggregation and also offers `--aggregation median`; scores are uncalibrated model outputs, not calibrated probabilities or proof of authenticity.

### Completed run (8 epochs)

The expanded run used 40 REAL / 2,100 AI_GENERATED training videos, 8 / 450 validation videos, and 11 / 450 test videos. The held-out test set had 1,427 sampled frames. The validation-selected mean-score threshold was **0.6288**. Held-out test accuracy was **0.957**, balanced accuracy **0.978**, macro F1 **0.751**, and ROC-AUC **0.997**. REAL precision/recall/F1 were **0.355 / 1.000 / 0.524**; AI_GENERATED precision/recall/F1 were **1.000 / 0.956 / 0.977**. Confusion matrix (REAL first) was `[[11, 0], [20, 430]]`.

These results remain limited: the real class has 59 usable videos across 15 source groups, and the test split contains 11 real videos across three groups. The validation monitor contains eight real videos from two groups. This is a small pilot, and precision for REAL remains low. The I2V class mapping is inferred from the folder name and still needs source documentation. Do not use these figures as evidence of generalization to unseen generators or camera scenes.

Two additional user-provided, out-of-test probes are recorded in `results/external_validation_cases.csv`. The Gemini AI clip now predicts AI, while the real WhatsApp clip is still a false positive under both mean and median aggregation. The model therefore remains unreliable on source types not represented in its small training set; more diverse, labeled messaging-app REAL videos and Gemini-generated examples are needed, with source groups held out during evaluation.

## Outputs

- `results/models/best_model.pt`: reusable model and preprocessing metadata.
- `results/{train,validation,test}_videos.csv`: video split manifests.
- `results/validation_monitor_videos.csv`: fixed balanced subset used for checkpoint selection.
- `results/sampled_frames.csv`, `dataset_summary.csv`, cached frames, and `extraction_errors.csv`.
- `results/training_history.csv`: per-epoch loss, accuracy, and F1 (frame-level).
- `results/training_loss.png`: train and validation loss curves.
- `results/evaluation_metrics.csv`, `decision_thresholds.json`, `video_predictions.csv`, `confusion_matrix.png`, and ROC plot when valid.

## Reproducibility and limitations

The seed and run settings are saved in the checkpoint; split manifests and preprocessing settings are retained. Exact deterministic GPU behavior is not guaranteed across hardware/library versions. Evaluation is video-level; training logs are frame-level. Multiple frames from a video remain together in one split, although source-level duplicate grouping across separate files requires dataset metadata and manual curation. Frame averaging discards temporal information. Results may be sensitive to compression, resolution, dataset source, generator version, and visual domain. A model trained on a narrow set of generators may not generalize to unseen generators or camera sources. The archive contributes only 19 nonempty real videos against 3,000 I2V candidates, so test estimates will be especially uncertain; report class counts and avoid broad claims. A frozen-feature logistic-regression baseline is not yet implemented.

## Troubleshooting

- **Missing REAL class / no binary dataset:** add a verified real-video directory to `config.yaml`; the code does not invent labels.
- **Group-wise split fails:** each class needs at least three independent groups. Collect more groups; do not split frames to work around this.
- **Unreadable video:** inspect `results/extraction_errors.csv`; OpenCV codec support varies by platform.
- **CUDA out of memory:** reduce `batch_size` and/or `max_frames`; use `device: cpu` if needed.
- **Pretrained weights unavailable:** resolve network access or provision the official torchvision ResNet50 weights in its cache before training.
- **Resume:** automatic optimizer-state resume is not implemented; rerun training. The best checkpoint remains available for inference.


