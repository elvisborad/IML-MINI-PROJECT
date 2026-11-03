# Dataset status

The supplied workspace contains 3,000 MP4 files under `I2V/`, organized into `moonvalley_mp4`, `neverends_mp4`, and `pika_mp4` directories (each has another same-named nested directory). The folder names suggest image-to-video outputs, but there is no dataset card, label manifest, or metadata. The experiment used this path as `AI_GENERATED` provisionally based on the project-specific folder name; verify it against source documentation before treating it as ground truth. No `T2V` directory was found.

No files have been moved or relabeled. The configured run uses the user-confirmed camera recordings at `archive/` for `REAL`. The code scans nested directories for common video extensions. Keep original source clips in a single class and avoid duplicates across classes.

The user provided `archive/` and confirmed its contents are authentic camera recordings. Inspection found 20 `.mpeg` files in five `scene_S...` directories; 19 are nonzero, and an OpenCV probe decoded a nonempty 1920x1080 sample. The zero-byte file is recorded as an extraction error and excluded before splitting. The source archive has no separate label document, so retain the user's confirmation with experiment notes. Real `scene_*` directories are treated as groups and kept within one split. `.mpeg`/`.mpg` discovery is supported.

## Additional REAL videos

`files/` contains 10 numbered folders, each with four MP4 files and four JPG files (40 MP4 and 40 JPG total). The user confirmed that all 40 MP4 files are REAL. The JPGs are still images and are not used by this video pipeline. Each numbered folder is grouped as one source during splitting to avoid camera/person/source leakage.

For other mixed folders without class subdirectories, use `extra_dataset_manifest_template.csv` with one row per video, a verified `label` (`REAL` or `AI_GENERATED`), and a shared `group_id` for related clips. The trainer validates each path and label and keeps each group within one split.

Potential public source: [FaceForensics++](https://github.com/ondyari/FaceForensics) provides original YouTube videos and manipulated versions. The original videos can supply a real class, while the manipulated videos are face manipulations rather than necessarily generative text/image-to-video outputs; describe that domain mismatch in the paper. [Celeb-DF](https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics.html) also documents real and DeepFake videos, with a similar limitation. Dataset access, licensing, and exact label mapping must be checked by the researcher; this project does not download data.

