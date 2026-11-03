# Model artifacts

Training writes `results/models/best_model.pt`. It contains ResNet50 weights, class IDs, image size, normalization, and the run configuration. Model weights are generated locally and excluded from Git. The checkpoint is selected by validation frame-level F1; test data is reserved for final video-level evaluation.
