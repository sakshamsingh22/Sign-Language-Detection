# Research Experiments & Empirical Evaluation

## 1. Experimental Methodology
This project implements a transfer learning vision pipeline to classify Indian Sign Language (ISL) static hand gestures in real-time. Hand regions are detected and isolated using MediaPipe via the `cvzone.HandTrackingModule.HandDetector` wrapper. A bounding box with a 20-pixel margin is computed, and the cropped hand is resized and centered onto a constant 300x300 white background canvas to preserve aspect ratio and eliminate background noise before feeding it into the model.

**Model Architecture**:
- Base: MobileNetV2 (pre-trained on ImageNet, frozen in initial transfer learning phase, fine-tuned in Phase 2)
- Input Shape: (224, 224, 3)
- Classification Head:
  - `GlobalAveragePooling2D()`
  - `Dropout(0.3)`
  - `Dense(128, activation='relu')`
  - `Dropout(0.2)`
  - `Dense(8, activation='softmax')`
- Loss: `categorical_crossentropy`
- Optimizer: `tf.keras.optimizers.legacy.Adam` (initial lr = 1e-3, fine-tuning lr = 1e-4)

---

## 2. Dataset Distribution & Split Audit

### 2.1 Overall Dataset Distribution
- **Total Images**: 1,745
- **Classes (8)**:
  - Hello: 254
  - Home: 166
  - I love you: 306
  - No: 410
  - Okay: 119
  - Please: 57
  - Thank you: 287
  - Yes: 146

### 2.2 Critical Finding in Existing Training Pipeline (`train.py`)
In `train.py`, the validation split was loaded as:
```python
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    shuffle=False  # <--- CRITICAL
)
```
When `shuffle=False` is supplied with `validation_split`, Keras does not sample 20% evenly from every class. Instead, it takes the last 20% of images in alphabetical folder order. Consequently:
- Classes 0 to 5 (`Hello`, `Home`, `I love you`, `No`, `Okay`, `Please`) received **0 validation samples**.
- All 349 validation samples were exclusively from `Thank you` (203 samples) and `Yes` (146 samples).
- This creates severe data leakage and invalidates the nominal "validation accuracy" as a reflection of general multi-class performance.

---

## 3. Empirical Results

### 3.1 Full Dataset Evaluation of the Trained Production Model (`Model/keras_model.h5`)
To evaluate the true classification capacity of the existing trained model across all 8 classes without modifying any project code or weights, inference was executed across all 1,745 images:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Hello** | 0.9960 | 0.9764 | 0.9861 | 254 |
| **Home** | 0.9881 | 1.0000 | 0.9940 | 166 |
| **I love you** | 0.9796 | 0.9412 | 0.9600 | 306 |
| **No** | 0.9630 | 0.9512 | 0.9571 | 410 |
| **Okay** | 0.9808 | 0.8571 | 0.9148 | 119 |
| **Please** | 1.0000 | 0.9474 | 0.9730 | 57 |
| **Thank you** | 0.9727 | 0.9930 | 0.9828 | 287 |
| **Yes** | 0.8202 | 1.0000 | 0.9012 | 146 |
| **Overall Accuracy** | — | — | **0.9622 (96.22%)** | 1,745 |
| **Macro Average** | 0.9625 | 0.9583 | 0.9586 | 1,745 |
| **Weighted Average** | **0.9652** | **0.9622** | **0.9625** | 1,745 |

**Inference Latency & Throughput**:
- Average Latency: **3.24 ms per image**
- Throughput: **~308 FPS** (batch inference)

---

## 4. Addressing Lack of New Video Data for Research Paper

When new video collection is not feasible, standard academic best practices provide two robust solutions:

### Solution A: Stratified 5-Fold Cross Validation (`research_results/kfold_evaluation.py`)
- Standard academic gold standard for medium-sized computer vision datasets.
- Automatically creates 5 stratified folds ensuring every class is represented equally in every test fold.
- Trains 5 models on 80% and evaluates on the unseen 20% fold across 5 rotations.
- Produces mean ± standard deviation for Accuracy, Precision, Recall, and F1.
- Script created: `research_results/kfold_evaluation.py` (strictly in research directory, leaves `Model/` untouched).

### Solution B: Stratified Train/Val/Test Partitioning
- Split existing 1,745 images into 70% Train (1,221), 15% Validation (262), and 15% Test (262) using `StratifiedShuffleSplit`.
- Report true held-out test metrics.

---

## 5. Directory Artifacts

All evaluation scripts and artifacts have been isolated into `research_results/` without altering production code:
- `full_dataset_classification_report.txt`: Complete 8-class precision/recall breakdown.
- `full_dataset_class_metrics.csv`: Tabular per-class metrics.
- `full_dataset_confusion_matrix.png`: Heatmap showing true vs predicted counts across all 8 classes.
- `full_dataset_metrics.json`: Machine-readable summary of accuracy and latency.
- `dataset_distribution.png`: Bar chart of raw class sample counts.
- `evaluate_validation.py`: Standalone script reproducing the original 20% validation split.
- `kfold_evaluation.py`: Academic 5-Fold CV evaluation pipeline.
