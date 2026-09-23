"""
K-Fold Cross Validation Evaluation
------------------------------------
Evaluates the EXISTING trained model's architecture using 5-fold
cross validation on the full dataset. Each fold trains a fresh model
from scratch to ensure unbiased evaluation.

NOTE: This does NOT modify the existing trained model (keras_model.h5).
All output is written to research_results/ only.
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.utils import shuffle as sk_shuffle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "../Data"
RESULTS_DIR = "."
IMG_SIZE    = 224
BATCH_SIZE  = 16
N_FOLDS     = 5
SEED        = 42


def build_model(num_classes):
    """Exact same architecture as the production model in train.py."""
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_dataset(data_dir, img_size):
    """Load all images and labels as numpy arrays."""
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    X, y = [], []
    print(f"\nLoading {len(classes)} classes:")
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files   = [f for f in os.listdir(cls_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  {cls}: {len(files)} images")
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            img   = tf.keras.utils.load_img(fpath, target_size=(img_size, img_size))
            arr   = tf.keras.utils.img_to_array(img) / 255.0
            X.append(arr)
            y.append(class_to_idx[cls])
    return np.array(X, dtype="float32"), np.array(y), classes


def augment_batch(X_batch):
    """Apply same augmentation as train.py."""
    augmentor = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])
    return augmentor(X_batch, training=True).numpy()


def run_kfold():
    print("=" * 60)
    print("  5-Fold Cross Validation – ISL Recognition Project")
    print("=" * 60)

    X, y, class_names = load_dataset(DATA_DIR, IMG_SIZE)
    num_classes = len(class_names)
    X, y = sk_shuffle(X, y, random_state=SEED)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    all_y_true, all_y_pred = [], []
    fold_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'─'*50}")
        print(f"  Fold {fold_idx + 1} / {N_FOLDS}")
        print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")
        print(f"{'─'*50}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes)

        # Build a fresh model for each fold
        model = build_model(num_classes)

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(factor=0.5, patience=3, verbose=0),
        ]

        # Create tf.data pipelines with augmentation
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
        train_ds = (
            train_ds
            .shuffle(len(train_idx), seed=SEED)
            .batch(BATCH_SIZE)
            .map(lambda x, y_: (tf.py_function(
                    func=lambda imgs: augment_batch(imgs.numpy()),
                    inp=[x], Tout=tf.float32), y_),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=callbacks,
            verbose=1,
        )
        fold_histories.append(history.history)

        # Inference
        t0    = time.time()
        preds = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
        t1    = time.time()

        y_pred_fold = np.argmax(preds, axis=1)
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred_fold.tolist())

        acc = accuracy_score(y_val, y_pred_fold)
        pre = precision_score(y_val, y_pred_fold, average="weighted", zero_division=0)
        rec = recall_score   (y_val, y_pred_fold, average="weighted", zero_division=0)
        f1  = f1_score       (y_val, y_pred_fold, average="weighted", zero_division=0)
        lat = (t1 - t0) / len(y_val) * 1000  # ms/image

        fold_results.append({"fold": fold_idx+1, "accuracy": acc,
                              "precision": pre, "recall": rec,
                              "f1": f1, "latency_ms": lat})
        print(f"  Fold {fold_idx+1} → Acc: {acc:.4f}  F1: {f1:.4f}  Lat: {lat:.2f}ms/img")

        tf.keras.backend.clear_session()

    # ── Aggregate results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Aggregated 5-Fold Results")
    print("=" * 60)
    accs = [r["accuracy"]  for r in fold_results]
    pres = [r["precision"] for r in fold_results]
    recs = [r["recall"]    for r in fold_results]
    f1s  = [r["f1"]        for r in fold_results]
    lats = [r["latency_ms"]for r in fold_results]

    summary = {
        "n_folds":    N_FOLDS,
        "total_samples": len(X),
        "classes":    class_names,
        "fold_results": fold_results,
        "aggregate": {
            "accuracy_mean":   float(np.mean(accs)),
            "accuracy_std":    float(np.std(accs)),
            "precision_mean":  float(np.mean(pres)),
            "precision_std":   float(np.std(pres)),
            "recall_mean":     float(np.mean(recs)),
            "recall_std":      float(np.std(recs)),
            "f1_mean":         float(np.mean(f1s)),
            "f1_std":          float(np.std(f1s)),
            "latency_ms_mean": float(np.mean(lats)),
            "latency_ms_std":  float(np.std(lats)),
        }
    }

    with open(os.path.join(RESULTS_DIR, "kfold_metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=4)

    # Classification report on aggregated predictions
    report = classification_report(
        all_y_true, all_y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )
    with open(os.path.join(RESULTS_DIR, "kfold_classification_report.txt"), "w") as fh:
        fh.write(report)

    # Confusion matrix (aggregated)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Aggregated 5-Fold Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "kfold_confusion_matrix.png"), dpi=150)
    plt.close()

    # Per-class CSV
    report_dict = classification_report(
        all_y_true, all_y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0, output_dict=True
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(RESULTS_DIR, "kfold_class_metrics.csv")
    )

    # Per-fold accuracy bar chart
    plt.figure(figsize=(8, 5))
    folds = [f"Fold {r['fold']}" for r in fold_results]
    plt.bar(folds, accs, color="steelblue")
    plt.axhline(np.mean(accs), color="red", linestyle="--",
                label=f"Mean: {np.mean(accs):.4f}")
    plt.ylim(0, 1.05)
    plt.title("Per-Fold Accuracy (5-Fold CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "kfold_per_fold_accuracy.png"), dpi=150)
    plt.close()

    print(f"\n  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Precision: {np.mean(pres):.4f} ± {np.std(pres):.4f}")
    print(f"  Recall   : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"  F1-Score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Latency  : {np.mean(lats):.2f} ± {np.std(lats):.2f} ms/image")
    print("\n✅ K-Fold evaluation complete. Results saved to research_results/")


if __name__ == "__main__":
    run_kfold()
