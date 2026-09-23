import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Ensure legacy keras is used for M-series compatibility as per project
os.environ['TF_USE_LEGACY_KERAS'] = '1'

DATA_DIR = "../Data"
MODEL_PATH = "../Model/keras_model.h5"
LABELS_PATH = "../Model/labels.txt"
IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42
RESULTS_DIR = "."

def evaluate_model():
    print("Loading dataset distribution...")
    # Dataset Distribution
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    class_counts = {}
    total_images = 0
    for c in classes:
        count = len(os.listdir(os.path.join(DATA_DIR, c)))
        class_counts[c] = count
        total_images += count
    
    print(f"Total images: {total_images}")
    
    # Plot dataset distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title("Dataset Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "dataset_distribution.png"))
    plt.close()

    print("Loading validation dataset...")
    # Since there is NO independent test set, we evaluate on the validation set
    # Using the exact same seed and split as train.py to perfectly reconstruct the val set
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )
    
    class_names = val_ds.class_names
    
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    print("Running inference on validation set...")
    y_true = []
    y_pred = []
    
    start_time = time.time()
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    end_time = time.time()
    
    inference_time = end_time - start_time
    total_samples = len(y_true)
    latency_ms = (inference_time / total_samples) * 1000
    fps = 1000 / latency_ms
    
    print("Computing metrics...")
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Save classification report
    report = classification_report(y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names, zero_division=0)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        
    report_dict = classification_report(y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names, output_dict=True, zero_division=0)
    df_metrics = pd.DataFrame(report_dict).transpose()
    df_metrics.to_csv(os.path.join(RESULTS_DIR, "class_metrics.csv"))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Validation Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    
    # Save inference metrics
    metrics = {
        "dataset": {
            "total_images": total_images,
            "classes": num_classes,
            "class_counts": class_counts
        },
        "evaluation_split": "Validation (20% of dataset, NO INDEPENDENT TEST SET AVAILABLE)",
        "evaluated_samples": total_samples,
        "performance": {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_score_weighted": f1
        },
        "latency": {
            "total_inference_time_sec": inference_time,
            "avg_latency_ms_per_image": latency_ms,
            "estimated_fps_batch_inference": fps
        }
    }
    
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    num_classes = 8
    evaluate_model()
