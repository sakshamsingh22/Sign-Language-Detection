import cv2
import os
import glob
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from sklearn.metrics import accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

def run_ablation():
    print("Running Ablation E5: Evaluating WITHOUT aspect-ratio preserving White Canvas...")
    detector = HandDetector(maxHands=1)
    classifier = Classifier("../Model/keras_model.h5", "../Model/labels.txt")
    offset = 20
    imgSize = 224 # direct resize to 224
    
    classes = sorted([d for d in os.listdir("../Data") if os.path.isdir(os.path.join("../Data", d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    y_true = []
    y_pred = []
    
    start_time = time.time()
    valid_count = 0
    total_images = 0
    
    for cls in classes:
        cls_dir = os.path.join("../Data", cls)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            total_images += 1
            fpath = os.path.join(cls_dir, fname)
            img = cv2.imread(fpath)
            if img is None: continue
            
            hands, img = detector.findHands(img, draw=False)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                imgH, imgW = img.shape[:2]
                x1 = max(0, x - offset)
                y1 = max(0, y - offset)
                x2 = min(imgW, x + w + offset)
                y2 = min(imgH, y + h + offset)
                imgCrop = img[y1:y2, x1:x2]
                
                if imgCrop.size == 0: continue
                
                # ABLATION: Directly resize to 224x224, squishing the aspect ratio
                # NO white canvas used!
                imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
                
                prediction, index = classifier.getPrediction(imgResize, draw=False)
                y_true.append(class_to_idx[cls])
                y_pred.append(index)
                valid_count += 1

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAblation Results (Without Canvas Normalization):")
    print(f"Accuracy: {acc:.4f} (Valid frames: {valid_count})")
    
    with open("ablation_metrics.txt", "w") as f:
        f.write(f"Ablation (No Canvas Normalization) Accuracy: {acc:.4f}\n")

if __name__ == '__main__':
    run_ablation()
