import cv2
import os
import glob
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

def run_svm_baseline():
    print("Extracting MediaPipe landmarks for SVM Baseline...")
    detector = HandDetector(maxHands=1)
    
    classes = sorted([d for d in os.listdir("../Data") if os.path.isdir(os.path.join("../Data", d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    X = []
    y = []
    
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
            
            hands, _ = detector.findHands(img, draw=False)
            if hands:
                hand = hands[0]
                lmList = hand['lmList']
                # flatten 21*3 to 63
                features = []
                for lm in lmList:
                    features.extend(lm)
                X.append(features)
                y.append(class_to_idx[cls])
                valid_count += 1

    extraction_time = time.time() - start_time
    print(f"Extracted landmarks for {valid_count}/{total_images} images in {extraction_time:.2f}s")
    
    X = np.array(X)
    y = np.array(y)
    
    print("\nRunning 5-Fold CV on SVM (RBF kernel)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accs = []
    lats = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        clf = SVC(kernel='rbf', probability=False)
        clf.fit(X_train, y_train)
        
        t0 = time.time()
        preds = clf.predict(X_val)
        t1 = time.time()
        
        acc = accuracy_score(y_val, preds)
        lat = (t1 - t0) / len(X_val) * 1000
        accs.append(acc)
        lats.append(lat)
    
    print(f"SVM Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"SVM Inference Latency (clf.predict only): {np.mean(lats):.2f} ms/image")
    
    with open("baseline_svm_metrics.txt", "w") as f:
        f.write(f"SVM Baseline 5-Fold CV:\n")
        f.write(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
        f.write(f"Latency: {np.mean(lats):.2f} ms/image\n")
        f.write(f"Valid images used: {valid_count}\n")

if __name__ == '__main__':
    run_svm_baseline()
