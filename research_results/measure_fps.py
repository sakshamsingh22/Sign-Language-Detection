import cv2
import time
import os
import glob
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

def run_benchmark():
    print("Loading models for FPS benchmark...")
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
    offset = 20
    imgSize = 300
    
    image_paths = glob.glob("Data/*/*.jpg") + glob.glob("Data/*/*.png")
    if len(image_paths) > 1000:
        np.random.seed(42)
        image_paths = np.random.choice(image_paths, 1000, replace=False)
    
    print(f"Running end-to-end pipeline on {len(image_paths)} images...")
    
    valid_frames = 0
    start_time = time.time()
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None: continue
        
        hands, img = detector.findHands(img, draw=False)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgH, imgW = img.shape[:2]
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(imgW, x + w + offset)
            y2 = min(imgH, y + h + offset)
            imgCrop = img[y1:y2, x1:x2]
            
            if imgCrop.size == 0: continue
            
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = min(math.ceil(k * w), imgSize)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = min(math.ceil(k * h), imgSize)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
                
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            valid_frames += 1

    end_time = time.time()
    total_time = end_time - start_time
    fps = valid_frames / total_time
    print(f"Processed {valid_frames} valid frames in {total_time:.2f} seconds.")
    print(f"End-to-End FPS: {fps:.2f} FPS")

if __name__ == '__main__':
    run_benchmark()
