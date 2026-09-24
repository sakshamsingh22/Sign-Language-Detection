import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf
import os
import threading
import time

# ── 1. Load Both Models ───────────────────────────────────────────────────────
print("⏳ Loading Custom Word Model (8 Classes)...")
model_custom = tf.keras.models.load_model("Model/keras_model.h5")
with open("Model/labels.txt", "r") as f:
    labels_custom = [line.strip().split(" ", 1)[1] for line in f if len(line.strip().split(" ", 1)) == 2]

print("⏳ Loading Kaggle Alphabet Model (35 Classes)...")
model_kaggle = tf.keras.models.load_model("Model/kaggle_model.h5")
with open("Model/kaggle_labels.txt", "r") as f:
    labels_kaggle = [line.strip().split(" ", 1)[1] for line in f if len(line.strip().split(" ", 1)) == 2]

print("✅ Both Models Loaded Successfully!")

# ── 2. Initialize Camera and Detectors ───────────────────────────────────────
cap = cv2.VideoCapture(0)
detector_1hand = HandDetector(maxHands=1)
detector_2hands = HandDetector(maxHands=2)

offset = 20
imgSize = 300
kaggleSize = 224

# State Variables
current_mode = "WORDS" # Can be "WORDS" or "ALPHABETS"
counter = 0
last_spoken_word = ""
last_speech_time = 0
debounce_frames = 15
cooldown_seconds = 3

def speak(text):
    os.system(f"say '{text}'")

print("\n🚀 Starting Combined Live Recognition!")
print("👉 Press 'SPACE' to switch between Words and Alphabets mode.")
print("👉 Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break
        
    imgOutput = img.copy()
    
    # UI Mode Indicator
    cv2.putText(imgOutput, f"MODE: {current_mode} (Press SPACE to switch)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0) if current_mode == "WORDS" else (0, 0, 255), 2)

    detected_label = ""
    confidence = 0.0
    x, y = 0, 0

    if current_mode == "WORDS":
        hands, img = detector_1hand.findHands(img, draw=False)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            if imgCrop.size != 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    
                    if imgResize.shape[1] + wGap <= imgSize:
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    
                    if imgResize.shape[0] + hGap <= imgSize:
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                img_normalized = cv2.resize(imgWhite, (224, 224)) / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                predictions = model_custom.predict(img_batch, verbose=0)[0]
                class_index = np.argmax(predictions)
                confidence = predictions[class_index]
                detected_label = labels_custom[class_index]

                cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)
                cv2.putText(imgOutput, f"{detected_label} {confidence*100:.0f}%",
                            (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 2)
                cv2.imshow("Crop", imgWhite)

    elif current_mode == "ALPHABETS":
        hands, img = detector_2hands.findHands(img, draw=False)
        if hands:
            x_min = min(hand['bbox'][0] for hand in hands)
            y_min = min(hand['bbox'][1] for hand in hands)
            x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands)
            y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands)
            x, y = x_min, y_min
            
            y1, y2 = max(0, y_min - offset), min(img.shape[0], y_max + offset)
            x1, x2 = max(0, x_min - offset), min(img.shape[1], x_max + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:
                imgResized = cv2.resize(imgCrop, (kaggleSize, kaggleSize))
                img_normalized = imgResized / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                predictions = model_kaggle.predict(img_batch, verbose=0)[0]
                class_index = np.argmax(predictions)
                confidence = predictions[class_index]
                detected_label = labels_kaggle[class_index]
                
                cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(imgOutput, f"{detected_label} {confidence*100:.0f}%",
                            (x_min, y_min-30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 2)
                cv2.imshow("Crop", imgCrop)

    # ── 3. Debounce and TTS ──────────────────────────────────────────────────
    current_time = time.time()
    if detected_label and confidence > 0.6:
        if detected_label == last_spoken_word:
            counter += 1
        else:
            counter = 1
            last_spoken_word = detected_label
            
        if counter >= debounce_frames and (current_time - last_speech_time) > cooldown_seconds:
            threading.Thread(target=speak, args=(detected_label,), daemon=True).start()
            last_speech_time = current_time
            counter = 0
    else:
        counter = 0

    cv2.imshow('Combined ISL System', imgOutput)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '): # SPACE bar
        current_mode = "ALPHABETS" if current_mode == "WORDS" else "WORDS"
        counter = 0
        last_spoken_word = ""

cap.release()
cv2.destroyAllWindows()
