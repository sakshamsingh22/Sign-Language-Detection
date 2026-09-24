import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
import os
import threading
import time

# ── 1. Load the Kaggle Model and Labels ───────────────────────────────────────
model_path = os.path.join("Model", "kaggle_model.h5")
labels_path = os.path.join("Model", "kaggle_labels.txt")

model = tf.keras.models.load_model(model_path)
print("✅ Kaggle Model Loaded Successfully")

labels = []
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                labels.append(parts[1])
else:
    print(f"❌ Error: {labels_path} not found.")
    exit()

print(f"✅ Loaded {len(labels)} labels: {labels}")

# ── 2. Initialize Camera and Hand Detector ───────────────────────────────────
cap = cv2.VideoCapture(0)
# Change to maxHands=2 for Indian Sign Language which uses both hands
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 224

# ── 3. Temporal Debounce & TTS Logic ─────────────────────────────────────────
counter = 0
last_spoken_word = ""
last_speech_time = 0
debounce_frames = 15
cooldown_seconds = 3

def speak(text):
    os.system(f"say '{text}'")

print("\n🚀 Starting Live Recognition (Kaggle Dataset) ... Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break
        
    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=False)
    
    detected_label = ""
    confidence = 0.0

    if hands:
        # Merge bounding boxes if multiple hands are detected
        x_min = min(hand['bbox'][0] for hand in hands)
        y_min = min(hand['bbox'][1] for hand in hands)
        x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands)
        y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands)
        
        # Ensure bounding box is within frame boundaries
        y1, y2 = max(0, y_min - offset), min(img.shape[0], y_max + offset)
        x1, x2 = max(0, x_min - offset), min(img.shape[1], x_max + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            # Resize directly to 224x224 (matching our train_kaggle.py logic)
            imgResized = cv2.resize(imgCrop, (imgSize, imgSize))
            
            # Predict
            img_normalized = imgResized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            predictions = model.predict(img_batch, verbose=0)[0]
            class_index = np.argmax(predictions)
            confidence = predictions[class_index]
            detected_label = labels[class_index]
            
            # Draw UI
            cv2.putText(imgOutput, f"{detected_label} {confidence*100:.0f}%",
                        (x_min, y_min-30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 2)
            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0,255,0), 4)
            cv2.imshow('ImageCrop', imgCrop)

    # ── 4. Debounce and Speak ────────────────────────────────────────────────
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

    cv2.imshow('Kaggle ISL Recognition', imgOutput)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
