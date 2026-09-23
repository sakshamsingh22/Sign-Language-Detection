import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import subprocess
import time
import threading

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5" , "Model/labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = []
with open("Model/labels.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            labels.append(line.split(" ", 1)[1])
print(f"Loaded {len(labels)} labels: {labels}")

# ── TTS state ────────────────────────────────────────────────────────────────
STABLE_FRAMES = 15       # sign must be held this many frames before speaking
COOLDOWN_SEC  = 3        # seconds before the same sign can be spoken again

prev_index      = -1
stable_count    = 0
last_spoken     = ""
last_speak_time = 0

def speak(text):
    """Speak text in a background thread using macOS 'say' command."""
    def _say():
        subprocess.run(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    threading.Thread(target=_say, daemon=True).start()


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgH, imgW = img.shape[:2]
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgW, x + w + offset)
        y2 = min(imgH, y + h + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = min(math.ceil(k * w), imgSize)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = min(math.ceil(k * h), imgSize)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        confidence = max(prediction)

        # ── Stability check & TTS ────────────────────────────────────────
        if index == prev_index:
            stable_count += 1
        else:
            stable_count = 1
            prev_index = index

        detected_label = labels[index]

        if (stable_count == STABLE_FRAMES
                and (detected_label != last_spoken
                     or time.time() - last_speak_time > COOLDOWN_SEC)):
            speak(detected_label)
            last_spoken = detected_label
            last_speak_time = time.time()

        # ── Draw UI ──────────────────────────────────────────────────────
        cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x-offset+400, y-offset+60-50),(0,255,0),cv2.FILLED)
        cv2.putText(imgOutput, f"{detected_label} {confidence*100:.0f}%",
                    (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
    