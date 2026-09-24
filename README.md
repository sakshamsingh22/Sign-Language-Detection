# Real-Time Indian Sign Language (ISL) Detection System (Dual-Mode Architecture)

## 1. Problem Statement
**What real-world problem is being solved?**
Communication barriers exist between the deaf and hard-of-hearing community and those who do not understand sign language. 

**What limitations exist?**
Many computer vision models for gesture recognition struggle with real-time performance on consumer-grade hardware or suffer from accuracy degradation due to lighting changes, background noise, or hand movement blurring.

**Why does this project need to exist?**
This project provides a lightweight, real-time ISL translation tool that runs entirely locally on consumer hardware. By translating gestures to text and speech instantly, it facilitates seamless two-way communication.

## 2. 🔥 New: Dual-Mode Hybrid Architecture
This project features a highly advanced **Dual-Mode UI**, allowing it to scale from custom vocabulary words to the entire ISL alphabet seamlessly in real-time.

- **Words Mode (8 Classes):** Uses a custom-built dataset for dynamic conversational words (e.g., *Hello, Please, Thank you*). It utilizes a novel **Aspect-Ratio-Preserving White Canvas Normalization** technique to completely eliminate background noise, achieving 93.75% K-Fold cross-validation accuracy.
- **Alphabets Mode (35 Classes):** Integrates a massive **42,000+ image open-source Kaggle dataset** (`prathumarikeri/indian-sign-language-isl`) for complete A-Z and 1-9 recognition. It automatically scales to detect both hands (merged bounding boxes) to correctly interpret two-handed ISL alphabets.

**You can switch between these two modes instantly in real-time by pressing the `SPACEBAR` during live inference!**

## 3. Proposed Solution
This system detects hands in a video feed, processes the cropped hand images to a standardized format, and passes them through a deep learning model to predict the corresponding ISL sign. The prediction is then converted into audio feedback.

## 4. System Architecture
- **Webcam Feed**: Standard RGB input via OpenCV.
- **cvzone HandDetector**: Wraps Google MediaPipe to locate hand landmarks and generate a bounding box (dynamically switches between 1-hand or 2-hands based on the selected mode).
- **Aspect Ratio Normalization**: The cropped hand is resized to fit a 300x300 canvas while maintaining its aspect ratio. 
- **MobileNetV2 Classifiers**: Fine-tuned MobileNetV2 models classify the processed image into either 8 classes (Words) or 35 classes (Alphabets).
- **Stability Tracking**: Ensures a gesture is held for a set duration (15 frames) before triggering audio to prevent erratic TTS spam.
- **Background TTS Thread**: Executes the native OS `say` command asynchronously to prevent video lag.

## 5. Methodology & AI / ML Details
- **Model Architecture**: MobileNetV2 (base) + GlobalAveragePooling2D + Dropout(0.3) + Dense + Dropout(0.2) + Softmax Output
- **Model Name**: MobileNetV2 Transfer Learning
- **Datasets**: 
  1. Custom captured dataset (~1,700 images)
  2. Kaggle ISL Dataset (~42,000 images)
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: `tf.keras.optimizers.legacy.Adam` (Optimized for Apple Silicon)
- **Train/Validation Split**: 80% / 20%
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint 

## 6. Real-Time Logic: Debounced Text-to-Speech (TTS)
- **Purpose**: Prevent audio spam and stuttering when model confidence fluctuates rapidly between frames.
- **Process**:
  1. Keep track of consecutive identical predictions.
  2. If a prediction is held stable for **15 consecutive frames** (and a 3-second cooldown has passed):
     - Spawn a background (Daemon) thread: `subprocess.run(["say", text])`
- **Output**: Audio synthesis without blocking the OpenCV camera feed, maintaining 37+ FPS.

## 7. Reproducibility
To run this project:
```bash
# 1. Clone the repository and navigate to the folder
git clone <repository_url>
cd Sign-Language-detection-main

# 2. Setup Virtual Environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install opencv-python cvzone mediapipe==0.10.14 tf-keras scipy kagglehub

# 4. Train the Custom Word Model (8 Classes)
TF_USE_LEGACY_KERAS=1 python train.py

# 5. Train the Kaggle Alphabet Model (35 Classes, 42,000 images)
TF_USE_LEGACY_KERAS=1 python train_kaggle.py

# 6. Run the Combined Real-Time Inference (Dual-Mode)
TF_USE_LEGACY_KERAS=1 python test_combined.py
# (Press SPACEBAR to switch between Words and Alphabets!)
```

## 8. Project Structure
```text
Sign-Language-detection-main/
├── Data/                 # Custom dataset categorized by class folders (8 words)
├── Model/                # Generated model artifacts
│   ├── keras_model.h5    # Custom Word Model weights
│   ├── labels.txt        # Custom Word Labels
│   ├── kaggle_model.h5   # Kaggle Alphabet Model weights
│   └── kaggle_labels.txt # Kaggle Alphabet Labels (35 classes)
├── datacollection.py     # Script for generating custom training data via webcam
├── train.py              # Script for training the Custom Word Model
├── train_kaggle.py       # Script for training the massive Kaggle dataset
├── test.py               # Legacy inference script (Words only)
├── test_kaggle.py        # Legacy inference script (Alphabets only)
├── test_combined.py      # 🔥 MAIN INFERENCE SCRIPT (Dual-Mode UI)
├── QA_Presentation_Guide.md # Comprehensive Viva/Interview Q&A
└── README.md             # This documentation
```
