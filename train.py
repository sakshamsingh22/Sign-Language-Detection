import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "Data"
MODEL_DIR  = "Model"
IMG_SIZE   = 224          # MobileNetV2 native input size
BATCH_SIZE = 16
EPOCHS     = 30

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data generators ─────────────────────────────────────────────────────────
import random

SEED = 42

# ── Data loading with proper random split ────────────────────────────────────
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\n✅ Classes found ({num_classes}): {class_names}")

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Augmentation layer (applied only during training)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Performance tuning
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)


# ── Save labels.txt ──────────────────────────────────────────────────────────
labels_path = os.path.join(MODEL_DIR, "labels.txt")
with open(labels_path, "w") as f:
    for idx, name in enumerate(class_names):
        f.write(f"{idx} {name}\n")
print(f"✅ Labels saved to {labels_path}")

# ── Model: MobileNetV2 fine-tune ─────────────────────────────────────────────
base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False   # freeze base first

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
model.summary()

# ── Phase 1: train head only ─────────────────────────────────────────────────
print("\n🚀 Phase 1 — training classification head …")
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, "keras_model.h5"),
                    save_best_only=True, verbose=1),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ── Phase 2: fine-tune top layers of base ────────────────────────────────────
print("\n🚀 Phase 2 — fine-tuning top layers of MobileNetV2 …")
base.trainable = True
# Freeze all but the last 30 layers
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(1e-4),   # lower lr for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
)

print("\n✅ Training complete! Model saved to Model/keras_model.h5")
