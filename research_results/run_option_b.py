import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import json
import time

DATA_DIR = "../Data"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15 # Shorter epochs for quick ablation

# Full Dataset loader for final eval
full_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
full_ds = full_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

def get_train_val(seed=42):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=seed,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=seed,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False
    )
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def build_model(weights="imagenet", freeze_base=True, use_flip=True):
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=weights)
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    if use_flip:
        x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.layers.RandomTranslation(0.1, 0.1)(x)
    
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(8, activation="softmax")(x)
    return models.Model(inputs, outputs)

results = {}

# 1. Scratch CNN (Full Dataset Eval for speed, or we can just train on 80/20 and eval on full)
print("=== Training Scratch CNN ===")
train_ds, val_ds = get_train_val()
model_scratch = build_model(weights=None, freeze_base=False, use_flip=True)
model_scratch.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model_scratch.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0)
loss, acc = model_scratch.evaluate(full_ds, verbose=0)
results["scratch_cnn_acc"] = acc
print(f"Scratch CNN Acc: {acc:.4f}")

# 2. Phase 1 Only Ablation (Pretrained, but base frozen, normal aug)
print("=== Training Phase 1 Only ===")
model_p1 = build_model(weights="imagenet", freeze_base=True, use_flip=True)
model_p1.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model_p1.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0)
loss, acc = model_p1.evaluate(full_ds, verbose=0)
results["phase1_only_acc"] = acc
print(f"Phase 1 Only Acc: {acc:.4f}")

# 3. No Flip Ablation (Phase 1 + 2, but no flip)
print("=== Training No Flip ===")
model_noflip = build_model(weights="imagenet", freeze_base=True, use_flip=False)
model_noflip.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model_noflip.fit(train_ds, validation_data=val_ds, epochs=10, verbose=0)
# Phase 2
model_noflip.layers[4].trainable = True # The MobileNetV2 base
model_noflip.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
model_noflip.fit(train_ds, validation_data=val_ds, epochs=10, verbose=0)
loss, acc = model_noflip.evaluate(full_ds, verbose=0)
results["no_flip_acc"] = acc
print(f"No Flip Acc: {acc:.4f}")

with open("ablation_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("Done!")
