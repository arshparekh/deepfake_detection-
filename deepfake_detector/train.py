"""
train.py — Standalone training script for the DeepTrace CNN
Usage:
    python train.py --data_dir /path/to/ff++  --epochs 30 --output model.weights.h5

FaceForensics++ directory structure expected:
    data_dir/
        real/    *.jpg | *.png
        fake/    *.jpg | *.png

If --data_dir is omitted the script generates a synthetic FaceForensics++
simulation (1 200 samples) so you can verify the pipeline immediately.
"""

import argparse
import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json

#CLI
parser = argparse.ArgumentParser(description="Train DeepTrace deepfake detector")
parser.add_argument("--data_dir", type=str, default=None,
                    help="Path to FaceForensics++ dataset (real/ and fake/ subdirs)")
parser.add_argument("--epochs",   type=int,   default=30)
parser.add_argument("--batch",    type=int,   default=32)
parser.add_argument("--img_size", type=int,   default=224)
parser.add_argument("--output",   type=str,   default="deepfake_model.weights.h5")
parser.add_argument("--plot",     action="store_true", help="Save training plot")
args = parser.parse_args()

IMG_SIZE = (args.img_size, args.img_size)

#MODEL
def build_model(input_shape):
    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling=None
    )
    freeze_until = int(len(base.layers) * 0.80)
    for l in base.layers[:freeze_until]: l.trainable = False
    for l in base.layers[freeze_until:]:  l.trainable = True

    inp = keras.Input(shape=input_shape)
    x = keras.layers.RandomFlip("horizontal")(inp)
    x = keras.layers.RandomRotation(0.05)(x)
    x = keras.layers.RandomZoom(0.05)(x)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(512, activation="relu",
                            kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256, activation="relu",
                            kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out, name="DeepfakeDetector")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model


#DATA LOADING
def load_ff_dataset(data_dir, img_size):
    """Load real/fake images from FaceForensics++ directory layout."""
    X, y = [], []
    for label, folder in [(0, "real"), (1, "fake")]:
        pattern = os.path.join(data_dir, folder, "*.[jJpP][pPnN][gG]*")
        paths = glob.glob(pattern)
        if not paths:
            raise FileNotFoundError(f"No images found in {os.path.join(data_dir, folder)}")
        print(f"  Loading {len(paths)} {folder} images…")
        for p in paths:
            img = cv2.imread(p)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(img.astype(np.float32))
            y.append(label)
    return np.array(X), np.array(y)


def generate_synthetic_dataset(n=1200, img_size=224):
    """Synthetic FaceForensics++ simulation for testing."""
    print("  Generating synthetic FF++ data…")
    X, y = [], []
    np.random.seed(42)
    for i in range(n):
        lbl = i % 2
        if lbl == 0:  # real
            base = np.random.randint(80, 200, (img_size, img_size, 3), dtype=np.uint8)
            base[:,:,0] = np.clip(base[:,:,0]+30, 0, 255)
            base[:,:,2] = np.clip(base[:,:,2]-20, 0, 255)
            noise = np.random.normal(0, 8, base.shape).astype(np.int16)
            img = np.clip(base.astype(np.int16)+noise, 0, 255).astype(np.uint8)
            img = cv2.GaussianBlur(img, (3,3), 0.5)
        else:           # fake
            base = np.random.randint(60, 220, (img_size, img_size, 3), dtype=np.uint8)
            base[:,:,1] = np.clip(base[:,:,1]+25, 0, 255)
            noise = np.random.normal(0, 18, base.shape).astype(np.int16)
            img = np.clip(base.astype(np.int16)+noise, 0, 255).astype(np.uint8)
            for r in range(0, img_size, 8):
                for c in range(0, img_size, 8):
                    if (r//8+c//8) % 2 == 0:
                        img[r:r+2,c:c+2] = np.clip(img[r:r+2,c:c+2]+15, 0, 255)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1].astype(np.int16)+20, 0, 255)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        X.append(img.astype(np.float32))
        y.append(lbl)
    return np.array(X), np.array(y)


#MAIN
if __name__ == "__main__":
    print("\n=== DeepTrace Training Pipeline ===")

    if args.data_dir:
        print(f"Loading FaceForensics++ from: {args.data_dir}")
        X, y = load_ff_dataset(args.data_dir, IMG_SIZE)
    else:
        print("No --data_dir provided; using synthetic FF++ simulation.")
        X, y = generate_synthetic_dataset(n=1200, img_size=args.img_size)

    print(f"Dataset: {len(X)} samples | Real: {(y==0).sum()} | Fake: {(y==1).sum()}")

    # Shuffle & split 85/15
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.85)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    print(f"Train: {len(X_tr)} | Val: {len(X_val)}")

    model = build_model((*IMG_SIZE, 3))
    print(f"Parameters: {model.count_params():,}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=6, restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            "best_checkpoint.weights.h5", monitor="val_auc",
            save_best_only=True, save_weights_only=True, mode="max"
        ),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        class_weight={0: 1.0, 1: 1.0},
        verbose=1,
    )

    model.save_weights(args.output)
    print(f"\n✓ Weights saved → {args.output}")

    # Summary
    best_acc = max(history.history.get("val_accuracy", [0]))
    best_auc = max(history.history.get("val_auc", [0]))
    print(f"Best val accuracy : {best_acc:.4f} ({best_acc*100:.1f}%)")
    print(f"Best val AUC      : {best_auc:.4f}")

    metrics_out = {
        "best_val_accuracy": float(best_acc),
        "best_val_auc":      float(best_auc),
        "epochs_trained":    len(history.history["loss"]),
    }
    with open("training_metrics.json", "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    print("Metrics saved → training_metrics.json")

    # Optional plot
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history["loss"],     label="Train Loss")
        ax1.plot(history.history["val_loss"], label="Val Loss")
        ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=.3)
        ax2.plot(history.history["accuracy"],     label="Train Acc")
        ax2.plot(history.history["val_accuracy"], label="Val Acc")
        ax2.axhline(0.95, color="red", linestyle="--", label="95 % target")
        ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig("training_plot.png", dpi=150)
        print("Plot saved → training_plot.png")
