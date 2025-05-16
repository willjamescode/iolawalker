#!/usr/bin/env python3
"""
test_cnn.py

Load test200hz.csv, preprocess into sliding‐window tensors,
load the trained CNN, and evaluate on the test set.
"""
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# ─── CONFIG ────────────────────────────────────────────────────────────
TEST_CSV     = "/home/u/footfallTraining/test200hz.csv"
WINDOW_SIZE  = 200   # must match training
STRIDE       = 50    # must match training
BATCH_SIZE   = 64
MODEL_PATH   = "best_cnn.pth"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MODEL DEFINITION (must match cnn_pipeline.py) ──────────────────────
class FootfallCNN(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),

            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ─── DATASET PREPARATION ────────────────────────────────────────────────
def prepare_test_dataset(csv_path):
    df = pd.read_csv(csv_path, sep=r"[,\t]+", engine="python")
    # 1) extract footfall timestamps
    event_ts = df.loc[df.eventType == "footfall", "timestamp"].values
    samples  = df[df.eventType == "sample"].reset_index(drop=True)

    # 2) rename & convert
    samples = samples.rename(columns={
        "accelerationX": "ax",
        "accelerationY": "ay",
        "accelerationZ": "az"
    })
    samples[["ax","ay","az"]] = samples[["ax","ay","az"]].astype(float)

    # 3) label nearest sample to each footfall
    samples["target"] = 0
    for t in event_ts:
        idx = (samples["timestamp"] - t).abs().idxmin()
        samples.at[idx, "target"] = 1

    # 4) magnitude channel
    acc = samples[["ax","ay","az"]].values
    samples["mag"] = np.linalg.norm(acc, axis=1)

    # 5) features & labels
    feats  = samples[["ax","ay","az","mag"]].values.astype(np.float32)
    labels = samples["target"].values.astype(np.int64)

    # 6) scale (fit on test data)
    scaler = StandardScaler().fit(feats)
    feats  = scaler.transform(feats).astype(np.float32)

    # 7) sliding windows
    X_windows, y_windows = [], []
    L = WINDOW_SIZE
    for start in range(0, len(feats) - L + 1, STRIDE):
        win = feats[start:start+L]            # (L,4)
        X_windows.append(win.T)               # (4, L)
        y_windows.append(labels[start+L-1])   # label = last sample

    X = torch.from_numpy(np.stack(X_windows))  # (N,4,L)
    y = np.array(y_windows)                    # (N,)
    return X, y

# ─── EVALUATION ────────────────────────────────────────────────────────
def evaluate():
    # Prepare data
    print("Preparing test dataset…")
    X_test, y_test = prepare_test_dataset(TEST_CSV)
    test_ds = TensorDataset(X_test, torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = FootfallCNN().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Run inference
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs > 0.5).astype(int)
            all_preds.extend(preds.tolist())

    y_true = y_test
    y_pred = np.array(all_preds)

    # Metrics
    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    # Report
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

if __name__ == "__main__":
    evaluate()
