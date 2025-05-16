#!/usr/bin/env python3
"""
test_cnn_tolerance.py

Load test200hz.csv, preprocess into sliding‐window tensors exactly as in cnn_pipeline.py,
load the trained CNN, and evaluate event‐level detection accuracy allowing
a ±TOLERANCE‐sample slack around each true footfall.
"""
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ─── CONFIG ────────────────────────────────────────────────────────────
TEST_CSV    = "/home/u/footfallTraining/test200hz.csv"
WINDOW_SIZE = 200     # same as training
STRIDE      = 50      # same as training
BATCH_SIZE  = 64
MODEL_PATH  = "best_cnn.pth"
TOLERANCE   = 10      # ±10 samples ≃ ±50 ms at 200 Hz
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MODEL DEFINITION ───────────────────────────────────────────────────
class FootfallCNN(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveMaxPool1d(1),

            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [batch, channels=4, length=WINDOW_SIZE]
        return self.net(x).squeeze(-1)

# ─── DATA PREPARATION ───────────────────────────────────────────────────
def prepare_test_data(csv_path):
    # read raw CSV
    df = pd.read_csv(csv_path, sep=r"[,\t]+", engine="python")
    # extract footfall timestamps
    event_ts = df.loc[df.eventType == "footfall", "timestamp"].values
    # keep only samples
    samples  = df[df.eventType == "sample"].reset_index(drop=True)

    # rename accelerometer cols
    samples = samples.rename(columns={
        "accelerationX": "ax",
        "accelerationY": "ay",
        "accelerationZ": "az"
    })
    for c in ("ax", "ay", "az"):
        samples[c] = samples[c].astype(float)

    # label nearest sample to each footfall
    samples["target"] = 0
    for t in event_ts:
        idx = (samples["timestamp"] - t).abs().idxmin()
        samples.at[idx, "target"] = 1

    # magnitude channel
    acc = samples[["ax","ay","az"]].values
    samples["mag"] = np.linalg.norm(acc, axis=1)

    # features + labels
    feats  = samples[["ax","ay","az","mag"]].values.astype(np.float32)
    labels = samples["target"].values.astype(np.int64)

    # scale features (same approach as training: fit on test set)
    scaler = StandardScaler().fit(feats)
    feats  = scaler.transform(feats).astype(np.float32)

    # sliding windows & track window end indices
    X_windows, y_windows, end_idxs = [], [], []
    L, S = WINDOW_SIZE, STRIDE
    N = len(feats)
    for start in range(0, N - L + 1, S):
        end_idx = start + L - 1
        window  = feats[start:start+L].T       # shape (4, L)
        X_windows.append(window)
        y_windows.append(labels[end_idx])
        end_idxs.append(end_idx)

    X = torch.from_numpy(np.stack(X_windows))   # (num_windows, 4, L)
    y = np.array(y_windows)                     # (num_windows,)
    return X, y, np.array(end_idxs)

# ─── EVALUATION ────────────────────────────────────────────────────────
def evaluate():
    print("Preparing test data…")
    X, y_true, end_idxs = prepare_test_data(TEST_CSV)

    # DataLoader
    ds     = TensorDataset(X, torch.from_numpy(y_true))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    model = FootfallCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Run inference window‐by‐window
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb    = xb.to(DEVICE)
            logits= model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend((probs > 0.5).astype(int).tolist())
    y_pred = np.array(preds)

    # Identify true and predicted event sample indices
    true_events = end_idxs[y_true == 1]
    pred_events = set(end_idxs[y_pred == 1])

    # Match with tolerance
    tp = 0
    matched = set()
    for te in true_events:
        # check for any prediction within ±TOLERANCE
        window = range(te - TOLERANCE, te + TOLERANCE + 1)
        hits = [pe for pe in window if pe in pred_events]
        if hits:
            tp += 1
            # choose closest to avoid double‐counting
            matched.add(min(hits, key=lambda pe: abs(pe - te)))

    fn = len(true_events) - tp
    fp = len(pred_events - matched)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nEvent‐level detection (±{TOLERANCE} samples):")
    print(f"  True Positives : {tp}")
    print(f"  False Negatives: {fn}")
    print(f"  False Positives: {fp}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1 Score       : {f1:.4f}")

if __name__ == "__main__":
    evaluate()
