#!/usr/bin/env python3
"""
cnn_pipeline.py

1) Prepare & cache sliding‐window dataset for 1D‐CNN
2) Define CNN model
3) Train & validate, saving best model
"""
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────
CSV_PATH     = "/home/u/footfallTraining/train200hz.csv"
CACHE_FILE   = "cnn_data.pt"
MODEL_OUT    = "best_cnn.pth"
WINDOW_SIZE  = 200   # 1s @200Hz
STRIDE       = 50    # 75% overlap
BATCH_SIZE   = 64
LR           = 1e-3
EPOCHS       = 50
VAL_RATIO    = 0.2
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 1) DATA PREP ─────────────────────────────────────────────────────────
def build_and_cache():
    df = pd.read_csv(CSV_PATH, sep=r"[,\t]+", engine="python")
    # extract footfall timestamps
    event_ts = df.loc[df.eventType=="footfall", "timestamp"].values
    samples  = df[df.eventType=="sample"].reset_index(drop=True)

    # rename and convert
    samples = samples.rename(columns={
        "accelerationX":"ax",
        "accelerationY":"ay",
        "accelerationZ":"az"
    })
    samples[["ax","ay","az"]] = samples[["ax","ay","az"]].astype(float)

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

    # scale
    scaler = StandardScaler().fit(feats)
    feats  = scaler.transform(feats).astype(np.float32)

    # sliding windows
    X_windows, y_windows = [], []
    L = WINDOW_SIZE
    for start in range(0, len(feats) - L + 1, STRIDE):
        win = feats[start:start+L]      # (L,4)
        X_windows.append(win.T)         # (4, L)
        y_windows.append(labels[start+L-1])  # last sample’s label

    X = torch.from_numpy(np.stack(X_windows))  # (N,4,L)
    y = torch.from_numpy(np.array(y_windows))  # (N,)

    # cache
    torch.save({"X": X, "y": y}, CACHE_FILE)
    print(f"  ↳ built and cached dataset → {CACHE_FILE}")
    return X, y

def load_or_build():
    if os.path.exists(CACHE_FILE):
        data = torch.load(CACHE_FILE)
        print(f"  ↳ loaded cached dataset ({CACHE_FILE})")
        return data["X"], data["y"]
    else:
        return build_and_cache()

# ─── 2) MODEL ────────────────────────────────────────────────────────────
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

            nn.Flatten(),              # (batch, 64)
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)           # output logits
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ─── 3) TRAIN/VALIDATE ────────────────────────────────────────────────────
def train():
    X, y = load_or_build()

    # split
    N = len(y)
    n_val = int(N * VAL_RATIO)
    n_trn = N - n_val
    full_ds = TensorDataset(X, y)
    trn_ds, val_ds = random_split(
        full_ds,
        [n_trn, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # model + criterion + optim
    model     = FootfallCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        # train
        model.train()
        total_loss = correct = total = 0
        for xb, yb in tqdm(trn_loader, desc=f"Epoch {epoch}/{EPOCHS} ▶ train"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds       = (torch.sigmoid(logits) > 0.5).long()
            correct    += (preds == yb.long()).sum().item()
            total      += xb.size(0)

        train_acc = correct / total

        # validate
        model.eval()
        vloss = correct = total = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} ▷ val  "):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
                logits  = model(xb)
                vloss  += criterion(logits, yb).item() * xb.size(0)
                preds   = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == yb.long()).sum().item()
                total   += xb.size(0)

        val_acc = correct / total
        print(
            f"Epoch {epoch:02d}: "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f" → Saved new best model (val_acc={best_acc:.4f})")

        scheduler.step(val_acc)

    print(f"Training complete. Best val_acc = {best_acc:.4f}")

if __name__ == "__main__":
    train()
