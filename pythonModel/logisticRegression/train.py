#!/usr/bin/env python3
"""
train_logistic.py

Train a logistic regression (linear) model on shoe-mounted accelerometer time-series data,
using the same data-loading and training loop style as the LSTM code.
"""
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────
CSV_PATH     = "/home/u/footfallTraining/newtrain2.csv"
SCALER_OUT   = "logistic_scaler.pkl"
MODEL_OUT    = "best_footfall_logistic.pt"
WINDOW_SIZE  = 50
BATCH_SIZE   = 64
LR           = 1e-3
EPOCHS       = 100
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42
CLIP_NORM    = 5.0

# ─── REPRODUCIBILITY ────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ─── 1) LOAD & LABEL ───────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, sep=r"[,\t]+", engine="python", header=0)
df = df.rename(columns={
    "deviceAddress":"id",
    "accelerationX":"ax",
    "accelerationY":"ay",
    "accelerationZ":"az"
})

# ground-truth timestamps
event_ts = df.loc[df["id"]=="FOOTFALL_EVENT", "timestamp"].values
sensor   = df[df["id"]!="FOOTFALL_EVENT"].copy()
sensor["target"] = 0
for t in event_ts:
    idx = (sensor["timestamp"] - t).abs().idxmin()
    sensor.at[idx, "target"] = 1

# add magnitude channel
acc_arr = sensor[["ax","ay","az"]].values
sensor["mag"] = np.linalg.norm(acc_arr, axis=1)

# ─── 2) SCALE & SAVE ───────────────────────────────────────────────────
features_df = sensor[["ax","ay","az","mag"]]
scaler = StandardScaler().fit(features_df)
joblib.dump(scaler, SCALER_OUT)
print(f"Saved scaler → {SCALER_OUT}")

features = scaler.transform(features_df).astype(np.float32)
labels   = sensor["target"].values.astype(np.int64)

# ─── 3) SLIDING WINDOWS ─────────────────────────────────────────────────
X, y = [], []
for i in range(WINDOW_SIZE-1, len(features)):
    window = features[i-WINDOW_SIZE+1 : i+1]  # shape (WINDOW_SIZE,4)
    X.append(window.flatten())               # flatten to (WINDOW_SIZE*4,)
    y.append(labels[i])
X = np.stack(X)  # (N, WINDOW_SIZE*4)
y = np.array(y)

# ─── 4) SPLIT & SAMPLER ─────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# weighted sampler for class balance
class_counts   = np.bincount(y_train)
class_weights  = 1.0 / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# ─── 5) DATASET & DATALOADERS ────────────────────────────────────────────
class FootfallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = FootfallDataset(X_train, y_train)
val_ds   = FootfallDataset(X_val,   y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ─── 6) MODEL DEFINITION ───────────────────────────────────────────────
class FootfallLogistic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # single linear layer → logistic regression
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        # x: [B, WINDOW_SIZE*4]
        return self.linear(x).squeeze(-1)  # raw logits

input_dim = WINDOW_SIZE * 4
model     = FootfallLogistic(input_dim).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# ─── 7) TRAIN LOOP ──────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, EPOCHS+1):
    # training
    model.train()
    total_loss = 0.0
    correct = total = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == yb.long()).sum().item()
        total += xb.size(0)

    train_loss = total_loss / total
    train_acc  = correct / total

    # validation
    model.eval()
    val_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]  "):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            val_loss += criterion(logits, yb).item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += xb.size(0)

    val_loss = val_loss / total
    val_acc  = correct / total

    print(f"Epoch {epoch:03d}: "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # checkpoint if improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUT)
        print(f" → New best model saved (val_acc={best_val_acc:.4f})")

    scheduler.step(val_acc)

print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
