#!/usr/bin/env python3
"""
test_logistic_events.py

Event‐level evaluation of the logistic‐regression model,
plus a bar chart of TP, FP, FN counts (±100 ms).
"""
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 50
SCALER_PATH = "logistic_scaler.pkl"
MODEL_PATH  = "best_footfall_logistic.pt"
TEST_CSV    = "/home/u/footfallTraining/newtest2.csv"
THRESHOLD   = 0.50      # probability cutoff
TOL_MS      = 100       # matching tolerance (ms)

# ─── MODEL ────────────────────────────────────────────────────────────────
class FootfallLogistic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze(-1)

# ─── MATCHING ──────────────────────────────────────────────────────────────
def match_events(pred, true, tol):
    TP = FP = FN = 0
    used = set()
    errors = []
    for t in true:
        cands = [(abs(p - t), i, p) for i,p in enumerate(pred) if i not in used]
        if not cands:
            FN += 1; continue
        diff,i,p_closest = min(cands)
        if diff <= tol:
            TP += 1; used.add(i); errors.append(p_closest - t)
        else:
            FN += 1
    FP = len(pred) - len(used)
    return TP, FP, FN, errors

# ─── LOAD & LABEL ─────────────────────────────────────────────────────────
df = pd.read_csv(TEST_CSV, sep=r"[,\t]+", engine="python", header=0)
df = df.rename(columns={"deviceAddress":"id","accelerationX":"ax","accelerationY":"ay","accelerationZ":"az"})
true_times = df.loc[df["id"]=="FOOTFALL_EVENT","timestamp"].astype(int).tolist()
sensor = df[df["id"]!="FOOTFALL_EVENT"].reset_index(drop=True)
sensor["target"] = 0
for t in true_times:
    idx = (sensor["timestamp"] - t).abs().idxmin()
    sensor.at[idx,"target"] = 1
acc = sensor[["ax","ay","az"]].values.astype(np.float32)
sensor["mag"] = np.linalg.norm(acc,axis=1)

# ─── SCALE & WINDOW ───────────────────────────────────────────────────────
feat_df = sensor[["ax","ay","az","mag"]].astype(np.float32)
timestamps = sensor["timestamp"].astype(int).values
labels     = sensor["target"].values.astype(int)
scaler     = joblib.load(SCALER_PATH)
scaled     = scaler.transform(feat_df)

Xw, Tw = [], []
for i in range(WINDOW_SIZE-1, len(scaled)):
    Xw.append(scaled[i-WINDOW_SIZE+1:i+1].flatten())
    Tw.append(timestamps[i])
Xw = np.stack(Xw)
Tw = np.array(Tw)

# ─── INFERENCE ────────────────────────────────────────────────────────────
device = torch.device("cpu")
model  = FootfallLogistic(WINDOW_SIZE*4).to(device)
model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(Xw).float())
probs = torch.sigmoid(logits).cpu().numpy()

# ─── PEAK DETECTION ───────────────────────────────────────────────────────
pred_times = []
for i in range(1,len(probs)-1):
    if probs[i]>THRESHOLD and probs[i]>=probs[i-1] and probs[i]>=probs[i+1]:
        pred_times.append(int(Tw[i]))

# ─── MATCH & REPORT ───────────────────────────────────────────────────────
TP,FP,FN,errors = match_events(pred_times,true_times,TOL_MS)
print("Logistic Regression (±100 ms):")
print(f"  True events:      {len(true_times)}")
print(f"  Predicted events: {len(pred_times)}")
print(f"  TP = {TP}, FP = {FP}, FN = {FN}")
if errors:
    print(f"  Mean timing error: {np.mean(errors):.1f} ms ±{np.std(errors):.1f} ms")

# ─── PLOT COUNTS ──────────────────────────────────────────────────────────
stats = {"TP":TP,"FP":FP,"FN":FN}
plt.figure(figsize=(6,4))
bars = plt.bar(stats.keys(), stats.values())
plt.title("Logistic Regression Predictions within 100 ms")
plt.ylabel("Count")
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, h+1, f"{int(h)}", ha="center")
if errors:
    me,std = np.mean(errors), np.std(errors)
    plt.text(0.5, max(stats.values())*0.9,
             f"Mean error: {me:.1f} ms ±{std:.1f} ms",
             ha="center", color="gray")
plt.tight_layout()
plt.savefig("logistic_event_stats.png")
print("Saved diagram → logistic_event_stats.png")
