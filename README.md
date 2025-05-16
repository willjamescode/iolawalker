````markdown
# Footfall Detection (LSTM) Pipeline

This repository contains the `trainAndTest.py` script (in the `lstmFinal/` folder) which:

1. **Trains** a Conv–LSTM model with dual (per-sample and per-window) heads.  
2. **Exports** the best checkpoint to TorchScript (`.pt`) and writes scaler + threshold metadata (`.json`).  
3. **Tests** on held-out data with a tolerance sweep, printing TP/FN/FP/Precision/Recall/F₁.

---

## Prerequisites

- **Python 3.8+**  
- A modern GPU (optional, but recommended for training speed).  
- The two CSV files:
  - `train200hz.csv`
  - `test200hz.csv`  
  placed somewhere on disk; paths are configured in the script.

---

## Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
````

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # (macOS/Linux)
   # .\venv\Scripts\Activate.ps1  # (Windows PowerShell)
   ```
3. **Install dependencies**

   ```bash
   pip install --upgrade pip setuptools
   pip install -r requirements.txt
   ```
4. **Place your CSV data**
   Edit the top of `lstmFinal/trainAndTest.py` to point to your
   `TRAIN_CSV` and `TEST_CSV` absolute paths.

---

## Usage

Run training, export, and testing in one go:

```bash
python lstmFinal/trainAndTest.py
```

The script will:

* Fit the `SeqFootfallDataset` on `TRAIN_CSV`, generate sliding windows.
* Train for `EPOCHS` epochs, tuning the best validation threshold.
* Save:

  * `best_refined_v2.pth` (checkpoint),
  * `refined_v2_ts.pt` (TorchScript model),
  * `metadata_refined_v2.json` (window size, stride, threshold, scaler stats).
* Load `TEST_CSV`, run a tolerance sweep (0–50 samples), and print JSON results.

---

## Configuration

All key constants live at the top of `trainAndTest.py`:

```python
TRAIN_CSV      = "/full/path/to/train200hz.csv"
TEST_CSV       = "/full/path/to/test200hz.csv"
SCALER_OUT     = "scaler_refined_v2.pkl"
MODEL_OUT      = "best_refined_v2.pth"
TS_MODEL_OUT   = "refined_v2_ts.pt"
METADATA_OUT   = "metadata_refined_v2.json"

WINDOW_SIZE    = 600    # samples per window (3 s @200 Hz)
STRIDE         = 150    # window hop (75% overlap)
BATCH_SIZE     = 64
LR             = 1e-3
EPOCHS         = 50
POS_WINDOW_OVERSAMPLE = 10.0
SEED           = 42
CLIP_NORM      = 5.0
```

Adjust these values as needed before running.

---

## Results

* **Checkpoint:** `best_refined_v2.pth`
* **TorchScript model:** `refined_v2_ts.pt`
* **Metadata:** `metadata_refined_v2.json`
* **Test metrics:** printed to console in JSON format.

---

## Running the Android App in Android Studio

1. **Open Android Studio**

   * Launch Android Studio (Arctic Fox or later).
   * Select **Open an existing project** and navigate to the `android-app/` (or `app/`) directory in this repo.

2. **Sync & Build**

   * Allow Gradle to sync dependencies.
   * The project uses the Android Gradle Plugin; ensure you have the Android SDK and the appropriate build tools installed.

3. **Run on Device or Emulator**

   * Connect a physical device via USB (enable Developer Options & USB debugging) or start an Android emulator.
   * Click the **Run** ▶️ button, choose your target, and deploy the app.

4. **Interactive Tuning**

   * Once the app launches, navigate to the “Footfall Tuning” screen.
   * Adjust sliders for sample/window thresholds, inference stride, hit window, required hits, and refractory period.
   * Observe real-time feedback and “ding” sounds as you walk.

Happy footfall detecting! 🚶‍♂️🎶

```
```

