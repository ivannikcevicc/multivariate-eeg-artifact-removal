from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EDF_FILE = BASE_DIR / "data" / "n1.edf"

EEG_CHANNELS = [
    "F3-C3",
    "C3-P3",
    "C4-A1"
]

ECG_CHANNEL = "ECG1-ECG2"

WINDOW_SECONDS = 10

# ============================================================
# LOAD DATA
# ============================================================

print("Loading EDF...")

raw = mne.io.read_raw_edf(
    EDF_FILE,
    preload=True,
    verbose=False
)

print("Loaded.")

# ============================================================
# EXTRACT CHANNELS
# ============================================================

sfreq = raw.info["sfreq"]

eeg = raw.get_data(picks=EEG_CHANNELS)

ecg = raw.get_data(picks=[ECG_CHANNEL])

print(f"EEG shape: {eeg.shape}")
print(f"ECG shape: {ecg.shape}")

# ============================================================
# TAKE SMALL WINDOW
# ============================================================

n_samples = int(WINDOW_SECONDS * sfreq)

eeg = eeg[:, :n_samples]

ecg = ecg[:, :n_samples]

print(f"\nWindow size: {WINDOW_SECONDS} sec")
print(f"Samples: {n_samples}")

# ============================================================
# BUILD MULTIVARIATE MATRIX
# ============================================================

X = np.vstack([eeg, ecg])

print(f"\nX shape: {X.shape}")

# ============================================================
# REMOVE MEAN
# ============================================================

X_centered = X - X.mean(axis=1, keepdims=True)

# ============================================================
# COVARIANCE MATRIX
# ============================================================

cov = np.cov(X_centered)

print("\nCovariance matrix:")
print(cov)

# ============================================================
# ECG REGRESSION
# ============================================================

# ECG signal
h = ecg[0]

cleaned_eeg = []

betas = []

for i in range(eeg.shape[0]):

    # EEG channel
    y = eeg[i]

    # ============================================
    # Estimate contamination coefficient
    # ============================================

    beta = np.cov(y, h)[0, 1] / np.var(h)

    betas.append(beta)

    # ============================================
    # Estimated artifact
    # ============================================

    artifact = beta * h

    # ============================================
    # Cleaned EEG
    # ============================================

    cleaned = y - artifact

    cleaned_eeg.append(cleaned)

cleaned_eeg = np.array(cleaned_eeg)

# ============================================================
# PRINT BETAS
# ============================================================

print("\nEstimated contamination coefficients:")

for ch, beta in zip(EEG_CHANNELS, betas):
    print(f"{ch}: {beta:.6f}")

# ============================================================
# PLOT RESULTS
# ============================================================

time = np.arange(n_samples) / sfreq

for i, ch in enumerate(EEG_CHANNELS):

    plt.figure(figsize=(14, 6))

    plt.plot(time, eeg[i], label="Raw EEG")

    plt.plot(time, cleaned_eeg[i], label="Cleaned EEG")

    plt.title(ch)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.legend()

    plt.tight_layout()

plt.show()

# ============================================================
# ECG PLOT
# ============================================================

plt.figure(figsize=(14, 4))

plt.plot(time, h)

plt.title("ECG Signal")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.show()

print("\nDone.")