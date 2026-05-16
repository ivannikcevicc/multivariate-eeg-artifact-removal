from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

EDF_FILE = BASE_DIR / "data" / "n1.edf"

EEG_CHANNELS = [
    "F3-C3",
    "C3-P3",
    "C4-A1",
]

ECG_CHANNEL = ["ECG1-ECG2"]
WINDOW_SECONDS = 10


raw = mne.io.read_raw_edf(EDF_FILE, preload=False, verbose=False)

sfreq = raw.info["sfreq"]

# print(f"Signal frequency: {SIGNAL_FREQ} Hz")
# print(raw.info)
# print(raw.info.ch_names)

n_samples = int(WINDOW_SECONDS * sfreq)


eeg_data = raw.get_data(picks=EEG_CHANNELS, start=0, stop=n_samples)
ecg_data = raw.get_data(picks=ECG_CHANNEL, start=0, stop=n_samples)

# print(f"EEG shape: {eeg_data}")
# print(f"ECG shape: {ecg_data}")


eeg = eeg_data[:, :n_samples]   # (3, n_samples)
ecg = ecg_data[:, :n_samples]   # (1, n_samples)

X = np.vstack([eeg, ecg])

cov = np.cov(X)

sigma_eeg_ecg = cov[3,:3] 
sigma_ecg_ecg = cov[3,3]

beta = sigma_eeg_ecg / sigma_ecg_ecg
beta = beta.reshape(-1, 1)

output_eeg = eeg - beta*ecg

print(output_eeg)



# ============================================================
# PLOTTING: RAW vs CLEANED EEG
# ============================================================



time = np.arange(n_samples) / sfreq

for i, ch in enumerate(EEG_CHANNELS):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, eeg[i],         label="Raw EEG",     alpha=0.7)
    ax.plot(time, output_eeg[i], label="Cleaned EEG", alpha=0.9)
    ax.set_title(ch)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")
    ax.legend()
    fig.tight_layout()

# ============================================================
# PLOT: ECG
# ============================================================

fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(time, ecg[0])
ax.set_title("ECG Signal")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (V)")
fig.tight_layout()

plt.show()

print("\nDone.")