import mne
import matplotlib.pyplot as plt

# ============================================
# CONFIG
# ============================================

EDF_FILE = "data/n1.edf"

# ============================================
# LOAD EDF
# ============================================

print("Loading EDF file...")

raw = mne.io.read_raw_edf(
    EDF_FILE,
    preload=True,
    verbose=False
)

print("EDF loaded successfully.\n")

# ============================================
# BASIC INFO
# ============================================

print("=" * 50)
print("GENERAL INFO")
print("=" * 50)

print(raw.info)

# ============================================
# CHANNEL NAMES
# ============================================

print("\n")
print("=" * 50)
print("CHANNELS")
print("=" * 50)

for i, ch in enumerate(raw.ch_names):
    print(f"{i:02d}: {ch}")

# ============================================
# SAMPLING RATE
# ============================================

sfreq = raw.info["sfreq"]

print("\n")
print("=" * 50)
print("SAMPLING RATE")
print("=" * 50)

print(f"{sfreq} Hz")

# ============================================
# DURATION
# ============================================

duration_sec = raw.times[-1]

print("\n")
print("=" * 50)
print("DURATION")
print("=" * 50)

print(f"{duration_sec / 60:.2f} minutes")

# ============================================
# PICK CHANNELS
# ============================================

# CHANGE THESE AFTER YOU SEE REAL CHANNEL NAMES
EEG_CHANNELS = [
    "F3-C3",
    "C3-P3",
    "C4-A1"
]

ECG_CHANNEL = "ECG1-ECG2"

# ============================================
# VERIFY CHANNELS EXIST
# ============================================

available_channels = raw.ch_names

for ch in EEG_CHANNELS + [ECG_CHANNEL]:
    if ch not in available_channels:
        print(f"WARNING: Channel not found -> {ch}")

# ============================================
# EXTRACT DATA
# ============================================

print("\n")
print("=" * 50)
print("EXTRACTING DATA")
print("=" * 50)

eeg_data = raw.get_data(picks=EEG_CHANNELS)

ecg_data = raw.get_data(picks=[ECG_CHANNEL])

print(f"EEG shape: {eeg_data.shape}")
print(f"ECG shape: {ecg_data.shape}")

# ============================================
# PLOT RAW SIGNALS
# ============================================

print("\nPlotting signals...")

raw.plot(
    duration=10,
    n_channels=5,
    scalings="auto"
)

plt.show()

# ============================================
# SAVE SMALL SEGMENT
# ============================================

print("\n")
print("=" * 50)
print("FIRST 10 SECONDS")
print("=" * 50)

start_sec = 0
end_sec = 10

start_sample = int(start_sec * sfreq)
end_sample = int(end_sec * sfreq)

segment = eeg_data[:, start_sample:end_sample]

print(f"Segment shape: {segment.shape}")

print("\nDone.")
