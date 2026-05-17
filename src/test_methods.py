from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from regression_method import ols_clean
from gaussian_method   import gmr_clean


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
EDF_FILE = BASE_DIR / "data" / "n1.edf"

# ------------------------------------------------------------
# EEG CHANNELS
# ------------------------------------------------------------

EEG_CHANNELS = [
    "F2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "F1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "C4-A1",
]

# ------------------------------------------------------------
# ARTIFACT CHANNELS
# ------------------------------------------------------------

ARTIFACT_CHANNELS = [
    "ECG1-ECG2",
    "EMG1-EMG2",
    "TORACE",
]

# ------------------------------------------------------------
# WINDOWS
# EEG and artifact windows intentionally DIFFERENT
# to create synthetic contamination benchmark
# ------------------------------------------------------------

EEG_START_SEC = 600
EEG_END_SEC   = 720

ART_START_SEC = 1800
ART_END_SEC   = 1920

# ------------------------------------------------------------

SNR_TARGETS_DB = [5, 10, 20]

USE_MULTICHANNEL_ARTIFACTS = True

# ============================================================
# LOAD EDF
# ============================================================

print("\nLoading EDF...")

raw = mne.io.read_raw_edf(
    EDF_FILE,
    preload=False,
    verbose=False
)

sfreq = raw.info["sfreq"]

# ============================================================
# LOAD CLEAN EEG
# ============================================================

eeg_clean = raw.get_data(
    picks=EEG_CHANNELS,
    start=int(EEG_START_SEC * sfreq),
    stop=int(EEG_END_SEC * sfreq)
)

# ============================================================
# LOAD ARTIFACT CHANNELS
# ============================================================

artifact_data = raw.get_data(
    picks=ARTIFACT_CHANNELS,
    start=int(ART_START_SEC * sfreq),
    stop=int(ART_END_SEC * sfreq)
)

# ============================================================
# MATCH LENGTHS
# ============================================================

T = min(
    eeg_clean.shape[1],
    artifact_data.shape[1]
)

eeg_clean    = eeg_clean[:, :T]
artifact_data = artifact_data[:, :T]

time = np.arange(T) / sfreq

n_eeg = eeg_clean.shape[0]
n_art = artifact_data.shape[0]

print(f"\nEEG channels:      {n_eeg}")
print(f"Artifact channels: {n_art}")
print(f"Samples:           {T}")
print(f"Sampling rate:     {sfreq} Hz")

# ============================================================
# INDEPENDENCE CHECK
# ============================================================

print("\nChecking EEG/artifact independence...")

for i, art_name in enumerate(ARTIFACT_CHANNELS):

    corr = np.mean([
        np.corrcoef(
            eeg_clean[ch],
            artifact_data[i]
        )[0,1]
        for ch in range(n_eeg)
    ])

    print(f"{art_name:<12} mean corr = {corr:.4f}")

# ============================================================
# CONTAMINATION GENERATOR
# ============================================================

def compute_scaling(eeg, artifact, snr_db):

    eeg_power = np.mean(np.var(eeg, axis=1))

    art_power = np.var(artifact)

    target = eeg_power / (10 ** (snr_db / 10.0))

    return np.sqrt(target / (art_power + 1e-30))

# ============================================================
# METRICS
# ============================================================

def snr_per_channel(clean, recovered):

    signal_power = np.mean(clean**2, axis=1)

    error_power = np.mean(
        (recovered - clean)**2,
        axis=1
    )

    return 10 * np.log10(
        signal_power / (error_power + 1e-30)
    )

# ------------------------------------------------------------

def mean_corr(clean, recovered):

    vals = []

    for i in range(clean.shape[0]):

        r = pearsonr(
            clean[i],
            recovered[i]
        )[0]

        vals.append(r)

    return float(np.mean(vals))

# ------------------------------------------------------------

def residual_artifact_r2(cleaned, artifacts):

    vals = []

    for ch in range(cleaned.shape[0]):

        for art in range(artifacts.shape[0]):

            r = np.corrcoef(
                cleaned[ch],
                artifacts[art]
            )[0,1]

            vals.append(r**2)

    return float(np.mean(vals))

# ============================================================
# RUN METHODS
# ============================================================

def run_methods(eeg_obs, artifacts):

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

    ols_cleaned, ols_beta = ols_clean(
        eeg_obs,
        artifacts
    )

    # --------------------------------------------------------
    # GMR
    # --------------------------------------------------------

    gmr_cleaned, gmr_beta = gmr_clean(
        eeg_obs,
        artifacts
    )

    return {

        "ols_result": ols_cleaned,
        "gmr_result": gmr_cleaned,

        "ols_beta": ols_beta,
        "gmr_beta": gmr_beta,

        "ols": {

            "snr": snr_per_channel(
                eeg_clean,
                ols_cleaned
            ),

            "corr": mean_corr(
                eeg_clean,
                ols_cleaned
            ),

            "r2": residual_artifact_r2(
                ols_cleaned,
                artifacts
            ),
        },

        "gmr": {

            "snr": snr_per_channel(
                eeg_clean,
                gmr_cleaned
            ),

            "corr": mean_corr(
                eeg_clean,
                gmr_cleaned
            ),

            "r2": residual_artifact_r2(
                gmr_cleaned,
                artifacts
            ),
        }
    }

# ============================================================
# BENCHMARK
# ============================================================

all_results = {}

for snr_db in SNR_TARGETS_DB:

    print(f"\n{'='*60}")
    print(f"Running benchmark @ {snr_db} dB")

    # --------------------------------------------------------
    # BUILD MIXED ARTIFACT
    # --------------------------------------------------------

    contamination = np.zeros(T)

    scales = []

    for i in range(n_art):

        scale = compute_scaling(
            eeg_clean,
            artifact_data[i],
            snr_db + i
        )

        scales.append(scale)

        contamination += scale * artifact_data[i]

    # --------------------------------------------------------
    # CONTAMINATE EEG
    # --------------------------------------------------------

    eeg_obs = eeg_clean + contamination[None, :]

    actual_snr = 10 * np.log10(
        np.mean(np.var(eeg_clean, axis=1))
        /
        np.var(contamination)
    )

    print(f"Actual SNR: {actual_snr:.2f} dB")

    # --------------------------------------------------------
    # RUN METHODS
    # --------------------------------------------------------

    res = run_methods(
        eeg_obs,
        artifact_data
    )

    res["contamination"] = contamination
    res["eeg_obs"]       = eeg_obs

    all_results[snr_db] = res

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n")
print("="*90)
print("MULTICHANNEL EEG ARTIFACT REMOVAL BENCHMARK")
print("="*90)

print(
    f"{'SNR':<8}"
    f"{'METHOD':<10}"
    f"{'MEAN SNR':>12}"
    f"{'MEAN CORR':>14}"
    f"{'RESIDUAL R²':>16}"
)

print("-"*90)

for snr_db in SNR_TARGETS_DB:

    for method in ["ols", "gmr"]:

        mean_snr = np.mean(
            all_results[snr_db][method]["snr"]
        )

        mean_corr_val = all_results[snr_db][method]["corr"]

        r2 = all_results[snr_db][method]["r2"]

        print(
            f"{snr_db:<8}"
            f"{method.upper():<10}"
            f"{mean_snr:>12.3f}"
            f"{mean_corr_val:>14.4f}"
            f"{r2:>16.6f}"
        )

print("="*90)

# ============================================================
# PLOT 1
# Mean SNR
# ============================================================

fig1, ax1 = plt.subplots(figsize=(8,5))

x = np.arange(len(SNR_TARGETS_DB))

ols_vals = [
    np.mean(all_results[s]["ols"]["snr"])
    for s in SNR_TARGETS_DB
]

gmr_vals = [
    np.mean(all_results[s]["gmr"]["snr"])
    for s in SNR_TARGETS_DB
]

width = 0.35

ax1.bar(
    x - width/2,
    ols_vals,
    width,
    label="OLS"
)

ax1.bar(
    x + width/2,
    gmr_vals,
    width,
    label="GMR"
)

ax1.set_xticks(x)
ax1.set_xticklabels(
    [f"{s} dB" for s in SNR_TARGETS_DB]
)

ax1.set_ylabel("Recovered SNR (dB)")
ax1.set_title("OLS vs GMR")
ax1.legend()

plt.tight_layout()

# ============================================================
# PLOT 2
# FULL TRACE
# ============================================================

snr_show = 10

res = all_results[snr_show]

ch = 0

fig2, axes = plt.subplots(
    4,
    1,
    figsize=(14,10),
    sharex=True
)

fig2.suptitle(
    f"Channel: {EEG_CHANNELS[ch]}"
)

# ------------------------------------------------------------

axes[0].plot(
    time,
    eeg_clean[ch] * 1e6,
    label="Clean EEG"
)

axes[0].set_ylabel("µV")
axes[0].legend()

# ------------------------------------------------------------

axes[1].plot(
    time,
    res["eeg_obs"][ch] * 1e6,
    label="Contaminated"
)

axes[1].plot(
    time,
    res["contamination"] * 1e6,
    alpha=0.6,
    label="Artifact"
)

axes[1].set_ylabel("µV")
axes[1].legend()

# ------------------------------------------------------------

axes[2].plot(
    time,
    eeg_clean[ch] * 1e6,
    alpha=0.5,
    label="True"
)

axes[2].plot(
    time,
    res["ols_result"][ch] * 1e6,
    label="OLS"
)

axes[2].set_ylabel("µV")
axes[2].legend()

# ------------------------------------------------------------

axes[3].plot(
    time,
    eeg_clean[ch] * 1e6,
    alpha=0.5,
    label="True"
)

axes[3].plot(
    time,
    res["gmr_result"][ch] * 1e6,
    label="GMR"
)

axes[3].set_ylabel("µV")
axes[3].set_xlabel("Time (s)")
axes[3].legend()

plt.tight_layout()

# ============================================================
# PLOT 3
# ARTIFACT CHANNELS
# ============================================================

fig3, axes3 = plt.subplots(
    n_art,
    1,
    figsize=(14, 8),
    sharex=True
)

fig3.suptitle("Artifact Channels")

for i in range(n_art):

    axes3[i].plot(
        time,
        artifact_data[i] * 1e6
    )

    axes3[i].set_ylabel(
        ARTIFACT_CHANNELS[i]
    )

axes3[-1].set_xlabel("Time (s)")

plt.tight_layout()

plt.show()

print("\nDone.")