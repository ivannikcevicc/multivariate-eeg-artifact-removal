import numpy as np
import matplotlib.pyplot as plt


def ols_clean(eeg, artifacts, reg=1e-6):
    """
    eeg:        (n_eeg, T)
    artifacts:  (n_art, T)
    """

    A = artifacts - artifacts.mean(axis=1, keepdims=True)

    ATA = A @ A.T
    ATA += reg * np.eye(A.shape[0])

    beta = eeg @ A.T @ np.linalg.inv(ATA)

    cleaned = eeg - beta @ A

    return cleaned, beta


def plot_ols(eeg: np.ndarray, cleaned: np.ndarray, ecg: np.ndarray,
             sfreq: float, channel_names: list = None):
    """Plot raw vs cleaned EEG and ECG signal."""
    n_channels, T = eeg.shape
    time = np.arange(T) / sfreq
    names = channel_names or [f"CH{i+1}" for i in range(n_channels)]

    for i, ch in enumerate(names):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(time, eeg[i],     label="Raw EEG",     alpha=0.7)
        ax.plot(time, cleaned[i], label="Cleaned EEG", alpha=0.9)
        ax.set_title(ch)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (V)")
        ax.legend()
        fig.tight_layout()

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(time, ecg.squeeze())
    ax.set_title("ECG Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")
    fig.tight_layout()
    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    from pathlib import Path
    import mne

    BASE_DIR = Path(__file__).resolve().parent.parent
    EDF_FILE = BASE_DIR / "data" / "n1.edf"

    EEG_CHANNELS   = ["F3-C3", "C3-P3", "C4-A1"]
    ECG_CHANNEL    = ["ECG1-ECG2"]
    WINDOW_SECONDS = 10

    raw     = mne.io.read_raw_edf(EDF_FILE, preload=False, verbose=False)
    sfreq   = raw.info["sfreq"]
    n_samples = int(WINDOW_SECONDS * sfreq)

    eeg_data = raw.get_data(picks=EEG_CHANNELS, start=0, stop=n_samples)
    ecg_data = raw.get_data(picks=ECG_CHANNEL,  start=0, stop=n_samples)

    cleaned, beta = ols_clean(eeg_data, ecg_data)

    print(f"Beta: {beta}")
    print(f"Cleaned shape: {cleaned.shape}")

    plot_ols(eeg_data, cleaned, ecg_data, sfreq, EEG_CHANNELS)
    print("\nDone.")