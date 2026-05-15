# Multivariate EEG Artifact Removal

Probabilistic EEG artifact removal using multivariate Gaussian modeling, covariance estimation, and auxiliary physiological signals such as ECG.

## Overview

This project investigates the removal of physiological artifacts from EEG signals using probabilistic multivariate modeling.

The core idea is to model EEG and auxiliary biosignals (e.g. ECG) jointly as a multivariate stochastic process and estimate artifact contamination through covariance structure.

The estimated artifact component is reconstructed and subtracted from EEG signals in a sliding-window framework.

---

# Motivation

EEG recordings are frequently contaminated by:

- cardiac artifacts
- motion artifacts
- muscle activity
- electrode movement

Traditional approaches such as Independent Component Analysis (ICA) require blind source separation assumptions and can remove useful neural information.

This project explores an alternative probabilistic framework based on:

- covariance modeling
- Gaussian assumptions
- adaptive regression
- multimodal physiological sensing

---

# Mathematical Model

At each time step:

\[
x_t \sim \mathcal{N}(\mu, \Sigma)
\]

where:

\[
x_t =
\begin{bmatrix}
EEG_1(t) \\
EEG_2(t) \\
\vdots \\
ECG(t)
\end{bmatrix}
\]

The covariance matrix:

\[
\Sigma = \frac{1}{T} X X^T
\]

captures dependencies between EEG and ECG channels.

Artifact contamination is modeled as:

\[
E(t) = S(t) + \beta h(t)
\]

where:

- \(S(t)\) is the clean EEG signal
- \(h(t)\) is the ECG signal
- \(\beta\) represents contamination coefficients

The artifact estimate is:

\[
\hat{A}(t) = \beta h(t)
\]

and cleaned EEG is obtained via:

\[
\hat{S}(t) = E(t) - \hat{A}(t)
\]

---

# Planned Features

- Sliding-window covariance estimation
- Adaptive artifact estimation
- ECG-informed EEG cleaning
- Optional accelerometer integration
- Comparison with ICA
- Signal visualization tools
- Spectral preservation analysis

---

# Dataset

Initial experiments use the PhysioNet CAP Sleep Database:

https://physionet.org/content/capslpdb/1.0.0/

Signals used:

- EEG
- ECG

Optional future signals:

- EMG
- motion sensors

---

# Tech Stack

- Python
- NumPy
- SciPy
- MNE
- Matplotlib
- Pandas
