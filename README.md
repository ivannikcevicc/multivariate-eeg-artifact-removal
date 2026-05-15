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

# EEG Artifact Removal via ECG Regression

## Signal Model

At each time step, the multivariate observation is modeled as:

$$x_t \sim \mathcal{N}(\mu, \Sigma)$$

where the observation vector stacks all channels:

$$x_t = \begin{bmatrix} \text{EEG}_1(t) \\ \text{EEG}_2(t) \\ \vdots \\ \text{ECG}(t) \end{bmatrix}$$

## Covariance Structure

The covariance matrix is estimated empirically across $T$ time steps:

$$\Sigma = \frac{1}{T} X X^T$$

This captures the statistical dependencies between EEG and ECG channels, which is the basis for identifying artifact contamination.

## Artifact Model

ECG contamination in the EEG is modeled as an additive term:

$$E(t) = S(t) + \beta h(t)$$

| Symbol  | Meaning                            |
| ------- | ---------------------------------- |
| $E(t)$  | Observed (contaminated) EEG signal |
| $S(t)$  | Clean underlying EEG signal        |
| $h(t)$  | ECG signal                         |
| $\beta$ | Contamination coefficient          |

## Artifact Removal

The artifact estimate is:

$$\hat{A}(t) = \beta h(t)$$

and the cleaned EEG is recovered by subtracting the estimate:

$$\hat{S}(t) = E(t) - \hat{A}(t)$$

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
