import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


def gmr_clean(eeg, artifacts, n_components=4, reg=1e-6):
    n_eeg, T = eeg.shape
    n_art     = artifacts.shape[0]

    X = np.vstack([eeg, artifacts]).T       # (T, n_eeg + n_art)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=reg,
        max_iter=300,
        random_state=0
    )
    gmm.fit(X)

    log_probs = np.zeros((T, n_components))

    for k in range(n_components):
        mu_art_k   = gmm.means_[k][n_eeg:]
        Sigma_aa_k = gmm.covariances_[k][n_eeg:, n_eeg:] + reg * np.eye(n_art)
        log_probs[:, k] = (
            np.log(gmm.weights_[k] + 1e-10)
            + multivariate_normal.logpdf(artifacts.T, mean=mu_art_k, cov=Sigma_aa_k)
        )

    # softmax to get responsibilities
    log_probs -= log_probs.max(axis=1, keepdims=True)
    r = np.exp(log_probs)
    r /= r.sum(axis=1, keepdims=True)      # (T, K)

    # conditional mean E[EEG | a_t]
    E_eeg_given_art = np.zeros((n_eeg, T))

    for k in range(n_components):
        mu_eeg_k   = gmm.means_[k][:n_eeg]
        mu_art_k   = gmm.means_[k][n_eeg:]
        Sigma_ea_k = gmm.covariances_[k][:n_eeg, n_eeg:]
        Sigma_aa_k = gmm.covariances_[k][n_eeg:, n_eeg:] + reg * np.eye(n_art)

        L_k        = np.linalg.solve(Sigma_aa_k, Sigma_ea_k.T).T   # (n_eeg, n_art)
        a_centered = artifacts - mu_art_k[:, None]                  # (n_art, T)
        mu_cond_k  = mu_eeg_k[:, None] + L_k @ a_centered          # (n_eeg, T)

        E_eeg_given_art += r[:, k][None, :] * mu_cond_k

    cleaned = eeg - E_eeg_given_art

    return cleaned, None