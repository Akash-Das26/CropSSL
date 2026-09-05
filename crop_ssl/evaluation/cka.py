"""
CKA (Centered Kernel Alignment) Representation Analysis.

Quantifies how *similar* two neural representations are, which is the core
diagnostic for the cross-domain robustness question: when a model trained on
lab (source) images is deployed to real field (target) images, how much do its
internal representations shift?

    CKA(X, Y) in [0, 1]   -- 1.0 means the two feature spaces encode the same
                              information up to invertible linear transform;
                              ~0 means they are essentially uncorrelated.

Following Kornblith et al., "Similarity of Neural Network Representations
Revisited" (ICML 2019), we use the *linear* kernel by default because it is
numerically stable, cheap, and the standard choice for comparing layers /
domains in deep networks. An RBF (Gaussian) kernel variant is provided with a
median-heuristic bandwidth for non-linear similarity.

Usage in the CropSSL pipeline
-----------------------------
    from crop_ssl.evaluation.cka import linear_cka, cka_similarity_matrix

    src_feats = encode(backbone, lab_loader)      # (N, D)
    fld_feats = encode(backbone, field_loader)    # (N, D)  -- same N samples

    sim = linear_cka(src_feats, fld_feats)        # domain-gap score
    mat = cka_similarity_matrix({"lab": src_feats, "field": fld_feats,
                                 "adapted": adapt_feats})  # pairwise matrix
"""

from typing import Dict, Optional, Union

import numpy as np
import torch


# --------------------------------------------------------------------------
# core CKA computations
# --------------------------------------------------------------------------
def _to_float64(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Normalise input to a float64 numpy array of shape (N, D)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"CKA expects 2D feature matrices (N samples, D dims), got shape {arr.shape}"
        )
    if arr.shape[0] < 2:
        raise ValueError(
            f"CKA needs at least 2 paired samples, got {arr.shape[0]}"
        )
    return arr


def _center(K: np.ndarray) -> np.ndarray:
    """Double-centre a kernel matrix: K - mean_row - mean_col + grand_mean."""
    col_mean = K.mean(axis=0, keepdims=True)
    row_mean = K.mean(axis=1, keepdims=True)
    return K - col_mean - row_mean + K.mean()


def _hsic(K_c: np.ndarray, L_c: np.ndarray) -> float:
    """Normalised HSIC (centered kernel alignment numerator)."""
    return float((K_c * L_c).sum())


def _kernel_cka(K: np.ndarray, L: np.ndarray) -> float:
    """CKA for two pre-computed kernel matrices."""
    K_c, L_c = _center(K), _center(L)
    numerator = _hsic(K_c, L_c)
    denom = np.sqrt(_hsic(K_c, K_c) * _hsic(L_c, L_c))
    if denom == 0.0:
        return 0.0
    return float(numerator / denom)


def linear_cka(
    X: Union[torch.Tensor, np.ndarray],
    Y: Union[torch.Tensor, np.ndarray],
) -> float:
    """Linear-kernel CKA between two paired feature matrices.

    Args:
        X: Feature matrix (N, D1) -- e.g. source-domain embeddings.
        Y: Feature matrix (N, D2) -- e.g. target-domain embeddings.
            X and Y must share the same number of samples N, with row i of
            each corresponding to the same input image.

    Returns:
        CKA similarity in [0, 1].
    """
    X, Y = _to_float64(X), _to_float64(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples: "
            f"{X.shape[0]} vs {Y.shape[0]}"
        )
    # Center features per dimension (standard practice, Kornblith et al.):
    # removes the "all samples share a common direction" component that would
    # otherwise inflate similarity toward 1 regardless of content.
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    return _kernel_cka(K, L)


def _median_pairwise_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Median squared euclidean distance across all cross pairs (bandwidth heuristic)."""
    n = X.shape[0]
    d2 = np.zeros((n, n))
    for i in range(n):
        diff = X[i] - Y
        d2[i] = (diff * diff).sum(axis=1)
    return float(np.median(d2))


def rbf_cka(
    X: Union[torch.Tensor, np.ndarray],
    Y: Union[torch.Tensor, np.ndarray],
    sigma: Optional[float] = None,
) -> float:
    """RBF (Gaussian) kernel CKA between two paired feature matrices.

    Args:
        X: Feature matrix (N, D1).
        Y: Feature matrix (N, D2). Same number of samples as X.
        sigma: Gaussian bandwidth. If None, uses the median pairwise distance
            heuristic (median of squared distances), which is scale-aware.

    Returns:
        CKA similarity in [0, 1].
    """
    X, Y = _to_float64(X), _to_float64(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples: "
            f"{X.shape[0]} vs {Y.shape[0]}"
        )
    n = X.shape[0]
    if sigma is None:
        sigma = _median_pairwise_distance(X, Y) or 1.0
    K = np.zeros((n, n))
    L = np.zeros((n, n))
    for i in range(n):
        dX = ((X[i] - X) ** 2).sum(axis=1)
        dY = ((Y[i] - Y) ** 2).sum(axis=1)
        K[i] = np.exp(-dX / (2.0 * sigma))
        L[i] = np.exp(-dY / (2.0 * sigma))
    return _kernel_cka(K, L)


# --------------------------------------------------------------------------
# multi-representation helpers
# --------------------------------------------------------------------------
def cka_similarity_matrix(
    features: Dict[str, Union[torch.Tensor, np.ndarray]],
    kernel: str = "linear",
    sigma: Optional[float] = None,
) -> np.ndarray:
    """Pairwise CKA similarity matrix over a set of named representations.

    Args:
        features: Mapping of name -> feature matrix. All matrices must share
            the same number of samples N (paired rows).
        kernel: "linear" or "rbf".
        sigma: RBF bandwidth (ignored for linear kernel).

    Returns:
        float64 (M, M) matrix with 1.0 on the diagonal (each representation
        is perfectly similar to itself). Rows/columns follow the insertion
        order of ``features``.
    """
    names = list(features.keys())
    if len(names) < 2:
        raise ValueError("cka_similarity_matrix needs at least 2 representations")
    n = _to_float64(features[names[0]]).shape[0]
    for name in names[1:]:
        if _to_float64(features[name]).shape[0] != n:
            raise ValueError(
                f"Representation '{name}' has a different number of samples; "
                "all representations must be paired (same N)."
            )
    mat = np.eye(len(names), dtype=np.float64)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if kernel == "rbf":
                s = rbf_cka(features[names[i]], features[names[j]], sigma=sigma)
            else:
                s = linear_cka(features[names[i]], features[names[j]])
            mat[i, j] = mat[j, i] = s
    return mat


def domain_shift_report(
    features: Dict[str, Union[torch.Tensor, np.ndarray]],
    kernel: str = "linear",
    sigma: Optional[float] = None,
) -> Dict:
    """One-call cross-domain diagnosis: pairwise CKA + per-pair summary.

    Args:
        features: Mapping of name -> feature matrix (paired rows).
        kernel: "linear" or "rbf".
        sigma: RBF bandwidth (ignored for linear kernel).

    Returns:
        Dict with "names", the pairwise "matrix" (list-of-lists), and
        "pairs": [{source, target, cka}] for every unordered pair.
    """
    names = list(features.keys())
    mat = cka_similarity_matrix(features, kernel=kernel, sigma=sigma)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append({
                "source": names[i],
                "target": names[j],
                "cka": float(mat[i, j]),
            })
    pairs.sort(key=lambda p: p["cka"])
    return {
        "names": names,
        "matrix": mat.tolist(),
        "pairs": pairs,
    }