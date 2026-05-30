"""Spectral preprocessing transformations for olive oil chemometric data."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

SpectralTuple = tuple[np.ndarray, np.ndarray, list[str]]


def row_profile(spectra: SpectralTuple) -> SpectralTuple:
    """Normalise each spectrum to unit row sum.

    Divides every intensity by the row total, converting absolute counts to
    relative profiles. Removes sample-to-sample total-intensity variation —
    standard first step for HS-MS count data.

    Parameters
    ----------
    spectra:
        ``(X, axis, sample_ids)`` tuple as returned by ``get_spectral_matrix``.

    Returns
    -------
    Same tuple structure with X replaced by the row-profiled matrix.
    """
    X, axis, sample_ids = spectra
    row_sums = X.sum(axis=1, keepdims=True)
    return X / row_sums, axis, sample_ids


def log_transform(
    spectra: SpectralTuple,
    base: float = np.e,
) -> SpectralTuple:
    """Apply a logarithmic transformation to spectral intensities.

    Uses ``log(1 + x)`` so that zero values map to zero rather than ``-inf``.

    Parameters
    ----------
    spectra:
        ``(X, axis, sample_ids)`` tuple as returned by ``get_spectral_matrix``.
    base:
        Logarithm base. Defaults to ``e`` (natural log). Pass ``10`` for
        log₁₀, which is common for MS count data.

    Returns
    -------
    Same tuple structure with X replaced by the log-transformed matrix.
    """
    X, axis, sample_ids = spectra
    X_log = np.log1p(X)
    if base != np.e:
        X_log = X_log / np.log(base)
    return X_log, axis, sample_ids


def snv(spectra: SpectralTuple) -> SpectralTuple:
    """Apply Standard Normal Variate (SNV) normalisation row-wise.

    Each spectrum is mean-centred and scaled to unit standard deviation
    independently. Removes multiplicative scatter and baseline effects —
    standard pre-treatment for MIR and UV-Vis diffuse-reflectance data.

    Parameters
    ----------
    spectra:
        ``(X, axis, sample_ids)`` tuple as returned by ``get_spectral_matrix``.

    Returns
    -------
    Same tuple structure with X replaced by the SNV-normalised matrix.
    """
    X, axis, sample_ids = spectra
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True, ddof=1)
    return (X - mean) / std, axis, sample_ids


def savgol_derivative(
    spectra: SpectralTuple,
    deriv: int = 1,
    polyorder: int = 2,
    half_window: int = 7,
) -> SpectralTuple:
    """Compute a Savitzky-Golay derivative of each spectrum.

    Fits a polynomial locally to a sliding window and returns the analytical
    derivative of that polynomial, combining smoothing and differentiation in
    one step.

    Parameters
    ----------
    spectra:
        ``(X, axis, sample_ids)`` tuple as returned by ``get_spectral_matrix``.
    deriv:
        Order of the derivative (1 = first derivative, 2 = second, …).
    polyorder:
        Degree of the fitting polynomial. Must be ≥ ``deriv``.
        Default 2 (quadratic) matches the reference paper for UV-Vis.
    half_window:
        Number of points on each side of the centre point, so the full
        window is ``2 * half_window + 1`` (always odd).
        Default 7 → window of 15 points.

    Returns
    -------
    Same tuple structure with X replaced by the derivative matrix.
    The axis values and sample IDs are unchanged.

    Notes
    -----
    ``delta`` is set to the absolute axis spacing so the derivative is
    expressed per axis unit (nm, cm⁻¹, or m/z). Assumes a uniformly
    spaced axis; for FTIR the axis is decreasing, so ``abs`` is applied.
    """
    X, axis, sample_ids = spectra
    window_length = 2 * half_window + 1
    delta = abs(float(axis[1] - axis[0])) if len(axis) > 1 else 1.0
    X_deriv = savgol_filter(
        X,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
        axis=1,
    )
    return X_deriv, axis, sample_ids


def mean_center(
    spectra: SpectralTuple,
    center: np.ndarray | None = None,
) -> tuple[SpectralTuple, np.ndarray]:
    """Subtract the column mean from each spectral variable.

    Parameters
    ----------
    spectra:
        ``(X, axis, sample_ids)`` tuple as returned by ``get_spectral_matrix``.
    center:
        Pre-computed column means to subtract (shape ``(n_variables,)``).
        When ``None``, the mean is computed from ``X`` itself.
        Pass the training-set mean when transforming a test set to avoid
        data leakage.

    Returns
    -------
    spectra_centered:
        Same tuple structure with X replaced by the mean-centered matrix.
    center_used:
        The column-mean vector that was subtracted. Store this when fitting
        on a calibration set so it can be applied to the test set.
    """
    X, axis, sample_ids = spectra
    if center is None:
        center = X.mean(axis=0)
    return (X - center, axis, sample_ids), center
