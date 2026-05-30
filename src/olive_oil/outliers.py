"""PCA-based outlier detection for spectral data.

Uses Hotelling's T² (distance in score space) and Q-residuals (reconstruction
error) to flag anomalous samples, with a cross-block discard rule that removes
a sample from all spectral blocks if it is flagged in any one of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import norm
from sklearn.decomposition import PCA

SpectralTuple = tuple[np.ndarray, np.ndarray, list[str]]


@dataclass
class OutlierResult:
    """T², Q-residuals, control limits, and flags for one spectral block.

    Parameters
    ----------
    sample_ids:
        Ordered list of sample identifiers.
    t2:
        Hotelling's T² value per sample.
    q:
        Q-residual (squared reconstruction error) per sample.
    t2_limit:
        Upper control limit for T² at the chosen significance level.
    q_limit:
        Upper control limit for Q at the chosen significance level.
    """

    sample_ids: list[str]
    t2: np.ndarray
    q: np.ndarray
    t2_limit: float
    q_limit: float

    def __eq__(self, other: object) -> bool:
        return NotImplemented  # arrays make default equality ambiguous

    @property
    def t2_flags(self) -> np.ndarray:
        """Boolean mask — True where T² exceeds its control limit."""
        return self.t2 > self.t2_limit

    @property
    def q_flags(self) -> np.ndarray:
        """Boolean mask — True where Q exceeds its control limit."""
        return self.q > self.q_limit

    @property
    def flags(self) -> np.ndarray:
        """Boolean mask — True where either T² or Q flags the sample."""
        return self.t2_flags | self.q_flags

    @property
    def outlier_ids(self) -> list[str]:
        """Sample IDs identified as outliers by T² or Q."""
        return [sid for sid, flag in zip(self.sample_ids, self.flags) if flag]


class PCAOutlierDetector:
    """Detect spectral outliers using PCA-based T² and Q-residuals.

    Fit the model on the calibration set; apply ``flag_outliers`` on any
    split (calibration for leverage detection, test for projection).

    Parameters
    ----------
    n_components:
        Number of PCA components to retain. Choose enough to capture the
        main spectral structure (inspect a scree plot before deciding).
    """

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self._pca: PCA | None = None
        self._n_train: int | None = None
        self._residual_eigenvalues: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, spectra: SpectralTuple) -> PCAOutlierDetector:
        """Fit PCA on calibration spectra.

        A second full-rank decomposition is performed internally to obtain
        the residual eigenvalues required by the Q control-limit formula.

        Parameters
        ----------
        spectra:
            ``(X, axis, sample_ids)`` tuple for the calibration set.

        Returns
        -------
        self
        """
        X, _, _ = spectra
        self._pca = PCA(n_components=self.n_components).fit(X)
        self._n_train = X.shape[0]
        # Residual eigenvalues are needed for the Jackson-Mudholkar Q limit
        full_pca = PCA(svd_solver="full").fit(X)
        self._residual_eigenvalues = full_pca.explained_variance_[self.n_components :]
        return self

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def hotelling_t2(self, spectra: SpectralTuple) -> np.ndarray:
        """Hotelling's T² for each sample.

        .. math::
            T^2_i = \\sum_{k=1}^{p} \\frac{t_{ik}^2}{\\lambda_k}

        where :math:`t_{ik}` are raw PCA scores and :math:`\\lambda_k` are
        the corresponding eigenvalues (explained variances).

        Returns
        -------
        Array of shape ``(n_samples,)``.
        """
        self._require_fit()
        X, _, _ = spectra
        T = self._pca.transform(X)
        return np.sum(T ** 2 / self._pca.explained_variance_, axis=1)

    def q_residuals(self, spectra: SpectralTuple) -> np.ndarray:
        """Q-residuals (squared prediction error) for each sample.

        .. math::
            Q_i = \\|x_i - \\hat{x}_i\\|^2

        where :math:`\\hat{x}_i` is the PCA reconstruction of sample *i*.

        Returns
        -------
        Array of shape ``(n_samples,)``.
        """
        self._require_fit()
        X, _, _ = spectra
        X_hat = self._pca.inverse_transform(self._pca.transform(X))
        return np.sum((X - X_hat) ** 2, axis=1)

    # ------------------------------------------------------------------
    # Control limits
    # ------------------------------------------------------------------

    def t2_limit(self, alpha: float = 0.05) -> float:
        """F-distribution upper control limit for Hotelling's T².

        .. math::
            T^2_\\alpha = \\frac{p(n-1)(n+1)}{n(n-p)} \\cdot F_{\\alpha;\\, p,\\, n-p}

        Parameters
        ----------
        alpha:
            Significance level (default 0.05 → 95 % confidence).
        """
        self._require_fit()
        p = self.n_components
        n = self._n_train
        f_crit = f_dist.ppf(1.0 - alpha, dfn=p, dfd=n - p)
        return float(p * (n - 1) * (n + 1) / (n * (n - p)) * f_crit)

    def q_limit(self, alpha: float = 0.05) -> float:
        """Upper control limit for Q-residuals (Jackson-Mudholkar, 1979).

        Based on the eigenvalues of the discarded (residual) PCA components.
        Falls back to a normal approximation when the distribution is
        degenerate (h₀ ≈ 0).

        Parameters
        ----------
        alpha:
            Significance level (default 0.05 → 95 % confidence).
        """
        self._require_fit()
        lambdas = self._residual_eigenvalues
        if len(lambdas) == 0:
            return 0.0

        theta1 = float(np.sum(lambdas))
        theta2 = float(np.sum(lambdas ** 2))
        theta3 = float(np.sum(lambdas ** 3))
        z = norm.ppf(1.0 - alpha)

        if theta1 == 0.0:
            return 0.0

        h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2 ** 2)

        if abs(h0) < 1e-10:
            # Degenerate case: normal approximation
            return float(theta1 + z * np.sqrt(2.0 * theta2))

        inner = (z * np.sqrt(2.0 * theta2 * h0 ** 2) / theta1
                 + 1.0
                 + theta2 * h0 * (h0 - 1.0) / theta1 ** 2)
        return float(theta1 * inner ** (1.0 / h0))

    # ------------------------------------------------------------------
    # Flagging
    # ------------------------------------------------------------------

    def flag_outliers(
        self,
        spectra: SpectralTuple,
        alpha: float = 0.05,
    ) -> OutlierResult:
        """Compute T², Q and return an OutlierResult for all samples.

        Parameters
        ----------
        spectra:
            SpectralTuple to evaluate (calibration or test set).
        alpha:
            Significance level for the control limits (default 0.05).

        Returns
        -------
        OutlierResult
        """
        _, _, sample_ids = spectra
        return OutlierResult(
            sample_ids=list(sample_ids),
            t2=self.hotelling_t2(spectra),
            q=self.q_residuals(spectra),
            t2_limit=self.t2_limit(alpha),
            q_limit=self.q_limit(alpha),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_fit(self) -> None:
        if self._pca is None:
            raise RuntimeError("Call fit() before using this method.")


# ---------------------------------------------------------------------------
# Cross-block rule
# ---------------------------------------------------------------------------

def cross_block_discard(results: dict[str, OutlierResult]) -> np.ndarray:
    """Union outlier flags across spectral blocks.

    A sample is marked for discard if flagged (by T² or Q) in **any** block.
    All ``OutlierResult`` objects must list samples in the same order.

    Parameters
    ----------
    results:
        Mapping of block name → OutlierResult, e.g.
        ``{'uvvis': res_uv, 'ftir': res_ir, 'hsms': res_ms}``.

    Returns
    -------
    combined_flags:
        Boolean array of length n_samples; ``True`` = discard from all blocks.

    Raises
    ------
    ValueError
        If sample IDs differ across blocks.
    """
    names = list(results.keys())
    result_list = list(results.values())
    ref_ids = result_list[0].sample_ids
    for name, res in zip(names[1:], result_list[1:]):
        if res.sample_ids != ref_ids:
            raise ValueError(
                f"Sample IDs in block '{name}' do not match block '{names[0]}'. "
                "Align sample order before combining."
            )
    return np.logical_or.reduce([r.flags for r in result_list])
