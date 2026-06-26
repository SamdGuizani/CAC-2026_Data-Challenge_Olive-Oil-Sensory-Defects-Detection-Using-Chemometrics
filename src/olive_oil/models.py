"""Mid-level PLS-DA data-fusion models for olive oil musty-defect classification.

CAC2026 Data Challenge — see PROJECT_CONTEXT.md for full background.

Architecture::

    X_ms  ──PLS-DA(LV_ms)───►  T_ms  ┐
    X_mir ──PLS-DA(LV_mir)──►  T_mir ├─ hstack ─► T_fused ─► [scale] ─► classifier ─► y_pred
    X_uv  ──PLS-DA(LV_uv)───►  T_uv  ┘

Each block's ``PLSScorer`` performs supervised dimensionality reduction (it is fit
with the binary label, so its X-scores already encode class-discriminative
structure). The scores from all blocks are concatenated and handed to a swappable
final ``classifier`` — by default a :class:`PLSDAClassifier`, but any sklearn
classifier (SVC, LDA, QDA, RandomForest, XGBoost, ...) can be plugged in without
changing the fusion code.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


# ── BlockSet: makes a list of multi-block arrays indexable/splittable ────────
# GridSearchCV / cross_validate expect X to support len(X) == n_samples and
# fancy indexing X[train_idx]. A plain Python list of 3 arrays fails both
# (len() gives 3, not n_samples). BlockSet wraps the blocks in a thin
# array-like that indexes each block consistently along axis 0.

class BlockSet:
    """Wrap multiple same-``n_samples`` blocks so sklearn CV treats them as one X.

    Usage::

        blocks = BlockSet([X_ms, X_mir, X_uv])
        blocks[train_idx]  -> BlockSet of the same blocks, row-subset
        len(blocks)        -> n_samples (NOT n_blocks)

    Always wrap multi-block data as ``BlockSet`` before passing to any
    ``GridSearchCV`` / ``cross_validate`` call; a plain list is only safe for
    one-off ``.fit`` / ``.predict`` outside of CV.
    """

    def __init__(self, blocks):
        self.blocks = list(blocks)
        n = self.blocks[0].shape[0]
        assert all(b.shape[0] == n for b in self.blocks), \
            "All blocks must have the same number of samples (rows)."
        self._n = n

    def __len__(self):
        return self._n

    @property
    def shape(self):
        # Reporting only n_samples on axis 0 is enough for sklearn's array-like
        # detection in _safe_indexing / indexable().
        return (self._n,)

    @property
    def ndim(self):
        return 1

    def __getitem__(self, idx):
        return BlockSet([b[idx] for b in self.blocks])

    def __iter__(self):
        return iter(self.blocks)


def as_blockset(blocks) -> BlockSet:
    """Return ``blocks`` as a :class:`BlockSet` (pass-through if already one)."""
    return blocks if isinstance(blocks, BlockSet) else BlockSet(blocks)


# ── PLSScorer: supervised dimensionality reducer for one block ───────────────

class PLSScorer(BaseEstimator, TransformerMixin):
    """Wrap ``PLSRegression`` as a transformer that returns X-scores.

    ``fit(X, y)`` learns discriminant latent directions; ``transform(X)`` returns
    the X-scores ``T`` of shape ``(n_samples, n_components)``.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.pls_ = PLSRegression(n_components=self.n_components)
        self.pls_.fit(X, y)
        return self

    def transform(self, X):
        # sklearn >= 1.x: PLSRegression.transform(X) returns a single array,
        # NOT a (T, _) tuple — do not unpack.
        return self.pls_.transform(X)


# ── PLSDAClassifier: PLS regression + threshold = a binary classifier ────────

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """PLS-DA classifier for a binary 0/1 target.

    Fits a :class:`PLSRegression` against the binary label and thresholds the
    continuous prediction. This is the default final classifier for the fused
    score space, mirroring the reference paper's ``3MLpls`` strategy (block
    PLS-DA scores fused, then a final PLS-DA on the fused scores).

    Parameters
    ----------
    n_components:
        Number of latent variables for the fusion-level PLS.
    threshold:
        Decision threshold on the regression output (default 0.5).

    Notes
    -----
    Assumes labels are ``{0, 1}``. ``predict_proba`` returns the regression
    output clipped to ``[0, 1]`` as a pseudo-probability for the positive class —
    adequate for ranking / threshold tuning, not a calibrated probability.
    """

    def __init__(self, n_components=2, threshold=0.5):
        self.n_components = n_components
        self.threshold = threshold

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(
                f"PLSDAClassifier supports binary targets only; got "
                f"classes {self.classes_!r}."
            )
        self.pls_ = PLSRegression(n_components=self.n_components).fit(X, y)
        return self

    def decision_function(self, X):
        """Continuous PLS regression output, one value per sample."""
        return np.asarray(self.pls_.predict(X)).ravel()

    def predict(self, X):
        scores = self.decision_function(X)
        pos = self.classes_[1]
        neg = self.classes_[0]
        return np.where(scores >= self.threshold, pos, neg)

    def predict_proba(self, X):
        scores = np.clip(self.decision_function(X), 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])


# ── MidLevelFusionTransformer: fuse N blocks into one score matrix ───────────

class MidLevelFusionTransformer(BaseEstimator, TransformerMixin):
    """Fit one :class:`PLSScorer` per block and hstack the scores.

    Parameters
    ----------
    n_components_list : tuple of int
        Number of PLS components for each block. Length must match the number of
        blocks. Must be a TUPLE (not list) for ``GridSearchCV`` hashing.
    """

    def __init__(self, n_components_list=(2, 3, 3)):
        self.n_components_list = n_components_list

    def fit(self, blocks, y):
        block_list = list(as_blockset(blocks))
        if len(self.n_components_list) != len(block_list):
            raise ValueError(
                f"n_components_list has {len(self.n_components_list)} entries "
                f"but {len(block_list)} blocks were passed."
            )
        self.scorers_ = [
            PLSScorer(n_components=nc).fit(X, y)
            for nc, X in zip(self.n_components_list, block_list)
        ]
        return self

    def transform(self, blocks):
        block_list = list(as_blockset(blocks))
        return np.hstack([
            scorer.transform(X)
            for scorer, X in zip(self.scorers_, block_list)
        ])


# ── MidLevelFusionClassifier: end-to-end estimator ───────────────────────────

class MidLevelFusionClassifier(BaseEstimator, ClassifierMixin):
    """End-to-end mid-level data-fusion classifier (GridSearchCV-compatible).

    Parameters
    ----------
    n_components_list : tuple of int
        PLS components per block. Tunable in the CV grid.
    classifier : sklearn estimator or None
        Final classifier on the fused scores. Default: :class:`PLSDAClassifier`.
        Swap freely for SVC, LDA, QDA, RandomForestClassifier, XGBClassifier, ...
        — keep this pluggable, do not hardcode a specific classifier downstream.
    scale_fused : bool
        StandardScale the fused score matrix before the final classifier.
        Recommended when blocks differ in scale.

    Final-classifier hyperparameters are set via ``classifier__<param>`` in
    ``GridSearchCV`` grids, following the standard sklearn nested convention.
    """

    def __init__(self, n_components_list=(2, 3, 3), classifier=None, scale_fused=True):
        self.n_components_list = n_components_list
        self.classifier = classifier
        self.scale_fused = scale_fused

    def _get_classifier(self):
        return self.classifier if self.classifier is not None else PLSDAClassifier()

    def fit(self, blocks, y):
        self.fusion_ = MidLevelFusionTransformer(n_components_list=self.n_components_list)
        T = self.fusion_.fit_transform(blocks, y)

        if self.scale_fused:
            self.scaler_ = StandardScaler()
            T = self.scaler_.fit_transform(T)

        self.classifier_ = clone(self._get_classifier())
        self.classifier_.fit(T, y)
        self.classes_ = self.classifier_.classes_
        return self

    def _fused(self, blocks):
        T = self.fusion_.transform(blocks)
        if self.scale_fused:
            T = self.scaler_.transform(T)
        return T

    def predict(self, blocks):
        return self.classifier_.predict(self._fused(blocks))

    def predict_proba(self, blocks):
        return self.classifier_.predict_proba(self._fused(blocks))

    def decision_function(self, blocks):
        return self.classifier_.decision_function(self._fused(blocks))

    def score(self, blocks, y):
        return (self.predict(blocks) == y).mean()

    # ── sklearn param routing (get/set_params) ──────────────────────────────
    def get_params(self, deep=True):
        params = {
            "n_components_list": self.n_components_list,
            "classifier": self.classifier,
            "scale_fused": self.scale_fused,
        }
        if deep and self.classifier is not None:
            for k, v in self.classifier.get_params(deep=True).items():
                params[f"classifier__{k}"] = v
        return params

    def set_params(self, **params):
        clf_params = {}
        for k, v in params.items():
            if k.startswith("classifier__"):
                clf_params[k[len("classifier__"):]] = v
            else:
                setattr(self, k, v)
        if clf_params and self.classifier is not None:
            self.classifier.set_params(**clf_params)
        return self
