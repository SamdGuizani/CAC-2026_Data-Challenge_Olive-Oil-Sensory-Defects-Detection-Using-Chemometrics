"""Nested cross-validation and reporting for the fusion classifier.

The challenge is scored by **F1**. To get an honest F1 estimate while also tuning
hyperparameters, we use *nested* CV:

* an **inner** ``GridSearchCV`` searches the hyperparameter grid (block PLS
  components + final-classifier params) on each outer training split, and
* an **outer** ``cross_validate`` measures F1/precision/recall on the held-out
  outer fold — folds the inner search never saw.

The reported mean ± std over outer folds is the unbiased performance estimate.
Separately, :func:`tune_final_model` runs a single ``GridSearchCV`` over *all*
calibration data to pick the hyperparameters and refit the model used for the
actual test-set predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from .models import BlockSet, as_blockset

# Metrics reported for every nested-CV run. F1 is the challenge score; precision
# and recall are tracked because class imbalance (~34% musty) makes them
# informative about the precision/recall trade-off behind the F1.
DEFAULT_SCORING = {"f1": "f1", "precision": "precision", "recall": "recall"}


def make_cv(n_splits: int = 5, *, shuffle: bool = True, random_state: int | None = 0) -> StratifiedKFold:
    """Stratified k-fold splitter (stratification preserves class balance)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def nested_cv(
    model,
    param_grid: dict,
    blocks,
    y,
    *,
    outer_cv=None,
    inner_cv=None,
    scoring: dict | None = None,
    refit_metric: str = "f1",
    inner_n_jobs: int = -1,
    outer_n_jobs: int = 1,
    return_estimators: bool = False,
):
    """Run nested CV and return per-fold scores plus the chosen inner params.

    Parameters
    ----------
    model:
        A :class:`~olive_oil.models.MidLevelFusionClassifier` instance (with its
        final ``classifier`` set).
    param_grid:
        Grid passed to the inner ``GridSearchCV``. Keys are
        ``"n_components_list"`` and ``"classifier__<param>"`` entries.
    blocks:
        Multi-block calibration data (``BlockSet`` or list of arrays).
    y:
        Binary labels.
    outer_cv, inner_cv:
        Splitters; default to 5-fold and 3-fold stratified.
    scoring:
        Metric dict; defaults to F1/precision/recall.
    refit_metric:
        Metric the inner search optimises (default ``"f1"``).
    inner_n_jobs, outer_n_jobs:
        Parallelism. Inner runs parallel by default; outer serial to avoid
        oversubscription. Swap if you prefer parallel outer folds.
    return_estimators:
        If True, include the fitted inner searches (one per outer fold).

    Returns
    -------
    dict with keys ``raw`` (the ``cross_validate`` output), ``summary``
    (per-metric mean/std DataFrame), and ``best_params`` (the inner-search
    winner per outer fold).
    """
    blocks = as_blockset(blocks)
    y = np.asarray(y)
    outer_cv = outer_cv if outer_cv is not None else make_cv(5, random_state=0)
    inner_cv = inner_cv if inner_cv is not None else make_cv(3, random_state=1)
    scoring = scoring if scoring is not None else DEFAULT_SCORING

    inner_search = GridSearchCV(
        model, param_grid, cv=inner_cv, scoring=refit_metric,
        n_jobs=inner_n_jobs, refit=True,
    )

    raw = cross_validate(
        inner_search, blocks, y,
        cv=outer_cv, scoring=scoring,
        return_train_score=True,
        return_estimator=True,  # always kept so we can read each fold's params
        n_jobs=outer_n_jobs,
        error_score="raise",
    )

    best_params = [est.best_params_ for est in raw["estimator"]]
    summary = _summarize(raw, scoring)

    if not return_estimators:
        raw = {k: v for k, v in raw.items() if k != "estimator"}

    return {"raw": raw, "summary": summary, "best_params": best_params}


def _summarize(raw: dict, scoring: dict) -> pd.DataFrame:
    """Mean/std of train & test scores per metric across outer folds."""
    rows = []
    for metric in scoring:
        test = raw[f"test_{metric}"]
        train = raw[f"train_{metric}"]
        rows.append({
            "metric": metric,
            "test_mean": test.mean(),
            "test_std": test.std(),
            "train_mean": train.mean(),
            "train_std": train.std(),
        })
    return pd.DataFrame(rows).set_index("metric")


def tune_final_model(
    model,
    param_grid: dict,
    blocks,
    y,
    *,
    cv=None,
    scoring: str = "f1",
    n_jobs: int = -1,
) -> GridSearchCV:
    """Single ``GridSearchCV`` over all calibration data; refits the best model.

    Use the returned (already-refit) search for test-set predictions. Its
    ``best_score_`` is the inner-CV F1 of the winning config — optimistic
    relative to :func:`nested_cv`, which is why both are reported.
    """
    blocks = as_blockset(blocks)
    cv = cv if cv is not None else make_cv(5, random_state=0)
    search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, refit=True,
    )
    search.fit(blocks, np.asarray(y))
    return search


def predict_test(estimator, test_blocks, test_ids) -> pd.DataFrame:
    """Predict the unlabeled test set and return a ``sample_id``/``prediction`` table.

    ``estimator`` may be a fitted ``GridSearchCV`` (uses its best estimator) or a
    fitted fusion classifier.
    """
    test_blocks = as_blockset(test_blocks)
    y_pred = estimator.predict(test_blocks)
    out = pd.DataFrame({"sample_id": list(test_ids), "prediction": np.asarray(y_pred).astype(int)})
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(test_blocks)
        out["proba_musty"] = proba[:, 1]
    return out
