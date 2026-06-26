"""Per-block configuration, preprocessing, and train/test assembly.

This module turns the raw sheets loaded by :func:`olive_oil.data.load_dataset`
into model-ready, sample-aligned multi-block arrays. For each spectral block you
declare:

* a **spectral window** ``region=(start, end)`` (axis units; ``None`` keeps the
  full range), and
* a **preprocessing recipe** ``steps`` — an ordered list of transforms applied
  after region selection.

You may also pass a list of **outlier sample IDs** to discard from the
calibration set (default: none). Per the challenge rules, a sample discarded for
one block is discarded from *all* blocks.

Stateless, per-sample steps (row-profile, log, SNV, Savitzky-Golay derivative)
are applied identically to calibration and test. The one stateful step,
``mean_center``, is fit on the calibration set and the *same* column means are
subtracted from the test set, so no test information leaks into preprocessing.

Example
-------
>>> from olive_oil.data import load_dataset
>>> from olive_oil.blocks import BlockConfig, prepare_blocks
>>> raw = load_dataset("Data/CAC2026_Data_challenge.xlsx")
>>> configs = {
...     "hsms":  BlockConfig(region=(100, 125), steps=["row_profile", "log", "mean_center"]),
...     "mir":   BlockConfig(region=(1040, 795), steps=["snv", "savgol_derivative", "mean_center"]),
...     "uvvis": BlockConfig(region=(580, 1000), steps=["snv", "savgol_derivative", "mean_center"]),
... }
>>> data = prepare_blocks(raw, configs, outliers=[])
>>> Xtr = data.train_blocks()      # BlockSet, ready for GridSearchCV
>>> y = data.y
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import get_spectral_matrix
from .models import BlockSet
from .preprocessing import (
    log_transform,
    mean_center,
    pqn,
    row_profile,
    savgol_derivative,
    select_region,
    snv,
)

SpectralTuple = tuple[np.ndarray, np.ndarray, list]

# Canonical block name -> (calibration sheet key, test sheet key) in the dict
# returned by load_dataset. Configure blocks using these canonical names.
BLOCK_KEYS: dict[str, tuple[str, str]] = {
    "hsms": ("cal_hsms", "test_hsms"),
    "mir": ("cal_ftir", "test_ftir"),
    "uvvis": ("cal_uvvis", "test_uvvis"),
}

# Stateless (per-sample) preprocessing steps: same operation on train and test,
# no fitted state, no leakage. Each maps a step name to its function.
_STATELESS_STEPS = {
    "row_profile": row_profile,
    "log": log_transform,
    "log_transform": log_transform,
    "snv": snv,
    "savgol_derivative": savgol_derivative,
    "savgol": savgol_derivative,
    "derivative": savgol_derivative,
}

# Stateful steps fit parameters on the calibration set and reuse them on test.
_STATEFUL_STEPS = {"mean_center", "pqn"}


@dataclass
class BlockConfig:
    """Spectral window and preprocessing recipe for one block.

    Parameters
    ----------
    region:
        ``(start, end)`` window in axis units (nm, cm⁻¹, m/z). Direction-agnostic
        — works for increasing and decreasing axes. ``None`` keeps the full range.
    steps:
        Ordered preprocessing steps applied after region selection. Each entry is
        either a step name (e.g. ``"snv"``) or a ``(name, kwargs)`` tuple to pass
        keyword arguments, e.g. ``("savgol_derivative", {"deriv": 1, "polyorder": 2,
        "half_window": 3})`` or ``("log", {"base": 10})``.

        Available steps: ``row_profile``, ``log`` (alias ``log_transform``),
        ``snv``, ``savgol_derivative`` (aliases ``savgol``, ``derivative``),
        ``mean_center``.
    """

    region: tuple[float, float] | None = None
    steps: list = field(default_factory=list)


@dataclass
class PreparedData:
    """Sample-aligned, preprocessed multi-block data ready for modeling.

    Attributes
    ----------
    block_names:
        Block order (matches ``X_train`` / ``X_test`` / ``n_components_list``).
    X_train, X_test:
        Lists of 2-D arrays (one per block), rows aligned to ``train_ids`` /
        ``test_ids``.
    y:
        Binary labels aligned to ``train_ids``.
    train_ids, test_ids:
        Sample IDs in row order.
    axes:
        Per-block selected spectral axis (post region selection).
    discarded_ids:
        Calibration sample IDs removed as outliers.
    """

    block_names: list[str]
    X_train: list[np.ndarray]
    X_test: list[np.ndarray]
    y: np.ndarray
    train_ids: list[str]
    test_ids: list[str]
    axes: dict[str, np.ndarray]
    discarded_ids: list[str]

    def train_blocks(self) -> BlockSet:
        """Calibration blocks wrapped as a :class:`BlockSet` for CV/fitting."""
        return BlockSet(self.X_train)

    def test_blocks(self) -> BlockSet:
        """Test blocks wrapped as a :class:`BlockSet`."""
        return BlockSet(self.X_test)

    def summary(self) -> pd.DataFrame:
        """One-row-per-block table of shapes and selected windows."""
        rows = []
        for i, name in enumerate(self.block_names):
            axis = self.axes[name]
            rows.append({
                "block": name,
                "n_variables": self.X_train[i].shape[1],
                "axis_min": float(axis.min()),
                "axis_max": float(axis.max()),
                "n_train": self.X_train[i].shape[0],
                "n_test": self.X_test[i].shape[0],
            })
        return pd.DataFrame(rows).set_index("block")


def _normalise_step(step) -> tuple[str, dict]:
    """Coerce a step spec into ``(name, kwargs)``."""
    if isinstance(step, str):
        return step, {}
    if isinstance(step, (tuple, list)) and len(step) == 2:
        name, kwargs = step
        return name, dict(kwargs)
    raise ValueError(
        f"Invalid step spec {step!r}; use a name or a (name, kwargs) pair."
    )


def _apply_recipe(
    train: SpectralTuple,
    test: SpectralTuple | None,
    steps: list,
) -> tuple[SpectralTuple, SpectralTuple | None]:
    """Apply a preprocessing recipe to train (and optionally test) spectra.

    Stateful steps (``mean_center``) are fit on ``train`` and reused on ``test``.
    """
    for raw_step in steps:
        name, kwargs = _normalise_step(raw_step)

        if name in _STATELESS_STEPS:
            func = _STATELESS_STEPS[name]
            train = func(train, **kwargs)
            if test is not None:
                test = func(test, **kwargs)

        elif name == "mean_center":
            train, center = mean_center(train, **kwargs)
            if test is not None:
                test, _ = mean_center(test, center=center)

        elif name == "pqn":
            train, reference = pqn(train, **kwargs)
            if test is not None:
                test, _ = pqn(test, reference=reference)

        else:
            raise ValueError(
                f"Unknown preprocessing step {name!r}. Available: "
                f"{sorted(set(_STATELESS_STEPS) | _STATEFUL_STEPS)}."
            )
    return train, test


def _matrix_by_id(spectra_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Average replicates and return ``{sample_id: row}`` plus the axis."""
    X, axis, ids = get_spectral_matrix(spectra_df, average=True)
    return {sid: X[i] for i, sid in enumerate(ids)}, axis


def _stack(id_to_row: dict[str, np.ndarray], ids: list[str]) -> np.ndarray:
    """Stack rows for ``ids`` in order into a 2-D array."""
    return np.vstack([id_to_row[sid] for sid in ids])


def prepare_blocks(
    raw: dict[str, pd.DataFrame],
    configs: dict[str, BlockConfig],
    outliers: list[str] | None = None,
    label_col: str = "label",
) -> PreparedData:
    """Build sample-aligned, preprocessed multi-block train/test data.

    Parameters
    ----------
    raw:
        Dict returned by :func:`olive_oil.data.load_dataset`.
    configs:
        Mapping of block name (keys of :data:`BLOCK_KEYS`) to :class:`BlockConfig`.
        Block order in this dict defines the order of ``X_train`` / ``X_test`` and
        therefore the order expected in ``n_components_list``.
    outliers:
        Calibration sample IDs to discard from *all* blocks. Default: none.
        Test samples are never discarded (all must be classified).
    label_col:
        Name of the label column in ``raw['labels']``.

    Returns
    -------
    PreparedData
    """
    if not configs:
        raise ValueError("configs is empty — declare at least one block.")
    outlier_set = {str(s) for s in (outliers or [])}

    labels = raw["labels"].copy()
    labels["sample_id"] = labels["sample_id"].astype(str)
    label_map = dict(zip(labels["sample_id"], labels[label_col].astype(int)))

    block_names = list(configs)
    train_by_block: dict[str, dict[str, np.ndarray]] = {}
    test_by_block: dict[str, dict[str, np.ndarray]] = {}
    raw_axes: dict[str, np.ndarray] = {}

    for name in block_names:
        if name not in BLOCK_KEYS:
            raise ValueError(
                f"Unknown block {name!r}. Known blocks: {sorted(BLOCK_KEYS)}."
            )
        cal_key, test_key = BLOCK_KEYS[name]
        train_by_block[name], raw_axes[name] = _matrix_by_id(raw[cal_key])
        test_by_block[name], _ = _matrix_by_id(raw[test_key])

    # Canonical calibration sample order: label order, present in every block and
    # carrying a label, minus outliers.
    common_train = set.intersection(*(set(d) for d in train_by_block.values()))
    train_ids = [
        sid for sid in labels["sample_id"]
        if sid in common_train and sid in label_map and sid not in outlier_set
    ]
    discarded = sorted(outlier_set & common_train)

    # Canonical test order: first block's order, present in every block.
    common_test = set.intersection(*(set(d) for d in test_by_block.values()))
    _, _, first_test_ids = get_spectral_matrix(
        raw[BLOCK_KEYS[block_names[0]][1]], average=True
    )
    test_ids = [sid for sid in first_test_ids if sid in common_test]

    y = np.array([label_map[sid] for sid in train_ids], dtype=int)

    X_train: list[np.ndarray] = []
    X_test: list[np.ndarray] = []
    axes: dict[str, np.ndarray] = {}

    for name in block_names:
        cfg = configs[name]
        axis = raw_axes[name]

        train_tuple: SpectralTuple = (_stack(train_by_block[name], train_ids), axis, train_ids)
        test_tuple: SpectralTuple = (_stack(test_by_block[name], test_ids), axis, test_ids)

        if cfg.region is not None:
            start, end = cfg.region
            train_tuple = select_region(train_tuple, start, end)
            test_tuple = select_region(test_tuple, start, end)

        train_tuple, test_tuple = _apply_recipe(train_tuple, test_tuple, cfg.steps)

        X_train.append(train_tuple[0])
        X_test.append(test_tuple[0])
        axes[name] = train_tuple[1]

    return PreparedData(
        block_names=block_names,
        X_train=X_train,
        X_test=X_test,
        y=y,
        train_ids=train_ids,
        test_ids=test_ids,
        axes=axes,
        discarded_ids=discarded,
    )
