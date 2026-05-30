"""Shared fixtures for olive_oil tests. No dependency on the real data file."""

import numpy as np
import pandas as pd
import pytest

# Spectral axis values used across fixtures
AXIS_5 = [300.0, 400.0, 500.0, 600.0, 700.0]


def _make_spectral_df(n_samples: int, n_replicates: int, axis=AXIS_5, seed=0) -> pd.DataFrame:
    """Build a synthetic spectral DataFrame with replicates."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_samples):
        for _ in range(n_replicates):
            row = {"sample_id": f"S{i+1:02d}"}
            row.update({float(w): rng.uniform(0, 1) for w in axis})
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def spectra_2rep():
    """3 samples × 2 replicates, 5 spectral variables (UV-Vis / HS-MS style)."""
    return _make_spectral_df(n_samples=3, n_replicates=2)


@pytest.fixture
def spectra_3rep():
    """3 samples × 3 replicates, 5 spectral variables (FTIR style)."""
    return _make_spectral_df(n_samples=3, n_replicates=3)


@pytest.fixture
def spectra_known_values():
    """Minimal DataFrame with known values to verify averaging arithmetic."""
    return pd.DataFrame({
        "sample_id": ["S01", "S01", "S02", "S02"],
        300.0: [2.0, 4.0, 1.0, 3.0],
        400.0: [6.0, 8.0, 5.0, 7.0],
    })


@pytest.fixture
def labels_df():
    """Synthetic labels DataFrame matching spectra_2rep sample IDs."""
    return pd.DataFrame({
        "sample_id": ["S01", "S02", "S03"],
        "label": [0, 1, 0],
    })
