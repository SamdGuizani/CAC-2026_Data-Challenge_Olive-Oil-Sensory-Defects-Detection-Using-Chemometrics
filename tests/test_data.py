"""Tests for olive_oil.data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from olive_oil.data import average_replicates, get_spectral_matrix, load_dataset

DATA_FILE = Path(__file__).parent.parent / "Data" / "CAC2026_Data_challenge.xlsx"

EXPECTED_KEYS = {
    "labels", "metadata",
    "cal_uvvis", "cal_ftir", "cal_hsms",
    "test_uvvis", "test_ftir", "test_hsms",
}


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

def test_load_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nonexistent.xlsx")


@pytest.mark.integration
@pytest.mark.skipif(not DATA_FILE.exists(), reason="real data file not available")
def test_load_dataset_returns_all_keys():
    data = load_dataset(DATA_FILE)
    assert set(data.keys()) == EXPECTED_KEYS


@pytest.mark.integration
@pytest.mark.skipif(not DATA_FILE.exists(), reason="real data file not available")
def test_load_dataset_spectral_shapes():
    data = load_dataset(DATA_FILE)
    # +1 for sample_id column
    assert data["cal_uvvis"].shape[1] == 701 + 1
    assert data["cal_ftir"].shape[1] == 549 + 1
    assert data["cal_hsms"].shape[1] == 301 + 1


@pytest.mark.integration
@pytest.mark.skipif(not DATA_FILE.exists(), reason="real data file not available")
def test_load_dataset_spectral_columns_are_float():
    data = load_dataset(DATA_FILE)
    for key in ("cal_uvvis", "cal_ftir", "cal_hsms"):
        spectral_cols = [c for c in data[key].columns if c != "sample_id"]
        assert all(isinstance(c, float) for c in spectral_cols), key


@pytest.mark.integration
@pytest.mark.skipif(not DATA_FILE.exists(), reason="real data file not available")
def test_load_dataset_sample_id_is_str():
    data = load_dataset(DATA_FILE)
    for key in ("cal_uvvis", "cal_ftir", "cal_hsms"):
        assert pd.api.types.is_string_dtype(data[key]["sample_id"]), key


@pytest.mark.integration
@pytest.mark.skipif(not DATA_FILE.exists(), reason="real data file not available")
def test_load_dataset_labels_binary():
    data = load_dataset(DATA_FILE)
    assert set(data["labels"]["label"].unique()).issubset({0, 1})


@pytest.mark.integration
@pytest.mark.skipif(not DATA_FILE.exists(), reason="real data file not available")
def test_load_dataset_replicates_counts():
    data = load_dataset(DATA_FILE)
    n_labels = len(data["labels"])
    assert len(data["cal_uvvis"]) == n_labels * 2
    assert len(data["cal_ftir"]) == n_labels * 3
    assert len(data["cal_hsms"]) == n_labels * 2


# ---------------------------------------------------------------------------
# average_replicates
# ---------------------------------------------------------------------------

def test_average_replicates_one_row_per_sample(spectra_2rep):
    averaged = average_replicates(spectra_2rep)
    assert len(averaged) == spectra_2rep["sample_id"].nunique()


def test_average_replicates_preserves_first_appearance_order():
    # Sample IDs whose lexicographic order differs from first-appearance order
    df = pd.DataFrame({
        "sample_id": ["S10", "S10", "S2", "S2", "S1", "S1"],
        300.0: [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    })
    averaged = average_replicates(df)
    assert list(averaged.index) == ["S10", "S2", "S1"]


def test_average_replicates_index_name(spectra_2rep):
    averaged = average_replicates(spectra_2rep)
    assert averaged.index.name == "sample_id"


def test_average_replicates_correct_values(spectra_known_values):
    averaged = average_replicates(spectra_known_values)
    assert averaged.loc["S01", 300.0] == pytest.approx(3.0)
    assert averaged.loc["S01", 400.0] == pytest.approx(7.0)
    assert averaged.loc["S02", 300.0] == pytest.approx(2.0)
    assert averaged.loc["S02", 400.0] == pytest.approx(6.0)


def test_average_replicates_three_reps(spectra_3rep):
    averaged = average_replicates(spectra_3rep)
    assert len(averaged) == spectra_3rep["sample_id"].nunique()


def test_average_replicates_no_sample_id_column(spectra_2rep):
    averaged = average_replicates(spectra_2rep)
    assert "sample_id" not in averaged.columns


# ---------------------------------------------------------------------------
# get_spectral_matrix
# ---------------------------------------------------------------------------

def test_get_spectral_matrix_shapes(spectra_2rep):
    X, axis, sample_ids = get_spectral_matrix(spectra_2rep, average=True)
    n_samples = spectra_2rep["sample_id"].nunique()
    n_vars = len([c for c in spectra_2rep.columns if c != "sample_id"])
    assert X.shape == (n_samples, n_vars)
    assert axis.shape == (n_vars,)
    assert len(sample_ids) == n_samples


def test_get_spectral_matrix_returns_ndarray(spectra_2rep):
    X, axis, sample_ids = get_spectral_matrix(spectra_2rep)
    assert isinstance(X, np.ndarray)
    assert isinstance(axis, np.ndarray)
    assert isinstance(sample_ids, list)


def test_get_spectral_matrix_axis_sorted(spectra_2rep):
    _, axis, _ = get_spectral_matrix(spectra_2rep)
    assert list(axis) == sorted(axis)


def test_get_spectral_matrix_no_average_keeps_all_rows(spectra_2rep):
    X, _, sample_ids = get_spectral_matrix(spectra_2rep, average=False)
    assert X.shape[0] == len(spectra_2rep)
    assert len(sample_ids) == len(spectra_2rep)


def test_get_spectral_matrix_average_reduces_rows(spectra_2rep):
    X_avg, _, _ = get_spectral_matrix(spectra_2rep, average=True)
    X_raw, _, _ = get_spectral_matrix(spectra_2rep, average=False)
    assert X_avg.shape[0] < X_raw.shape[0]


def test_get_spectral_matrix_dtype_float(spectra_2rep):
    X, axis, _ = get_spectral_matrix(spectra_2rep)
    assert X.dtype == float
    assert axis.dtype == float
