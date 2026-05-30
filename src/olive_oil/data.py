"""Data loading for CAC2026 olive oil dataset."""

from pathlib import Path
import numpy as np
import pandas as pd

# Sheet names as constants to avoid magic strings
SHEET_LABELS = "CAL Reference values"
SHEET_META = "CAL metadata"
SHEET_CAL_UVVIS = "CAL UV-Vis"
SHEET_CAL_FTIR = "CAL ATR-FTIR"
SHEET_CAL_HSMS = "CAL HS-MS"
SHEET_TEST_UVVIS = "TEST UV-Vis"
SHEET_TEST_FTIR = "TEST ATR-FTIR"
SHEET_TEST_HSMS = "TEST HS-MS"


def load_dataset(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load all sheets from the CAC2026 xlsx file.

    Each spectral sheet is returned as-is (replicates kept, sample ID in
    the first column). Labels and metadata are returned as separate frames.

    Parameters
    ----------
    path:
        Path to CAC2026_Data_challenge.xlsx.

    Returns
    -------
    dict with keys:
        'labels'     — DataFrame(sample_id, label)
        'metadata'   — DataFrame with calibration metadata
        'cal_uvvis'  — DataFrame with CAL UV-Vis spectra (2 replicates/sample)
        'cal_ftir'   — DataFrame with CAL ATR-FTIR spectra (3 replicates/sample)
        'cal_hsms'   — DataFrame with CAL HS-MS spectra (2 replicates/sample)
        'test_uvvis' — DataFrame with TEST UV-Vis spectra
        'test_ftir'  — DataFrame with TEST ATR-FTIR spectra
        'test_hsms'  — DataFrame with TEST HS-MS spectra
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    xl = pd.ExcelFile(path, engine="openpyxl")

    labels = _load_labels(xl)
    metadata = _load_metadata(xl)
    cal_uvvis = _load_spectra(xl, SHEET_CAL_UVVIS)
    cal_ftir = _load_spectra(xl, SHEET_CAL_FTIR)
    cal_hsms = _load_spectra(xl, SHEET_CAL_HSMS)
    test_uvvis = _load_spectra(xl, SHEET_TEST_UVVIS)
    test_ftir = _load_spectra(xl, SHEET_TEST_FTIR)
    test_hsms = _load_spectra(xl, SHEET_TEST_HSMS)

    return {
        "labels": labels,
        "metadata": metadata,
        "cal_uvvis": cal_uvvis,
        "cal_ftir": cal_ftir,
        "cal_hsms": cal_hsms,
        "test_uvvis": test_uvvis,
        "test_ftir": test_ftir,
        "test_hsms": test_hsms,
    }


def _load_labels(xl: pd.ExcelFile) -> pd.DataFrame:
    """Parse the labels sheet, keeping only sample ID and binary label."""
    df = xl.parse(SHEET_LABELS, usecols=[0, 1], header=0)
    df.columns = ["sample_id", "label"]
    df["sample_id"] = df["sample_id"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def _load_metadata(xl: pd.ExcelFile) -> pd.DataFrame:
    """Parse the metadata sheet, normalising the first column to 'sample_id'."""
    df = xl.parse(SHEET_META, header=0)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def _load_spectra(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """Parse a spectral sheet.

    The first column is the sample ID; remaining columns are spectral
    variables. Column headers (wavelengths / wavenumbers / m/z) are parsed
    from the first row.
    """
    df = xl.parse(sheet, header=0)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str)
    df.columns = ["sample_id"] + [float(c) for c in df.columns[1:]]
    return df


def average_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """Average replicate measurements per sample.

    Parameters
    ----------
    df:
        Spectral DataFrame with a 'sample_id' column and replicate rows.

    Returns
    -------
    DataFrame with one row per sample (mean across replicates), sample_id
    set as the index.
    """
    numeric_cols = df.columns.drop("sample_id")
    return (
        df.groupby("sample_id", sort=False)[numeric_cols]
        .mean()
        .rename_axis("sample_id")
    )


def get_spectral_matrix(
    df: pd.DataFrame,
    average: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract a numeric spectral matrix from a spectral DataFrame.

    Parameters
    ----------
    df:
        Spectral DataFrame as returned by load_dataset.
    average:
        If True, average replicates before returning.

    Returns
    -------
    X:
        2-D array of shape (n_samples, n_variables).
    axis:
        1-D array of spectral axis values (wavelengths / wavenumbers / m/z)
        parsed from column names.
    sample_ids:
        List of sample ID strings, aligned with rows of X.
    """
    if average:
        df = average_replicates(df).reset_index()

    numeric_cols = [c for c in df.columns if c != "sample_id"]
    X = df[numeric_cols].to_numpy(dtype=float)
    axis = np.array([float(c) for c in numeric_cols])
    sample_ids = df["sample_id"].tolist()
    return X, axis, sample_ids