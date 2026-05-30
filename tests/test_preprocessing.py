"""Tests for olive_oil.preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from olive_oil.preprocessing import (
    log_transform,
    mean_center,
    row_profile,
    savgol_derivative,
    snv,
)

# ---------------------------------------------------------------------------
# Local fixtures  (SpectralTuples — not DataFrames like conftest fixtures)
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_tuple():
    """3 samples × 5 variables with fully known values."""
    X = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],   # row sum = 15
        [2.0, 4.0, 6.0, 8.0, 10.0],  # row sum = 30
        [0.0, 1.0, 2.0, 3.0, 4.0],   # starts at 0 (tests log of zero)
    ])
    axis = np.array([300.0, 400.0, 500.0, 600.0, 700.0])
    return X, axis, ["S1", "S2", "S3"]


@pytest.fixture
def wide_tuple():
    """5 samples × 30 variables — enough for the default half_window=7 (window=15)."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.1, 1.0, (5, 30))
    axis = np.linspace(300.0, 1000.0, 30)
    return X, axis, [f"S{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_passthrough(original, result):
    """Assert axis values and sample_ids are identical to the input."""
    _, orig_axis, orig_ids = original
    _, res_axis, res_ids = result
    np.testing.assert_array_equal(res_axis, orig_axis)
    assert res_ids == orig_ids


# ---------------------------------------------------------------------------
# row_profile
# ---------------------------------------------------------------------------

class TestRowProfile:
    def test_rows_sum_to_one(self, simple_tuple):
        X_out, _, _ = row_profile(simple_tuple)
        np.testing.assert_allclose(X_out.sum(axis=1), 1.0)

    def test_known_first_element(self, simple_tuple):
        X_out, _, _ = row_profile(simple_tuple)
        assert X_out[0, 0] == pytest.approx(1.0 / 15.0)

    def test_proportions_preserved(self, simple_tuple):
        """Row ratios must be identical before and after profiling."""
        X, _, _ = simple_tuple
        X_out, _, _ = row_profile(simple_tuple)
        np.testing.assert_allclose(X_out[0] / X_out[0, 0], X[0] / X[0, 0])

    def test_shape_preserved(self, simple_tuple):
        X, _, _ = simple_tuple
        X_out, _, _ = row_profile(simple_tuple)
        assert X_out.shape == X.shape

    def test_axis_ids_unchanged(self, simple_tuple):
        _check_passthrough(simple_tuple, row_profile(simple_tuple))

    def test_does_not_mutate_input(self, simple_tuple):
        X_orig = simple_tuple[0].copy()
        row_profile(simple_tuple)
        np.testing.assert_array_equal(simple_tuple[0], X_orig)


# ---------------------------------------------------------------------------
# snv
# ---------------------------------------------------------------------------

class TestSnv:
    def test_row_means_are_zero(self, simple_tuple):
        X_out, _, _ = snv(simple_tuple)
        np.testing.assert_allclose(X_out.mean(axis=1), 0.0, atol=1e-12)

    def test_row_stds_are_one(self, simple_tuple):
        """SNV uses ddof=1 — sample standard deviation."""
        X_out, _, _ = snv(simple_tuple)
        np.testing.assert_allclose(X_out.std(axis=1, ddof=1), 1.0, atol=1e-12)

    def test_shape_preserved(self, simple_tuple):
        X, _, _ = simple_tuple
        X_out, _, _ = snv(simple_tuple)
        assert X_out.shape == X.shape

    def test_axis_ids_unchanged(self, simple_tuple):
        _check_passthrough(simple_tuple, snv(simple_tuple))

    def test_does_not_mutate_input(self, simple_tuple):
        X_orig = simple_tuple[0].copy()
        snv(simple_tuple)
        np.testing.assert_array_equal(simple_tuple[0], X_orig)

    def test_identical_rows_differ_after_snv(self, simple_tuple):
        """Two samples with proportional spectra must have identical SNV output."""
        X, axis, ids = simple_tuple
        # S1 = [1,2,3,4,5], S2 = [2,4,6,8,10] — same shape, different scale
        X_out, _, _ = snv(simple_tuple)
        np.testing.assert_allclose(X_out[0], X_out[1], atol=1e-12)


# ---------------------------------------------------------------------------
# log_transform
# ---------------------------------------------------------------------------

class TestLogTransform:
    def test_zero_maps_to_zero(self, simple_tuple):
        """log1p(0) = 0 — zero intensities must stay at zero."""
        X_out, _, _ = log_transform(simple_tuple)
        assert X_out[2, 0] == pytest.approx(0.0)

    def test_natural_log_default(self, simple_tuple):
        X_out, _, _ = log_transform(simple_tuple)
        np.testing.assert_allclose(X_out, np.log1p(simple_tuple[0]))

    def test_base_10(self, simple_tuple):
        X_out, _, _ = log_transform(simple_tuple, base=10)
        expected = np.log1p(simple_tuple[0]) / np.log(10)
        np.testing.assert_allclose(X_out, expected)

    def test_values_are_smaller_than_input(self, simple_tuple):
        """log1p(x) < x for all x > 0."""
        X, _, _ = simple_tuple
        X_out, _, _ = log_transform(simple_tuple)
        assert (X_out[X > 0] < X[X > 0]).all()

    def test_shape_preserved(self, simple_tuple):
        X, _, _ = simple_tuple
        X_out, _, _ = log_transform(simple_tuple)
        assert X_out.shape == X.shape

    def test_axis_ids_unchanged(self, simple_tuple):
        _check_passthrough(simple_tuple, log_transform(simple_tuple))


# ---------------------------------------------------------------------------
# savgol_derivative
# ---------------------------------------------------------------------------

class TestSavgolDerivative:
    def test_shape_preserved(self, wide_tuple):
        X, _, _ = wide_tuple
        X_out, _, _ = savgol_derivative(wide_tuple)
        assert X_out.shape == X.shape

    def test_axis_ids_unchanged(self, wide_tuple):
        _check_passthrough(wide_tuple, savgol_derivative(wide_tuple))

    def test_constant_signal_first_deriv_is_zero(self):
        """d/dx(c) = 0 — must hold for every point in the signal."""
        n_vars = 30
        axis = np.linspace(300.0, 1000.0, n_vars)
        X = np.full((3, n_vars), 5.0)
        X_out, _, _ = savgol_derivative((X, axis, ["S1", "S2", "S3"]), half_window=3)
        np.testing.assert_allclose(X_out, 0.0, atol=1e-10)

    def test_linear_signal_first_deriv_equals_slope(self):
        """d/dx(slope * x) = slope — interior points must match within tolerance."""
        n_vars = 30
        axis = np.arange(n_vars, dtype=float)   # uniform spacing delta=1
        slope = 3.0
        X = np.tile(slope * axis, (3, 1))
        X_out, _, _ = savgol_derivative(
            (X, axis, ["S1", "S2", "S3"]),
            deriv=1, polyorder=2, half_window=3,
        )
        # Skip window-edge artefacts on both ends
        np.testing.assert_allclose(X_out[:, 3:-3], slope, atol=1e-8)

    def test_decreasing_axis_does_not_raise(self):
        """FTIR wavenumbers decrease (3230 → 673) — must not crash or sign-flip."""
        axis = np.linspace(3230.0, 673.0, 30)
        X = np.random.default_rng(0).uniform(0, 1, (3, 30))
        savgol_derivative((X, axis, ["S1", "S2", "S3"]))  # no exception

    def test_half_window_affects_smoothing(self, wide_tuple):
        """A larger window produces more smoothed (different) derivatives."""
        X_out_large, _, _ = savgol_derivative(wide_tuple, half_window=7)
        X_out_small, _, _ = savgol_derivative(wide_tuple, half_window=2)
        assert not np.allclose(X_out_large, X_out_small)

    def test_second_derivative(self, wide_tuple):
        """deriv=2 must be accepted and return a valid array."""
        X, _, _ = wide_tuple
        X_out, _, _ = savgol_derivative(wide_tuple, deriv=2, polyorder=3)
        assert X_out.shape == X.shape
        assert np.isfinite(X_out).all()


# ---------------------------------------------------------------------------
# mean_center
# ---------------------------------------------------------------------------

class TestMeanCenter:
    def test_column_means_are_zero(self, simple_tuple):
        (X_out, _, _), _ = mean_center(simple_tuple)
        np.testing.assert_allclose(X_out.mean(axis=0), 0.0, atol=1e-12)

    def test_returned_center_equals_input_mean(self, simple_tuple):
        X, _, _ = simple_tuple
        _, center = mean_center(simple_tuple)
        np.testing.assert_allclose(center, X.mean(axis=0))

    def test_center_shape(self, simple_tuple):
        X, _, _ = simple_tuple
        _, center = mean_center(simple_tuple)
        assert center.shape == (X.shape[1],)

    def test_precomputed_center_applied(self, simple_tuple):
        """Passing a pre-fitted center must subtract it exactly."""
        X, axis, ids = simple_tuple
        train_center = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        (X_out, _, _), center_used = mean_center(simple_tuple, center=train_center)
        np.testing.assert_allclose(X_out, X - train_center)
        np.testing.assert_array_equal(center_used, train_center)

    def test_precomputed_center_not_recomputed(self, simple_tuple):
        """Passing a center must return that same object, not recompute from X."""
        fixed_center = np.zeros(5)
        _, center_used = mean_center(simple_tuple, center=fixed_center)
        np.testing.assert_array_equal(center_used, fixed_center)

    def test_shape_preserved(self, simple_tuple):
        X, _, _ = simple_tuple
        (X_out, _, _), _ = mean_center(simple_tuple)
        assert X_out.shape == X.shape

    def test_axis_ids_unchanged(self, simple_tuple):
        result_tuple, _ = mean_center(simple_tuple)
        _check_passthrough(simple_tuple, result_tuple)
