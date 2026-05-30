"""Visualization utilities for olive oil spectral data."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_spectra(
    spectra: pd.DataFrame | tuple[np.ndarray, np.ndarray, list[str]],
    *,
    metadata: pd.DataFrame | None = None,
    color_by: str | None = None,
    facet_row_by: str | None = None,
    facet_col_by: str | None = None,
    show_mean: bool = True,
    regions: list[tuple[float, float]] | None = None,
    xlabel: str = "Spectral variable",
    ylabel: str = "Response",
    alpha: float = 0.3,
    height: float = 4,
    aspect: float = 2.5,
    title: str | None = None,
) -> sns.FacetGrid:
    """Plot spectra with optional grouping, faceting, mean overlay and region shading.

    Parameters
    ----------
    spectra:
        Either a spectral DataFrame with a 'sample_id' column (as returned by
        load_dataset or average_replicates), or a ``(X, axis, sample_ids)``
        tuple as returned by get_spectral_matrix. Works with raw replicates,
        averaged data, and preprocessed spectra.
    metadata:
        Optional DataFrame with a 'sample_id' column. Its columns become
        available for color_by / facet_row_by / facet_col_by. Merge labels
        into this DataFrame before passing if you want to colour by class.
    color_by:
        Column name to use for hue colouring (e.g. ``"label"``, ``"origin"``).
    facet_row_by:
        Column name to use for row facets.
    facet_col_by:
        Column name to use for column facets.
    show_mean:
        Overlay a thick mean spectrum. When color_by is set, one mean per
        hue group is drawn in the corresponding colour.
    regions:
        List of ``(start, end)`` tuples defining spectral regions to shade in
        green (e.g. model-selected regions). Applied to every subplot.
    xlabel:
        X-axis label — e.g. ``"Wavelength (nm)"``, ``"Wavenumber (cm⁻¹)"``.
    ylabel:
        Y-axis label — e.g. ``"Absorbance"``, ``"Counts"``.
    alpha:
        Transparency of individual spectrum lines.
    height:
        Height of each facet in inches.
    aspect:
        Width-to-height ratio of each facet.

    Returns
    -------
    sns.FacetGrid
        The seaborn FacetGrid object. Access ``.figure`` for the matplotlib
        Figure and ``.axes`` for the individual Axes.

    Examples
    --------
    Colour by class label::

        labels = data["labels"].rename(columns={"label": "musty"})
        g = plot_spectra(data["cal_uvvis"], metadata=labels,
                         color_by="musty", xlabel="Wavelength (nm)",
                         ylabel="Absorbance")

    Facet by origin, colour by label::

        merged_meta = data["metadata"].merge(data["labels"], on="sample_id")
        g = plot_spectra(data["cal_uvvis"], metadata=merged_meta,
                         color_by="label", facet_col_by="origin",
                         regions=[(580, 1000)])
    """
    long_df, spectral_cols = _build_long_df(spectra, metadata)

    g = sns.FacetGrid(
        long_df,
        hue=color_by,
        row=facet_row_by,
        col=facet_col_by,
        palette="tab10",
        height=height,
        aspect=aspect,
        margin_titles=True,
    )

    def _draw(data: pd.DataFrame, color: str = "steelblue", **kwargs) -> None:
        ax = plt.gca()
        for _, grp in data.groupby("_row_id", sort=False):
            ax.plot(
                grp["spectral_variable"],
                grp["response"],
                color=color,
                alpha=alpha,
                lw=0.7,
            )
        if show_mean:
            mean_vals = (
                data.groupby("spectral_variable", sort=True)["response"].mean()
            )
            ax.plot(mean_vals.index, mean_vals.values,
                    color=_darken(color), lw=2.5, linestyle="--")

    g.map_dataframe(_draw)

    # Shade spectral regions on every subplot
    if regions:
        for ax in g.axes.flat:
            for start, end in regions:
                ax.axvspan(start, end, color="green", alpha=0.15)

    g.set_axis_labels(xlabel, ylabel)

    # Build a unified figure legend (hue groups + region patch)
    legend_handles = _build_legend_handles(long_df, color_by, regions)
    if legend_handles:
        g.figure.legend(
            handles=legend_handles,
            title=color_by,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            framealpha=0.7,
        )
        g.figure.tight_layout()

    if title is not None:
        g.figure.suptitle(title, y=1.02, fontsize=13, fontweight="bold")

    return g


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_long_df(
    spectra: pd.DataFrame | tuple,
    metadata: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list]:
    """Convert wide spectral input to a tidy long DataFrame for seaborn.

    Returns (long_df, spectral_cols) where spectral_cols is the list of
    column values that were melted into 'spectral_variable'.
    """
    # --- Normalise to wide DataFrame ----------------------------------------
    if isinstance(spectra, tuple):
        X, axis, sample_ids = spectra
        spectral_cols = [float(a) for a in axis]
        wide_df = pd.DataFrame(X, columns=spectral_cols)
        wide_df.insert(0, "sample_id", sample_ids)
    else:
        wide_df = spectra.copy().reset_index(drop=True)
        if "sample_id" not in wide_df.columns:
            wide_df.insert(0, "sample_id", wide_df.index.astype(str))
        spectral_cols = [c for c in wide_df.columns if isinstance(c, (int, float))]

    wide_df["sample_id"] = wide_df["sample_id"].astype(str)

    # Unique row ID to distinguish replicates of the same sample
    wide_df["_row_id"] = wide_df["sample_id"] + "_" + wide_df.index.astype(str)

    # --- Merge metadata on sample_id ----------------------------------------
    if metadata is not None:
        meta = metadata.copy()
        meta["sample_id"] = meta["sample_id"].astype(str)
        extra_cols = [c for c in meta.columns if c != "sample_id"]
        wide_df = wide_df.merge(
            meta[["sample_id"] + extra_cols], on="sample_id", how="left"
        )

    # --- Melt to long format -------------------------------------------------
    id_cols = [c for c in wide_df.columns if c not in spectral_cols]
    long_df = wide_df.melt(
        id_vars=id_cols,
        value_vars=spectral_cols,
        var_name="spectral_variable",
        value_name="response",
    )
    long_df["spectral_variable"] = long_df["spectral_variable"].astype(float)
    long_df = long_df.sort_values(["_row_id", "spectral_variable"]).reset_index(drop=True)

    return long_df, spectral_cols


def _darken(color, factor: float = 0.6) -> tuple:
    """Return a darker shade of color by scaling RGB values."""
    r, g, b, *a = mcolors.to_rgba(color)
    return (r * factor, g * factor, b * factor, *(a or (1.0,)))


def _build_legend_handles(
    long_df: pd.DataFrame,
    color_by: str | None,
    regions: list[tuple[float, float]] | None,
) -> list:
    """Build matplotlib legend handles for hue groups and shaded regions."""
    handles = []

    if color_by is not None:
        # Match seaborn's internal sort (sorted string representation)
        unique_vals = sorted(long_df[color_by].dropna().unique(), key=str)
        palette = sns.color_palette("tab10", n_colors=len(unique_vals))
        for val, color in zip(unique_vals, palette):
            handles.append(mpatches.Patch(color=color, label=str(val)))

    if regions:
        handles.append(
            mpatches.Patch(facecolor="green", alpha=0.4, label="Selected region")
        )

    return handles
