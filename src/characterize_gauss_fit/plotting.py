################################################################################
# characterize_gauss_fit/plotting.py
################################################################################

"""Reusable matplotlib plot helpers for characterize_gauss_fit.

All functions accept pre-computed data arrays (not raw :class:`~trial.TrialResult`
lists) and return a :class:`matplotlib.figure.Figure` which the caller saves via
:func:`save_figure`. This keeps data transformation and rendering cleanly
separated.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

import matplotlib
import matplotlib.patheffects as _pe
import numpy as np
import numpy.typing as npt

matplotlib.use('Agg')  # non-interactive backend; must be set before other imports
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# DPI for saved PNG files.
_SAVE_DPI = 150

# Colour used to mark cells / points where the fitter did not converge.
_FAIL_COLOUR = '#cccccc'

# Line styles cycled across series in plot_line_with_bands so that series
# remain distinguishable when they overlap or when colour alone is ambiguous.
_LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]

# Path effects applied to heatmap cell annotations so text is readable on any
# background colour.
_HEATMAP_TEXT_EFFECTS = [_pe.withStroke(linewidth=3, foreground='white')]

# Text automatically appended to every line-with-bands plot explaining the
# shaded confidence region.
_BANDS_NOTE = 'Shaded bands: mean \u00b1 1 std.\u202fdev. across repeated trials'

# Style used for all figure footnotes.
_NOTE_STYLE: dict[str, Any] = {
    'ha': 'center',
    'va': 'bottom',
    'fontsize': 7,
    'color': '#555555',
    'style': 'italic',
    'transform': None,  # overridden per-call with fig.transFigure
}


def _add_figure_note(fig: Figure, note: str, *, bottom: float = 0.12) -> None:
    """Render ``note`` as a small italic footnote at the bottom of ``fig``.

    Uses ``tight_layout(rect=[0, note_strip, 1, 1])`` so matplotlib positions
    the axes, tick labels, and x-axis title entirely *above* the note strip in
    a single consistent pass.  This prevents the x-axis title from ending up
    in the same vertical band as the note text.

    Parameters:
        fig: The figure to annotate.
        note: Text to display.  May contain newlines.
        bottom: Minimum fraction of figure height reserved for the note strip.
    """
    n_lines = note.count('\n') + 1
    fig_height = fig.get_figheight()
    # 7 pt font at 72 pt/inch with 1.15x line spacing plus 0.06 in padding.
    note_frac = (n_lines * 7 * 1.15 / 72.0 + 0.06) / fig_height
    actual_bottom = max(bottom, note_frac)
    kw = dict(_NOTE_STYLE)
    kw['transform'] = fig.transFigure
    # Place note text just above the figure bottom edge, inside the reserved strip.
    fig.text(0.5, 0.005, note, **kw)
    # Single tight_layout call: everything (axes + labels) goes in the rect
    # above the note strip, so x-axis title never overlaps the note.
    fig.tight_layout(rect=(0.0, actual_bottom, 1.0, 1.0))


def save_figure(fig: Figure, output_dir: pathlib.Path, filename: str) -> pathlib.Path:
    """Save a figure to a PNG file and close it.

    Parameters:
        fig: The :class:`matplotlib.figure.Figure` to save.
        output_dir: Directory where the file will be written (must exist).
        filename: File name (should end in ``.png``).

    Returns:
        Path to the written PNG file.
    """
    path = output_dir / filename
    fig.savefig(path, dpi=_SAVE_DPI)
    plt.close(fig)
    return path


def plot_heatmap(
    data: npt.NDArray[np.float64],
    x_labels: list[str],
    y_labels: list[str],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    log_scale: bool = False,
    mask: npt.NDArray[np.bool_] | None = None,
    annotate: bool = True,
    note: str = '',
) -> Figure:
    """Create a 2-D heatmap (imshow) with optional log scaling and a fail mask.

    Parameters:
        data: 2-D array of shape ``(len(y_labels), len(x_labels))``.
        x_labels: Labels for the X axis (columns).
        y_labels: Labels for the Y axis (rows).
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        cbar_label: Colour-bar label.
        log_scale: If ``True``, apply ``log10`` to positive values before display.
        mask: Boolean array of the same shape as ``data``. Masked cells (``True``)
            are displayed in :data:`_FAIL_COLOUR` to indicate non-convergence.
        annotate: If ``True``, write the numeric value in each cell.
        note: Optional assumptions/context string rendered as a small italic
            footnote at the bottom of the figure.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    n_rows, n_cols = data.shape
    # Extra height (+1.2) reserves room for the title and the footnote without
    # either being clipped.  tight_layout() further adjusts subplot margins.
    fig, ax = plt.subplots(figsize=(max(7, n_cols * 0.9 + 1), max(5.5, n_rows * 0.7 + 1.2)))

    display = data.astype(float)
    if log_scale:
        with np.errstate(divide='ignore', invalid='ignore'):
            display = np.where(display > 0, np.log10(display), np.nan)

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color=_FAIL_COLOUR)

    if mask is not None:
        if mask.shape != data.shape:
            raise ValueError(
                f'mask.shape {mask.shape} does not match data.shape {data.shape}'
            )
        display = np.where(mask, np.nan, display)

    img = ax.imshow(display, aspect='auto', cmap=cmap, origin='upper')
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if annotate:
        for row in range(n_rows):
            for col in range(n_cols):
                val = display[row, col]
                if np.isnan(val):
                    text = 'N/C'
                elif log_scale:
                    text = f'{10**val:.2e}'
                else:
                    text = f'{val:.3f}'
                ax.text(
                    col,
                    row,
                    text,
                    ha='center',
                    va='center',
                    fontsize=6,
                    path_effects=_HEATMAP_TEXT_EFFECTS,
                )

    if note:
        _add_figure_note(fig, note, bottom=0.10)
    else:
        fig.tight_layout()
    return fig


def plot_line_with_bands(
    x: npt.NDArray[np.float64],
    y_means: Sequence[npt.NDArray[np.float64]],
    y_stds: Sequence[npt.NDArray[np.float64]],
    *,
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    log_x: bool = False,
    log_y: bool = False,
    note: str = '',
) -> Figure:
    """Create a multi-series line plot with shaded mean +/- 1 std bands.

    Parameters:
        x: 1-D array of X values (common to all series).
        y_means: Sequence of 1-D mean arrays, one per series.
        y_stds: Sequence of 1-D std arrays, one per series (same length as ``y_means``).
        labels: Legend labels, one per series.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        log_x: If ``True``, use a log scale for the X axis.
        log_y: If ``True``, use a log scale for the Y axis.
        note: Optional study-specific assumptions string.  It is combined with
            the automatic shaded-band explanation and rendered as a footnote.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (y_mean, y_std, label) in enumerate(zip(y_means, y_stds, labels, strict=True)):
        ls = _LINE_STYLES[i % len(_LINE_STYLES)]
        (line,) = ax.plot(x, y_mean, label=label, marker='o', markersize=3, linestyle=ls)
        colour = line.get_color()
        if log_y:
            lower = np.maximum(y_mean - y_std, 1e-15)
        else:
            lower = y_mean - y_std
        upper = y_mean + y_std
        ax.fill_between(x, lower, upper, alpha=0.2, color=colour)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        # Only apply log scale if there are positive values; otherwise the
        # matplotlib locator raises on tick-generation.
        all_values = np.concatenate(list(y_means))
        if np.any(all_values[np.isfinite(all_values)] > 0):
            ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(visible=True, which='both', alpha=0.3)

    full_note = f'{_BANDS_NOTE}  |  {note}' if note else _BANDS_NOTE
    _add_figure_note(fig, full_note, bottom=0.12)
    return fig


def plot_grouped_bars(
    categories: list[str],
    group_labels: list[str],
    values: npt.NDArray[np.float64],
    *,
    title: str,
    ylabel: str,
    log_scale: bool = False,
) -> Figure:
    """Create a grouped bar chart.

    Parameters:
        categories: Labels for the X-axis groups (outer variable).
        group_labels: Labels for the colour-coded bars within each group.
        values: 2-D array of shape ``(len(categories), len(group_labels))`` with
            the bar heights.
        title: Figure title.
        ylabel: Y-axis label.
        log_scale: If ``True``, use a log scale for the Y axis.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    n_cat = len(categories)
    n_groups = len(group_labels)
    fig, ax = plt.subplots(figsize=(max(7, n_cat * 1.2), 5))

    x = np.arange(n_cat, dtype=float)
    width = 0.8 / n_groups

    for g_idx, g_label in enumerate(group_labels):
        offsets = x + (g_idx - (n_groups - 1) / 2.0) * width
        bar_vals = values[:, g_idx].astype(float)
        nan_mask = np.isnan(bar_vals)
        display_vals = np.where(nan_mask, 0.0, bar_vals)
        ax.bar(offsets, display_vals, width=width * 0.9, label=g_label)
        for _cat_i, (off, is_nan) in enumerate(zip(offsets, nan_mask, strict=True)):
            if is_nan:
                ax.text(off, 0.0, 'NaN', ha='center', va='bottom', fontsize=5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    if log_scale:
        float_vals = values.astype(float)
        if (np.isfinite(float_vals) & (float_vals > 0)).any():
            ax.set_yscale('log')
    ax.grid(visible=True, axis='y', alpha=0.3)

    return fig


def plot_multi_panel_heatmaps(
    data_panels: Sequence[npt.NDArray[np.float64]],
    x_labels: list[str],
    y_labels: list[str],
    panel_titles: list[str],
    *,
    fig_title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    log_scale: bool = False,
    mask_panels: list[npt.NDArray[np.bool_]] | None = None,
) -> Figure:
    """Create a row of heatmap subplots sharing the same colour scale.

    Parameters:
        data_panels: Sequence of 2-D arrays, one per panel.
        x_labels: Shared X-axis labels.
        y_labels: Shared Y-axis labels.
        panel_titles: Title for each panel (one per element in ``data_panels``).
        fig_title: Overall figure title (suptitle).
        xlabel: Shared X-axis label.
        ylabel: Shared Y-axis label.
        cbar_label: Colour-bar label.
        log_scale: If ``True``, apply ``log10`` before display.
        mask_panels: Optional list of boolean mask arrays, one per panel.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    n_panels = len(data_panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(max(5, n_panels * 3.5), max(4, len(y_labels) * 0.6))
    )
    if n_panels == 1:
        axes = [axes]

    all_display: list[npt.NDArray[np.float64]] = []
    for panel_data in data_panels:
        arr = panel_data.astype(float)
        if log_scale:
            with np.errstate(divide='ignore', invalid='ignore'):
                arr = np.where(arr > 0, np.log10(arr), np.nan)
        all_display.append(arr)

    finite_vals = np.concatenate([d.ravel() for d in all_display])
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    vmin = float(finite_vals.min()) if len(finite_vals) > 0 else 0.0
    vmax = float(finite_vals.max()) if len(finite_vals) > 0 else 1.0

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color=_FAIL_COLOUR)
    img_ref = None

    if mask_panels is not None and len(mask_panels) != len(data_panels):
        raise ValueError(
            f'mask_panels length ({len(mask_panels)}) must equal '
            f'data_panels length ({len(data_panels)})'
        )

    for p_idx, (ax, display, ptitle) in enumerate(
        zip(axes, all_display, panel_titles, strict=True)
    ):
        panel_display = display.copy()
        if mask_panels is not None:
            panel_display = np.where(mask_panels[p_idx], np.nan, panel_display)

        img = ax.imshow(
            panel_display,
            aspect='auto',
            cmap=cmap,
            origin='upper',
            vmin=vmin,
            vmax=vmax,
        )
        img_ref = img
        ax.set_title(ptitle, fontsize=8)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(y_labels)))
        if p_idx == 0:
            ax.set_yticklabels(y_labels, fontsize=7)
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(xlabel)

    if img_ref is not None:
        fig.colorbar(img_ref, ax=axes, label=cbar_label, shrink=0.8)
    fig.suptitle(fig_title, fontsize=10)
    fig.tight_layout()

    return fig


def plot_recovery_fraction_heatmap(
    recovery_fractions: npt.NDArray[np.float64],
    x_labels: list[str],
    y_labels: list[str],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    note: str = '',
) -> Figure:
    """Create a heatmap of recovery fractions in [0, 1] with white-to-green scale.

    Used by Study 3 (minimum detectable offset) to show the fraction of trials
    where the offset was successfully recovered.

    Parameters:
        recovery_fractions: 2-D array of shape ``(len(y_labels), len(x_labels))``
            with values in [0, 1].
        x_labels: Labels for the X axis (columns).
        y_labels: Labels for the Y axis (rows).
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        note: Optional assumptions/context string rendered as a small italic
            footnote at the bottom of the figure.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    n_rows, n_cols = recovery_fractions.shape
    fig, ax = plt.subplots(figsize=(max(7, n_cols * 0.9 + 1), max(5.5, n_rows * 0.7 + 1.2)))

    img = ax.imshow(
        recovery_fractions,
        aspect='auto',
        cmap='Greens',
        origin='upper',
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Recovery fraction')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for row in range(n_rows):
        for col in range(n_cols):
            val = recovery_fractions[row, col]
            if np.isnan(val):
                text = 'N/C'
            else:
                text = f'{val:.2f}'
            ax.text(
                col,
                row,
                text,
                ha='center',
                va='center',
                fontsize=6,
                path_effects=_HEATMAP_TEXT_EFFECTS,
            )

    if note:
        _add_figure_note(fig, note, bottom=0.10)
    else:
        fig.tight_layout()
    return fig


def plot_constraint_summary(
    categories: list[str],
    group_labels: list[str],
    pos_err_vals: npt.NDArray[np.float64],
    pos_err_y_vals: npt.NDArray[np.float64],
    pos_err_x_vals: npt.NDArray[np.float64],
    scale_err_vals: npt.NDArray[np.float64],
    sigma_y_err_vals: npt.NDArray[np.float64],
    angle_err_vals: npt.NDArray[np.float64],
    *,
    title: str,
    note: str = '',
) -> Figure:
    """Create a 6-panel grouped bar chart for Study 5 constraint modes.

    Shows position error (Euclidean, Y-axis, and X-axis), scale error,
    sigma_y error, and angle error side by side so that the effect of parameter
    constraints on all metrics is visible at once.

    Parameters:
        categories: Constraint mode names (X-axis categories).
        group_labels: PSF shape labels (bar groups within each category).
        pos_err_vals: ``(n_cat, n_groups)`` Euclidean position error array.
        pos_err_y_vals: ``(n_cat, n_groups)`` absolute Y-axis position error.
        pos_err_x_vals: ``(n_cat, n_groups)`` absolute X-axis position error.
        scale_err_vals: ``(n_cat, n_groups)`` relative scale error array.
        sigma_y_err_vals: ``(n_cat, n_groups)`` relative sigma_y error array.
        angle_err_vals: ``(n_cat, n_groups)`` absolute angle error array (degrees).
        title: Overall figure title.
        note: Optional assumptions/context string rendered as a small italic
            footnote at the bottom of the figure.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    metrics: list[tuple[Axes, npt.NDArray[np.float64], str]] = [
        (axes[0, 0], pos_err_vals, 'Position error, Euclidean (pixels)'),
        (axes[0, 1], pos_err_y_vals, '|pos_err_y| (pixels)'),
        (axes[0, 2], pos_err_x_vals, '|pos_err_x| (pixels)'),
        (axes[1, 0], scale_err_vals, 'Relative scale error'),
        (axes[1, 1], sigma_y_err_vals, 'Relative sigma_y error'),
        (axes[1, 2], angle_err_vals, 'Angle error (\u00b0, floating modes only)'),
    ]

    n_cat = len(categories)
    n_groups = len(group_labels)
    x = np.arange(n_cat, dtype=float)
    width = 0.8 / n_groups

    for ax, vals, metric_label in metrics:
        for g_idx, g_label in enumerate(group_labels):
            offsets = x + (g_idx - (n_groups - 1) / 2.0) * width
            bar_vals = vals[:, g_idx].astype(float)
            nan_mask = np.isnan(bar_vals)
            display_vals = np.where(nan_mask, 0.0, bar_vals)
            ax.bar(offsets, display_vals, width=width * 0.9, label=g_label)
            for off, is_nan in zip(offsets, nan_mask, strict=True):
                if is_nan:
                    ax.text(off, 0.0, 'NaN', ha='center', va='bottom', fontsize=5, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=7)
        ax.set_ylabel(metric_label, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(visible=True, axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=10)
    if note:
        _add_figure_note(fig, note, bottom=0.08)
    else:
        fig.tight_layout()
    return fig
