"""Synthetic camera images from LMT pulse sequences.

Each row of an AtomState is a coherent branch.  Branches sharing the same
(m, internal state) at the readout are the same final channel and interfere
coherently -- amplitudes are summed before squaring to get the imaged weight.
The image is a 2D histogram of branch (x, z) positions, blurred along x by a
Gaussian PSF (the cameras here have no real transverse expansion of their own).

The frame-invariant fact this module relies on: per (m, internal) channel,
the Bordé-frame phase factor is the same for every contributing row, so
|sum amplitudes|^2 is identical in the Bordé and lab frames.  That lets us
snapshot the state once after each event without paying for the lab-frame
round trip.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import lmt_sim.lmt_simulation as sim
from lmt_sim.lmt_simulation import make_atom_states
from lmt_sim.lmt_sequence import iter_pulse_sequence_in_borde_representation


def collect_branches(state):
    """Coherently group rows of one ``AtomState`` by ``(m, internal)``.

    Returns
    -------
    ground, excited : ndarray, shape (n, 4)
        Columns ``(x, z, v_z, weight)`` for the ground and excited final
        channels of this atom.  ``weight = |sum amplitudes|^2`` within the
        channel.  Returns empty ``(0, 4)`` arrays if no rows survive.
    """
    out_g, out_e = [], []
    for is_g, sink in [(True, out_g), (False, out_e)]:
        mask_int = state.internal_is_ground == is_g
        for m in np.unique(state.m_values[mask_int]):
            rows = mask_int & (state.m_values == m)
            w = abs(state.amplitudes[rows].sum()) ** 2
            if w == 0:
                continue
            i = np.flatnonzero(rows)[0]
            sink.append((
                state.positions[i, 0],
                state.positions[i, 2],
                state.velocities[i, 2],
                w,
            ))
    return (
        np.array(out_g) if out_g else np.empty((0, 4)),
        np.array(out_e) if out_e else np.empty((0, 4)),
    )


def render(branches, x_edges, z_edges, *, psf_sigma_pix=1.5):
    """Histogram ``(x, z, weight)`` branches into a 2D image and blur along x."""
    if not len(branches):
        return np.zeros((len(z_edges) - 1, len(x_edges) - 1))
    img, _, _ = np.histogram2d(
        branches[:, 1], branches[:, 0],
        bins=(z_edges, x_edges), weights=branches[:, 3],
    )
    return gaussian_filter1d(img, psf_sigma_pix, axis=1)


def pixel_grid(branch_arrays, *, n_x=21, n_z=48,
               x_pad_frac=0.25, z_pad_frac=0.05,
               x_pad_min=6e-6, z_pad_min=1e-6):
    """Build shared ``(x_edges, z_edges)`` spanning every branch array given."""
    pts = np.vstack([b[:, :2] for b in branch_arrays if len(b)])
    x_pad = max(x_pad_frac * np.ptp(pts[:, 0]), x_pad_min)
    z_pad = max(z_pad_frac * np.ptp(pts[:, 1]), z_pad_min)
    x_edges = np.linspace(pts[:, 0].min() - x_pad, pts[:, 0].max() + x_pad, n_x + 1)
    z_edges = np.linspace(pts[:, 1].min() - z_pad, pts[:, 1].max() + z_pad, n_z + 1)
    return x_edges, z_edges


def collect_filmstrip(pulse_sequence, velocities, *,
                      c0=1.0, c1=0.0,
                      discard_threshold=1e-9,
                      progress=True, desc="Filmstrip"):
    """Snapshot the ensemble after every event of ``pulse_sequence``.

    For each atom velocity, the sequence is run once via
    ``iter_pulse_sequence_in_borde_representation`` and a snapshot is taken
    after every event.  The yielded states are in the Bordé frame but that's
    fine for ``collect_branches`` -- the frame-change phase factor is the same
    for every row in a ``(m, internal)`` group, so ``|sum amplitudes|^2`` is
    identical to the lab frame.

    Returns
    -------
    snap_g, snap_e : list of ndarray
        ``len(pulse_sequence) + 1`` entries each, columns
        ``(x, z, v_z, weight)``.  The first entry is the initial state.
    """
    n_snap = len(pulse_sequence) + 1
    snap_g = [[] for _ in range(n_snap)]
    snap_e = [[] for _ in range(n_snap)]

    iterator = tqdm(velocities, desc=desc) if progress else velocities
    for v in iterator:
        initial = make_atom_states(initial_velocity_z=v, c0=c0, c1=c1)
        for i, (state, _, _) in enumerate(iter_pulse_sequence_in_borde_representation(
            initial, pulse_sequence,
            initial_velocity_z=v,
            discard_threshold=discard_threshold,
        )):
            g, e = collect_branches(state)
            if len(g):
                snap_g[i].append(g)
            if len(e):
                snap_e[i].append(e)

    return (
        [np.vstack(s) if s else np.empty((0, 4)) for s in snap_g],
        [np.vstack(s) if s else np.empty((0, 4)) for s in snap_e],
    )


def plot_camera_shot(ground, excited, *,
                     ground_title=None, excited_title=None, suptitle=None,
                     n_x=21, n_z=48, psf_sigma_pix=1.5,
                     x_pad_frac=0.25, z_pad_frac=0.15):
    """Two-panel ground / excited camera image from pre-collected branches.

    Each panel shares a single color scale (taken as the smaller of the two
    peaks so neither saturates).  ``ground`` and ``excited`` are ``(n, 4)``
    arrays from ``collect_branches`` (columns ``x, z, v_z, weight``).
    """
    x_edges, z_edges = pixel_grid([ground, excited], n_x=n_x, n_z=n_z,
                                  x_pad_frac=x_pad_frac, z_pad_frac=z_pad_frac)
    g_image = render(ground, x_edges, z_edges, psf_sigma_pix=psf_sigma_pix)
    e_image = render(excited, x_edges, z_edges, psf_sigma_pix=psf_sigma_pix)
    vmax = min(g_image.max(), e_image.max())
    extent = [1e6 * x_edges[0], 1e6 * x_edges[-1], 1e6 * z_edges[0], 1e6 * z_edges[-1]]

    g_total = ground[:, 3].sum() if len(ground) else 0.0
    e_total = excited[:, 3].sum() if len(excited) else 0.0
    if ground_title is None:
        ground_title = f"Ground-state camera (weight = {g_total:.2f})"
    if excited_title is None:
        excited_title = f"Excited-state camera (weight = {e_total:.2f})"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, constrained_layout=True)
    for ax, img, title in zip(axes, [g_image, e_image], [ground_title, excited_title]):
        ax.imshow(img, origin="lower", aspect="auto", extent=extent,
                  cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r"$x$ position ($\mu$m)")
    axes[0].set_ylabel(r"$z$ position ($\mu$m)")
    fig.colorbar(axes[1].images[0], ax=axes, shrink=0.9, label="Weighted atom density")
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def plot_filmstrip(pulse_sequence, velocities, *,
                   labels=None, title=None,
                   panel_width=2.5, panel_height=4.0,
                   n_x=21, n_z=48, psf_sigma_pix=1.5,
                   discard_threshold=1e-9,
                   progress=True, desc="Filmstrip"):
    """Render a 2-row ground/excited filmstrip stepping through ``pulse_sequence``.

    Each column is the camera image after one more event; the first column is
    the initial state.  Each panel autoscales to its own peak.  Annotations
    show integrated weight (``w=``) and peak pixel value (``peak=``) so
    brightness stays comparable across panels.
    """
    snap_g, snap_e = collect_filmstrip(pulse_sequence, velocities,
                                       discard_threshold=discard_threshold,
                                       progress=progress, desc=desc)
    n_snap = len(snap_g)

    if labels is None:
        labels = ["initial"] + [
            f"after {getattr(ev, 'label', type(ev).__name__)}" for ev in pulse_sequence
        ]

    x_edges, z_edges = pixel_grid(snap_g + snap_e, n_x=n_x, n_z=n_z)
    imgs_g = [render(s, x_edges, z_edges, psf_sigma_pix=psf_sigma_pix) for s in snap_g]
    imgs_e = [render(s, x_edges, z_edges, psf_sigma_pix=psf_sigma_pix) for s in snap_e]
    extent = [1e6 * x_edges[0], 1e6 * x_edges[-1], 1e6 * z_edges[0], 1e6 * z_edges[-1]]

    fig, axes = plt.subplots(
        2, n_snap,
        figsize=(panel_width * n_snap, panel_height * 2),
        sharex=True, sharey=True, constrained_layout=True,
    )
    if n_snap == 1:
        axes = np.asarray(axes).reshape(2, 1)

    rotate_titles = n_snap > 8
    for col, label in enumerate(labels):
        for row, (image_set, snap_set, channel) in enumerate([
            (imgs_g, snap_g, "ground"),
            (imgs_e, snap_e, "excited"),
        ]):
            ax = axes[row, col]
            ax.imshow(image_set[col], origin="lower", aspect="auto",
                      extent=extent, cmap="magma", vmin=0)
            total = snap_set[col][:, 3].sum() if len(snap_set[col]) else 0.0
            peak = image_set[col].max()
            if row == 0:
                ax.set_title(label, fontsize=8,
                             rotation=30 if rotate_titles else 0,
                             ha="left" if rotate_titles else "center")
            if col == 0:
                ax.set_ylabel(f"{channel}\n$z$ ($\\mu$m)")
            if row == 1:
                ax.set_xlabel(r"$x$ ($\mu$m)")
            ax.text(0.03, 0.97, f"w={total:.1f}\npeak={peak:.1f}",
                    transform=ax.transAxes, fontsize=7, color="white", va="top")

    g_final = snap_g[-1][:, 3].sum() if len(snap_g[-1]) else 0.0
    e_final = snap_e[-1][:, 3].sum() if len(snap_e[-1]) else 0.0
    excited_fraction = e_final / (g_final + e_final) if (g_final + e_final) else 0.0
    fig.suptitle(title or
                 f"Filmstrip (final excited fraction = {excited_fraction:.2f}; "
                 f"each panel autoscaled)")
    return fig
