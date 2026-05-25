"""Synthetic camera images from LMT pulse sequences.

Each row of an AtomState is a coherent branch with its own position and complex
amplitude.  To form an image, within ONE atom we bin the complex branch
amplitudes onto the pixel grid, convolve that complex field with a Gaussian of
size ``SINGLE_ATOM_WAVEPACKET_SIGMA_M`` (the single-atom wave packet), then
square.  So branches closer than ~a packet width interfere and well-separated
ones do not.  Different atoms in the ensemble add incoherently (intensities sum).

==========================================================================
TODO (BIG ONE -- the whole point of the position/velocity tracking):
    Replace this blurred-point-field heuristic with proper GAUSSIAN WAVE
    PACKET TRACKING.

    What we do now: each branch is a point amplitude, given a wave-packet
    extent only at imaging time by a single fixed Gaussian blur applied to the
    complex field before squaring.  This restores interference between branches
    within ~a packet width (e.g. recombining interferometer arms), which a
    naive per-pixel rule would destroy whenever they straddle a bin edge.  But
    it is still NOT correct:
      - one fixed blur for all branches; real packets have their own size and
        expand ballistically (uncertainty principle) over the sequence;
      - it ignores the momentum (m-dependent) phase gradient ACROSS each packet,
        which suppresses interference between different-m branches even when
        they spatially overlap;
      - it is frame-sensitive once branches of different m overlap (worked
        around by imaging in the lab frame, see ``collect_filmstrip``).

    The correct treatment: give every branch a finite Gaussian wave packet
    (initial trap size, expanding ballistically), propagate those packets with
    their momentum phase, and overlap the COMPLEX fields.  Interference then
    falls out of real spatial overlap.  All the per-row position/velocity
    bookkeeping carried through the simulation exists precisely so this can be
    done -- that is its entire purpose.

    Not doing it now (deliberately).  Until it lands, every image from this
    module is a heuristic.
==========================================================================
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import lmt_sim.lmt_simulation as sim
from lmt_sim.lmt_simulation import make_atom_states
from lmt_sim.lmt_sequence import iter_pulse_sequence_in_borde_representation


# Rough single-atom wave-packet size to blur each image by, standing in for the
# real (untracked) spatial extent of an atom's wave packet at readout.
#
# Estimate (Rb-87): an atom released from the low harmonic-oscillator levels
# (n = 0..10) of an optical trap with ~10 um beam waist starts at an in-trap
# position spread of sigma_0 ~ 0.2-1.1 um.  The uncertainty principle then fixes
# a velocity spread sigma_v = hbar / (2 m sigma_0), so after ~4 ms of free fall
# the packet has grown to
#     sigma(t) = sqrt(sigma_0^2 + (sigma_v * t)^2) ~ 6 um (n=0) .. 28 um (n=10).
# ~10 um is a representative ground-to-low-n value; see the module TODO -- this
# whole blur is a placeholder for proper wave-packet tracking.
SINGLE_ATOM_WAVEPACKET_SIGMA_M = 10e-6


COLUMNS = ("x", "z", "v_z", "weight", "amp_re", "amp_im")


def collect_branches(state):
    """Split the rows of one ``AtomState`` into ground and excited branches.

    Every surviving row is kept as its own branch -- crucially we do NOT group
    by ``(m, internal)`` and we do NOT pick a single position for a group.  Each
    branch keeps its own position and complex amplitude; whether two branches
    interfere is decided later by their wave-packet overlap in :func:`render`.

    Returns
    -------
    ground, excited : ndarray, shape (n, 6)
        Columns ``COLUMNS`` = ``(x, z, v_z, weight, amp_re, amp_im)`` where
        ``weight`` is the per-row probability ``|amplitude|^2`` (incoherent; for
        sanity only -- the imaged intensity is the coherent wave-packet sum, see
        :func:`render`) and ``amp_re``/``amp_im`` are the real/imag parts of the
        branch amplitude used for that coherent sum.  Empty ``(0, 6)`` if no
        rows survive.  These are one atom's branches; amplitudes must be in the
        lab frame for cross-``m`` interference to be physical (see
        :func:`collect_filmstrip`).
    """
    out = []
    for is_g in (True, False):
        rows = state.internal_is_ground == is_g
        amps = state.amplitudes[rows]
        keep = amps != 0
        out.append(
            np.column_stack((
                state.positions[rows, 0][keep],
                state.positions[rows, 2][keep],
                state.velocities[rows, 2][keep],
                np.abs(amps[keep]) ** 2,
                amps[keep].real,
                amps[keep].imag,
            ))
            if np.any(keep)
            else np.empty((0, 6))
        )
    return out[0], out[1]


def stack_atoms(branch_arrays):
    """Stack per-atom branch arrays into one array with a trailing atom-id column.

    Each input array (from :func:`collect_branches`) is one atom; the id column
    lets :func:`render` keep distinct atoms incoherent while summing each atom's
    own branches coherently.  Returns shape ``(N, 7)`` (``COLUMNS`` + ``atom_id``)
    or empty ``(0, 7)``.
    """
    arrays = [b for b in branch_arrays if len(b)]
    if not arrays:
        return np.empty((0, 7))
    tagged = [
        np.hstack((b, np.full((len(b), 1), aid, dtype=b.dtype)))
        for aid, b in enumerate(arrays)
    ]
    return np.vstack(tagged)


def render(branches, x_edges, z_edges, *, blur_sigma_m=SINGLE_ATOM_WAVEPACKET_SIGMA_M):
    """Image branches: per-atom coherent wave-packet sum, atoms incoherent.

    Each branch is treated as a Gaussian wave packet of size ``blur_sigma_m``
    (metres).  Within one atom (column ``atom_id`` if present, else all branches
    are one atom) the COMPLEX field is binned by ``(x, z)`` and convolved with
    that Gaussian, then squared -- so branches closer than ~a packet width
    interfere and well-separated ones do not.  Distinct atoms add incoherently
    (their intensities sum).

    The blur scale is what sets the interference range, NOT the (arbitrary,
    grid-dependent) pixel size: with ``blur_sigma_m=0`` each branch only
    interferes with others in the very same pixel, which spuriously kills
    fringes whenever recombining arms straddle a bin edge -- so 0 is for
    diagnostics only, not real images.  This whole "blur a point field" model is
    still a heuristic for true wave-packet overlap; see the module-level TODO.
    """
    n_z, n_x = len(z_edges) - 1, len(x_edges) - 1
    img = np.zeros((n_z, n_x))
    if not len(branches):
        return img

    sigma_pix = None
    if blur_sigma_m:
        sigma_pix = (blur_sigma_m / (z_edges[1] - z_edges[0]),
                     blur_sigma_m / (x_edges[1] - x_edges[0]))

    if branches.shape[1] >= 7:
        atoms = (branches[branches[:, 6] == a] for a in np.unique(branches[:, 6]))
    else:
        atoms = (branches,)
    for atom in atoms:
        re, _, _ = np.histogram2d(atom[:, 1], atom[:, 0],
                                  bins=(z_edges, x_edges), weights=atom[:, 4])
        im, _, _ = np.histogram2d(atom[:, 1], atom[:, 0],
                                  bins=(z_edges, x_edges), weights=atom[:, 5])
        if sigma_pix is not None:
            # Convolve the COMPLEX field (not the intensity) so co-located
            # branches interfere; square afterwards to get this atom's density.
            re = gaussian_filter(re, sigma_pix)
            im = gaussian_filter(im, sigma_pix)
        img += re ** 2 + im ** 2
    return img


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
    after every event.  Each Bordé-frame snapshot is transformed back to the lab
    frame before collecting branches: the per-pixel coherent sum in
    :func:`render` can combine branches of different ``m``, whose relative phase
    is frame-dependent, so it must be done in the (physical) lab frame.

    Returns
    -------
    snap_g, snap_e : list of ndarray
        ``len(pulse_sequence) + 1`` entries each, shape ``(N, 7)`` (``COLUMNS`` +
        ``atom_id``, so :func:`render` keeps atoms incoherent).  The first entry
        is the initial state.
    """
    n_snap = len(pulse_sequence) + 1
    snap_g = [[] for _ in range(n_snap)]
    snap_e = [[] for _ in range(n_snap)]

    iterator = tqdm(velocities, desc=desc) if progress else velocities
    for v in iterator:
        initial = make_atom_states(initial_velocity_z=v, c0=c0, c1=c1)
        for i, (state, detuning_hz, t) in enumerate(
            iter_pulse_sequence_in_borde_representation(
                initial, pulse_sequence,
                initial_velocity_z=v,
                discard_threshold=discard_threshold,
            )
        ):
            lab_state = sim.transform_state_vector(
                state,
                omega_laser=2 * np.pi * (sim.TRANSITION_FREQUENCY + detuning_hz),
                t=t, z=0.0, vz=v, inverse=True,
            )
            g, e = collect_branches(lab_state)
            if len(g):
                snap_g[i].append(g)
            if len(e):
                snap_e[i].append(e)

    return (
        [stack_atoms(s) for s in snap_g],
        [stack_atoms(s) for s in snap_e],
    )


def plot_camera_shot(ground, excited, *,
                     ground_title=None, excited_title=None, suptitle=None,
                     n_x=21, n_z=48, blur_sigma_m=SINGLE_ATOM_WAVEPACKET_SIGMA_M,
                     x_pad_frac=0.25, z_pad_frac=0.15):
    """Two-panel ground / excited camera image from pre-collected branches.

    Each panel shares a single color scale (taken as the smaller of the two
    peaks so neither saturates).  ``ground`` and ``excited`` are arrays from
    :func:`collect_branches`/:func:`stack_atoms` (``COLUMNS`` + optional
    ``atom_id``).
    """
    x_edges, z_edges = pixel_grid([ground, excited], n_x=n_x, n_z=n_z,
                                  x_pad_frac=x_pad_frac, z_pad_frac=z_pad_frac)
    g_image = render(ground, x_edges, z_edges, blur_sigma_m=blur_sigma_m)
    e_image = render(excited, x_edges, z_edges, blur_sigma_m=blur_sigma_m)
    vmax = min(g_image.max(), e_image.max())
    extent = [1e6 * x_edges[0], 1e6 * x_edges[-1], 1e6 * z_edges[0], 1e6 * z_edges[-1]]

    g_total = g_image.sum()
    e_total = e_image.sum()
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
                   n_x=21, n_z=48, blur_sigma_m=SINGLE_ATOM_WAVEPACKET_SIGMA_M,
                   discard_threshold=1e-9,
                   progress=True, desc="Filmstrip"):
    """Render a 2-row ground/excited filmstrip stepping through ``pulse_sequence``.

    Each column is the camera image after one more event; the first column is
    the initial state.  Each panel autoscales to its own peak.  Annotations
    show imaged weight (``w=``) and peak pixel value (``peak=``) so brightness
    stays comparable across panels.
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
    imgs_g = [render(s, x_edges, z_edges, blur_sigma_m=blur_sigma_m) for s in snap_g]
    imgs_e = [render(s, x_edges, z_edges, blur_sigma_m=blur_sigma_m) for s in snap_e]
    totals_g = [img.sum() for img in imgs_g]
    totals_e = [img.sum() for img in imgs_e]
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
        for row, (image_set, total_set, channel) in enumerate([
            (imgs_g, totals_g, "ground"),
            (imgs_e, totals_e, "excited"),
        ]):
            ax = axes[row, col]
            ax.imshow(image_set[col], origin="lower", aspect="auto",
                      extent=extent, cmap="magma", vmin=0)
            peak = image_set[col].max()
            if row == 0:
                ax.set_title(label, fontsize=8,
                             rotation=30 if rotate_titles else 0,
                             ha="left" if rotate_titles else "center")
            if col == 0:
                ax.set_ylabel(f"{channel}\n$z$ ($\\mu$m)")
            if row == 1:
                ax.set_xlabel(r"$x$ ($\mu$m)")
            ax.text(0.03, 0.97, f"w={total_set[col]:.1f}\npeak={peak:.1f}",
                    transform=ax.transAxes, fontsize=7, color="white", va="top")

    g_final, e_final = totals_g[-1], totals_e[-1]
    excited_fraction = e_final / (g_final + e_final) if (g_final + e_final) else 0.0
    fig.suptitle(title or
                 f"Filmstrip (final excited fraction = {excited_fraction:.2f}; "
                 f"each panel autoscaled)")
    return fig
