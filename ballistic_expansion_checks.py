"""Ballistic expansion imaging sanity checks.

Three checks that each validate a different aspect of the imaging pipeline
for a freefall-only (no-pulse) sequence.

Check 1 — static atom
    Single atom at rest at the origin.  All 11 panels (t=0 … t=10 ms) must
    show an identical Gaussian blob; the fixed blur is independent of drop
    time so nothing should change.

Check 2 — drifting atom
    Single atom with a small upward vz and a transverse vx.  The blob must
    drift monotonically upward in z and sideways in x across the filmstrip.

Check 3 — thermal cloud at 1 µK
    Ensemble of atoms drawn from a Gaussian position distribution (σ=100 µm)
    and a Maxwell–Boltzmann velocity distribution at 1 µK.  The cloud must
    expand visibly over the 10 ms sequence.

Usage
-----
    uv run python ballistic_expansion_checks.py
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import constants
from tqdm import tqdm

import lmt_sim.lmt_simulation as sim
from lmt_sim.lmt_simulation import make_atom_states
from lmt_sim.lmt_sequence import Freefall, iter_pulse_sequence_in_borde_representation
from lmt_sim.imaging import (
    collect_branches,
    stack_atoms,
    render,
    pixel_grid,
)

# ---------------------------------------------------------------------------
# Shared sequence: 10 × 1 ms freefalls
# ---------------------------------------------------------------------------
N_STEPS = 10
DT = 1e-3  # 1 ms per step
SEQUENCE = [Freefall(duration=DT, label=f"t={i + 1}ms") for i in range(N_STEPS)]
LABELS = ["t=0 ms"] + [f"t={i + 1} ms" for i in range(N_STEPS)]


# ---------------------------------------------------------------------------
# Core collection helper (supports full 3-D position + velocity)
# ---------------------------------------------------------------------------


def collect_snapshots(atoms: list[dict], sequence: list) -> tuple:
    """Snapshot ground/excited branch arrays after every sequence event.

    Parameters
    ----------
    atoms:
        List of dicts with optional keys ``px``, ``py``, ``pz``,
        ``vx``, ``vy``, ``vz`` (all default to 0.0).
    sequence:
        List of ``Freefall`` (or other) sequence events.

    Returns
    -------
    snap_g, snap_e : list[ndarray]
        One stacked branch array per snapshot (``len(sequence)+1`` entries).
    """
    n_snap = len(sequence) + 1
    snap_g = [[] for _ in range(n_snap)]
    snap_e = [[] for _ in range(n_snap)]

    for atom_id, atom in enumerate(tqdm(atoms, desc="Atoms", leave=False)):
        px = atom.get("px", 0.0)
        py = atom.get("py", 0.0)
        pz = atom.get("pz", 0.0)
        vx = atom.get("vx", 0.0)
        vy = atom.get("vy", 0.0)
        vz = atom.get("vz", 0.0)

        initial = make_atom_states(
            position_x=px,
            position_y=py,
            position_z=pz,
            velocity_x=vx,
            velocity_y=vy,
            initial_velocity_z=vz,
        )

        # At t=0 with m=0 the Bordé frame coincides with the lab frame, so the
        # initial state can be passed directly to the Bordé iterator.
        for i, (state, detuning_hz, t) in enumerate(
            iter_pulse_sequence_in_borde_representation(
                initial,
                sequence,
                initial_velocity_z=vz,
                discard_threshold=1e-9,
            )
        ):
            lab_state = sim.transform_state_vector(
                state,
                omega_laser=2 * np.pi * (sim.TRANSITION_FREQUENCY + detuning_hz),
                t=t,
                z=0.0,
                vz=vz,
                inverse=True,
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


# ---------------------------------------------------------------------------
# Filmstrip renderer
# ---------------------------------------------------------------------------


def render_filmstrip_figure(
    snaps: list,
    labels: list[str],
    title: str,
    *,
    channel: str = "ground",
    n_x: int = 30,
    n_z: int = 60,
    panel_width: float = 2.3,
    panel_height: float = 3.8,
    x_pad_min: float = 30e-6,
    z_pad_min: float = 30e-6,
    common_scale: bool = True,
) -> plt.Figure:
    """Render a single-row filmstrip from pre-collected branch snapshots.

    Parameters
    ----------
    snaps:
        List of stacked branch arrays (one per time step).
    labels:
        Column header labels.
    title:
        Figure suptitle.
    channel:
        Label string for the y-axis (e.g. ``"ground"``).
    common_scale:
        If True, all panels share the same colour scale (global peak).
        If False, each panel autoscales independently.
    """
    non_empty = [s for s in snaps if len(s)]
    if not non_empty:
        raise ValueError("All snapshots are empty — nothing to render.")

    x_edges, z_edges = pixel_grid(
        non_empty,
        n_x=n_x,
        n_z=n_z,
        x_pad_min=x_pad_min,
        z_pad_min=z_pad_min,
    )
    imgs = [render(s, x_edges, z_edges) for s in snaps]
    global_max = max(img.max() for img in imgs) or 1.0
    extent = [
        1e6 * x_edges[0],
        1e6 * x_edges[-1],
        1e6 * z_edges[0],
        1e6 * z_edges[-1],
    ]

    n_panels = len(snaps)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(panel_width * n_panels, panel_height),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for col, (ax, img, label) in enumerate(zip(axes, imgs, labels)):
        vmax = global_max if common_scale else (img.max() or 1.0)
        ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(label, fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{channel}\n" + r"$z$ ($\mu$m)", fontsize=8)
        ax.set_xlabel(r"$x$ ($\mu$m)", fontsize=7)
        peak = img.max()
        total = img.sum()
        ax.text(
            0.03,
            0.97,
            f"pk={peak:.1f}\nΣ={total:.1f}",
            transform=ax.transAxes,
            fontsize=6,
            color="white",
            va="top",
        )

    fig.suptitle(title, fontsize=9)
    return fig


# ===========================================================================
# Check 1 — static atom
# ===========================================================================


def check_1_static_atom():
    """Single atom at rest: image must not change across frames."""
    print("\n=== Check 1: Static atom ===")

    atoms = [{"px": 0.0, "py": 0.0, "pz": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}]
    snap_g, _snap_e = collect_snapshots(atoms, SEQUENCE)

    # Quantitative check: all images should be identical
    imgs = [render(s, *pixel_grid([snap_g[0]], n_x=30, n_z=60)) for s in snap_g]
    diffs = [np.max(np.abs(imgs[i] - imgs[0])) for i in range(1, len(imgs))]
    max_diff = max(diffs)
    print(f"  Max pixel difference across frames: {max_diff:.2e}  (expect ~0)")

    fig = render_filmstrip_figure(
        snap_g,
        LABELS,
        "Check 1: Static atom (v=0, pos=0)\n"
        "All frames should be an identical Gaussian blob — "
        f"max frame-to-frame pixel diff = {max_diff:.2e}",
        channel="ground",
    )
    path = "check1_static_atom.png"
    fig.savefig(path, dpi=100)
    print(f"  Saved: {path}")
    plt.close(fig)
    return max_diff


# ===========================================================================
# Check 2 — drifting atom (vz + vx)
# ===========================================================================


def check_2_drifting_atom():
    """Atom with vz upward and vx sideways: blob must drift across frames."""
    print("\n=== Check 2: Drifting atom ===")

    VZ = +15e-3  # +15 mm/s upward  → +150 µm over 10 ms
    VX = +5e-3  # +5 mm/s sideways  → +50 µm over 10 ms
    print(f"  vz = +{VZ * 1e3:.0f} mm/s,  vx = +{VX * 1e3:.0f} mm/s")
    print(
        f"  Expected drift after 10 ms: Δz = {VZ * N_STEPS * DT * 1e6:.0f} µm,  "
        f"Δx = {VX * N_STEPS * DT * 1e6:.0f} µm"
    )

    atoms = [{"px": 0.0, "py": 0.0, "pz": 0.0, "vx": VX, "vy": 0.0, "vz": VZ}]
    snap_g, _snap_e = collect_snapshots(atoms, SEQUENCE)

    # Extract centroid positions to verify drift
    def centroid(branches):
        if not len(branches):
            return None, None
        w = branches[:, 3]  # per-branch weight
        x_c = np.sum(branches[:, 0] * w) / np.sum(w)
        z_c = np.sum(branches[:, 1] * w) / np.sum(w)
        return x_c * 1e6, z_c * 1e6  # µm

    print("  Centroids (x µm, z µm):")
    for i, (label, sg) in enumerate(zip(LABELS, snap_g)):
        xc, zc = centroid(sg)
        if xc is not None:
            expected_x = VX * i * DT * 1e6
            expected_z = VZ * i * DT * 1e6
            print(
                f"    {label}: ({xc:+7.1f}, {zc:+7.1f}) µm  "
                f"[expected ({expected_x:+7.1f}, {expected_z:+7.1f}) µm]"
            )

    fig = render_filmstrip_figure(
        snap_g,
        LABELS,
        f"Check 2: Drifting atom (vz=+{VZ * 1e3:.0f} mm/s up, vx=+{VX * 1e3:.0f} mm/s right)\n"
        "Blob must move upward in z and rightward in x with each step",
        channel="ground",
    )
    path = "check2_drifting_atom.png"
    fig.savefig(path, dpi=100)
    print(f"  Saved: {path}")
    plt.close(fig)


# ===========================================================================
# Check 3 — thermal cloud at 1 µK
# ===========================================================================


def check_3_thermal_cloud():
    """Gaussian cloud at 1 µK: cloud must expand ballistically over time."""
    print("\n=== Check 3: Thermal cloud at 1 µK ===")

    N_ATOMS = 200
    T_UK = 1.0e-6  # 1 µK
    SIGMA_POS = 100e-6  # 100 µm initial cloud radius (1σ)

    # 1-D rms velocity for Maxwell–Boltzmann: σ_v = sqrt(k_B T / m)
    sigma_v = np.sqrt(constants.k * T_UK / sim.MASS_ATOM)
    print(f"  N = {N_ATOMS} atoms, T = {T_UK * 1e6:.0f} µK")
    print(
        f"  σ_pos = {SIGMA_POS * 1e6:.0f} µm,  σ_v = {sigma_v * 1e3:.2f} mm/s per axis"
    )

    expected_sigma_final = np.sqrt(SIGMA_POS**2 + (sigma_v * N_STEPS * DT) ** 2)
    print(
        f"  Expected cloud σ after {N_STEPS} ms: "
        f"{expected_sigma_final * 1e6:.0f} µm  "
        f"({expected_sigma_final / SIGMA_POS:.2f}× initial)"
    )

    rng = np.random.default_rng(42)
    positions = rng.normal(0.0, SIGMA_POS, (N_ATOMS, 3))
    velocities = rng.normal(0.0, sigma_v, (N_ATOMS, 3))

    atoms = [
        {
            "px": positions[i, 0],
            "py": positions[i, 1],
            "pz": positions[i, 2],
            "vx": velocities[i, 0],
            "vy": velocities[i, 1],
            "vz": velocities[i, 2],
        }
        for i in range(N_ATOMS)
    ]

    snap_g, _snap_e = collect_snapshots(atoms, SEQUENCE)

    # Report approximate cloud size at each step from the x–z extent of branches
    print("  Approx cloud σ_z at each step:")
    for label, sg in zip(LABELS, snap_g):
        if len(sg):
            z_positions = sg[:, 1]  # column 1 = z
            sigma_z = np.std(z_positions)
            print(f"    {label}: σ_z = {sigma_z * 1e6:.1f} µm")

    fig = render_filmstrip_figure(
        snap_g,
        LABELS,
        f"Check 3: Thermal cloud ({N_ATOMS} atoms, T=1 µK, σ_pos=100 µm)\n"
        f"Cloud must expand from {SIGMA_POS * 1e6:.0f} µm "
        f"→ ~{expected_sigma_final * 1e6:.0f} µm over 10 ms",
        channel="ground",
        n_x=40,
        n_z=80,
        x_pad_min=50e-6,
        z_pad_min=50e-6,
        common_scale=False,  # per-panel scale: cloud thins as it expands
    )
    path = "check3_thermal_cloud.png"
    fig.savefig(path, dpi=100)
    print(f"  Saved: {path}")
    plt.close(fig)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    check_1_static_atom()
    check_2_drifting_atom()
    check_3_thermal_cloud()
    print("\nAll checks complete.")
