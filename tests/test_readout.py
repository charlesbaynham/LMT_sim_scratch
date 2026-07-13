"""Tests for the velocity-resolved excited-state readout (lmt_sim.readout)."""

import numpy as np
import pytest

import lmt_sim.lmt_simulation as sim
import lmt_sim.readout as readout

READOUT_DURATION = 380e-6
READOUT_RABI = 1 / (2 * READOUT_DURATION)  # pi pulse


def make_state(rows, v0=0.0):
    """Build an AtomState from (m, is_ground, amplitude, z_position) rows.

    Velocities respect the v = v0 + m * v_recoil invariant.
    """
    m = np.array([r[0] for r in rows], dtype=int)
    is_ground = np.array([r[1] for r in rows], dtype=bool)
    amps = np.array([r[2] for r in rows], dtype=np.complex128)
    z = np.array([r[3] for r in rows], dtype=float)
    positions = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
    vz = v0 + m * sim.RECOIL_VELOCITY
    velocities = np.column_stack((np.zeros_like(vz), np.zeros_like(vz), vz))
    return sim.AtomState(
        m_values=m,
        positions=positions,
        velocities=velocities,
        amplitudes=amps,
        internal_is_ground=is_ground,
    )


def test_ground_manifold_is_invisible():
    """A purely ground-state atom produces zero signal at every detuning."""
    state = make_state([(0, True, 0.8, 0.0), (2, True, 0.6, 1e-6)])
    detunings = np.linspace(-50e3, 50e3, 41)
    m_classes, signal = readout.simulate_excited_state_readout(
        state,
        detunings,
        pulse_duration=READOUT_DURATION,
        pulse_rabi_frequency=READOUT_RABI,
    )
    assert len(m_classes) == 0
    assert signal.shape == (41, 0)
    assert np.all(signal.sum(axis=1) == 0.0)


@pytest.mark.parametrize("k_sign", [+1, -1])
@pytest.mark.parametrize("m_excited", [1, 3, -1])
def test_single_class_peaks_at_resonance(k_sign, m_excited):
    """A lone excited class transfers ~fully at its resonance detuning and
    negligibly one recoil-class away."""
    state = make_state([(m_excited, False, 1.0, 0.0)])
    resonance = readout.readout_resonance_detuning_hz(m_excited, k_sign=k_sign)
    detunings = resonance + np.linspace(-500.0, 500.0, 21)
    m_classes, signal = readout.simulate_excited_state_readout(
        state,
        detunings,
        pulse_duration=READOUT_DURATION,
        pulse_rabi_frequency=READOUT_RABI,
        k_sign=k_sign,
    )
    assert list(m_classes) == [m_excited]
    total = signal.sum(axis=1)
    # Peak of the sweep is at the predicted resonance (grid centre) and is a
    # full pi-pulse transfer.
    assert np.argmax(total) == 10
    assert total[10] == pytest.approx(1.0, abs=1e-9)
    # Two recoil frequencies away (the neighbouring class position) the
    # transfer is tiny -- this is the sub-recoil resolution.
    far = readout.simulate_excited_state_readout(
        state,
        np.array([resonance + 2 * sim.RECOIL_FREQUENCY_HZ]),
        pulse_duration=READOUT_DURATION,
        pulse_rabi_frequency=READOUT_RABI,
        k_sign=k_sign,
    )[1]
    assert far.sum() < 0.05


def test_resonance_shifts_with_atom_velocity():
    """An atom with base velocity v0 peaks at the Doppler-shifted resonance."""
    v0 = 3e-3  # 3 mm/s
    state = make_state([(1, False, 1.0, 0.0)], v0=v0)
    resonance = readout.readout_resonance_detuning_hz(1, k_sign=+1, v0=v0)
    # v0 / lambda ~ 4.3 kHz here, so getting the peak right at the shifted
    # position (within a fraction of the linewidth) checks the Doppler sign.
    detunings = resonance + np.linspace(-2e3, 2e3, 81)
    _, signal = readout.simulate_excited_state_readout(
        state,
        detunings,
        pulse_duration=READOUT_DURATION,
        pulse_rabi_frequency=READOUT_RABI,
        k_sign=+1,
        vz=v0,
    )
    total = signal.sum(axis=1)
    assert abs(detunings[np.argmax(total)] - resonance) < 100.0
    assert total.max() > 0.99


def test_matches_unmerged_canonical_pulse():
    """The merged fast path must reproduce the canonical row-by-row pulse
    applied to the unmerged state, with coherent per-class sums."""
    rng = np.random.default_rng(seed=42)
    rows = []
    # Several classes, each split across multiple rows with random amplitudes
    # and positions (the row structure a real sequence produces).
    for m, is_ground in [
        (1, False),
        (1, False),
        (3, False),
        (3, False),
        (3, False),
        (-1, False),
        (0, True),
        (2, True),
    ]:
        amp = rng.normal(scale=0.3) + 1j * rng.normal(scale=0.3)
        rows.append((m, is_ground, amp, rng.normal(scale=1e-4)))
    state = make_state(rows, v0=1e-3)

    detunings = np.linspace(-30e3, 40e3, 29)
    m_classes, signal = readout.simulate_excited_state_readout(
        state,
        detunings,
        pulse_duration=READOUT_DURATION,
        pulse_rabi_frequency=READOUT_RABI,
        k_sign=+1,
        vz=1e-3,
        probe_shift_coefficient=1.8e-5,
    )

    # Brute force: strip ground rows but do NOT merge; apply the canonical
    # interaction per detuning; coherently sum ground amplitudes per class.
    stripped = readout.remove_ground_rows(state)
    for i, delta in enumerate(detunings):
        after = sim.pulse_interaction_in_borde_representation(
            stripped,
            pulse_detuning=float(delta),
            t_pulse=READOUT_DURATION,
            pulse_rabi_freq=READOUT_RABI,
            k_sign=+1,
            vz=1e-3,
            probe_shift_coefficient=1.8e-5,
        )
        for j, m_class in enumerate(m_classes):
            sel = (
                after.internal_is_ground
                & (after.m_values == m_class - 1)  # ground output of class m
            )
            expected = np.abs(after.amplitudes[sel].sum()) ** 2
            assert signal[i, j] == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_signal_bounded_by_excited_population():
    """Total transferred signal never exceeds the excited population."""
    state = make_state(
        [(1, False, 0.5, 0.0), (3, False, 0.5j, 0.0), (0, True, 0.7, 0.0)]
    )
    _, p_e = sim.calculate_ground_and_excited_probabilities(state)
    detunings = np.linspace(-60e3, 60e3, 101)
    _, signal = readout.simulate_excited_state_readout(
        state,
        detunings,
        pulse_duration=READOUT_DURATION,
        pulse_rabi_frequency=READOUT_RABI,
    )
    total = signal.sum(axis=1)
    assert np.all(total <= p_e + 1e-12)
    # ... and the resonant shots really do recover each class's population.
    peak_1 = total[
        np.argmin(np.abs(detunings - readout.readout_resonance_detuning_hz(1)))
    ]
    assert peak_1 == pytest.approx(0.25, abs=0.01)


def test_remove_ground_rows_keeps_absolute_weight():
    state = make_state([(0, True, 0.6, 0.0), (1, False, 0.8, 0.0)])
    stripped = readout.remove_ground_rows(state)
    assert len(stripped.amplitudes) == 1
    assert np.abs(stripped.amplitudes[0]) ** 2 == pytest.approx(0.64)


def test_merge_is_coherent():
    """Two same-class rows with opposite phases cancel in the merged state."""
    state = make_state([(1, False, 0.5, 0.0), (1, False, -0.5, 1e-5)])
    merged = readout.merge_rows_by_momentum_class(state)
    assert len(merged.amplitudes) == 1
    assert merged.amplitudes[0] == pytest.approx(0.0)
