"""Tests for composite (multi-block / ARP) pulses in the sequence model.

A :class:`CompositePulse` applies a whole staircase of fixed-parameter sub-blocks
as a SINGLE branching event (ground at ``m``, excited at ``m+-k``) using the
composed 2x2 propagator from :mod:`lmt_sim.arp`, instead of branching every
sub-pulse (which would blow the row count up to ``2**N``). These tests pin the
sequence path to the bare 2x2 composer, check the one-branching behaviour and the
configurable momentum-kick split, and exercise the trajectory overlay.
"""

import numpy as np

from lmt_sim import arp
from lmt_sim import lmt_simulation as sim
from lmt_sim.lmt_simulation import AtomState, RABI_FREQ, RECOIL_VELOCITY
from lmt_sim.lmt_sequence import (
    Clearout,
    CompositePulse,
    Freefall,
    Pulse,
    compute_spacetime_trajectory,
    run_pulse_sequence_in_borde_representation,
    run_pulse_sequence_in_lab_frame,
)


def _single_ground_state():
    """A single ground-state row at m=0, t=0, z=0 (already in the Bordé frame)."""
    return AtomState(
        m_values=np.array([0], dtype=int),
        positions=np.zeros((1, 3), dtype=float),
        velocities=np.zeros((1, 3), dtype=float),
        amplitudes=np.array([1.0 + 0.0j], dtype=complex),
        internal_is_ground=np.array([True], dtype=bool),
    )


def _collapse_closed_two_level(state):
    """Sum row amplitudes onto ``(c_excited @ m=1, c_ground @ m=0)``."""
    is_g = state.internal_is_ground
    c_ground = state.amplitudes[is_g & (state.m_values == 0)].sum()
    c_excited = state.amplitudes[(~is_g) & (state.m_values == 1)].sum()
    return c_excited, c_ground


def _nontrivial_subpulses():
    """A sweep with delta_start != delta_end and phi_cycles != delta_end*T."""
    return arp.make_arp_subpulses(
        T=60e-6,
        delta_sweep_hz=1.0e5,
        omega0_hz=RABI_FREQ,
        n=5,
        sweep_shape="linear",
        omega_shape="const",
        delta_centre_hz=4.0e4,
    )


def test_composite_pulse_lab_frame_matches_bare_composer():
    """T1: full sequence (lab boundary) == bare composer + the lab-frame phase.

    The headline tripwire. The sequence does lab->Bordé, the composite
    interaction, the staircase integral fold, then Bordé->lab. The reference is
    the bare ``compose_arp_2x2`` result pushed through the SAME inverse lab
    boundary (constant-rate frame carrying the genuine ``Phi = phi_cycles``). If
    the frame fold used ``delta_start*T`` instead of ``phi_cycles``, or recorded
    the wrong end detuning, the imprinted phase would differ and this fails.
    """
    subpulses = _nontrivial_subpulses()
    cp = CompositePulse(subpulses=tuple(subpulses), k=+1)

    phi_cycles = sum(s.detuning_hz * s.duration for s in subpulses)
    total_time = sum(s.duration for s in subpulses)
    # Sanity: the sweep really is non-trivial.
    assert not np.isclose(subpulses[0].detuning_hz, subpulses[-1].detuning_hz)
    assert not np.isclose(phi_cycles, subpulses[-1].detuning_hz * total_time)

    # Reference: bare Bordé-frame amplitudes through the inverse lab boundary.
    c_e_borde, c_g_borde = arp.compose_arp_2x2(subpulses) @ np.array(
        [0.0, 1.0], dtype=complex
    )
    probe = AtomState(
        m_values=np.array([1, 0]),
        positions=np.zeros((2, 3)),
        velocities=np.zeros((2, 3)),
        amplitudes=np.array([c_e_borde, c_g_borde]),
        internal_is_ground=np.array([False, True]),
    )
    ref = sim.transform_state_vector(
        probe,
        detuning_hz=phi_cycles / total_time,  # constant-rate frame with same Phi
        t=total_time,
        t_ref=0.0,
        accumulated_detuning_cycles=0.0,
        z=0.0,
        vz=0.0,
        inverse=True,
    )
    c_e_ref, c_g_ref = ref.amplitudes

    # Sequence path.
    final, _, _ = run_pulse_sequence_in_lab_frame(
        _single_ground_state(), [cp], initial_velocity_z=0.0, discard_threshold=0.0
    )
    c_e_seq, c_g_seq = _collapse_closed_two_level(final)

    assert np.isclose(c_g_seq, c_g_ref, atol=1e-9, rtol=0)
    assert np.isclose(c_e_seq, c_e_ref, atol=1e-9, rtol=0)


def test_composite_pulse_borde_frame_matches_bare_composer():
    """T2: Bordé-frame sequence == bare ``compose_arp_2x2`` (isolates the interaction)."""
    subpulses = _nontrivial_subpulses()
    cp = CompositePulse(subpulses=tuple(subpulses), k=+1)

    c_e_2x2, c_g_2x2 = arp.compose_arp_2x2(subpulses) @ np.array(
        [0.0, 1.0], dtype=complex
    )

    final, _, _ = run_pulse_sequence_in_borde_representation(
        _single_ground_state(), [cp], initial_velocity_z=0.0, discard_threshold=0.0
    )
    c_e_row, c_g_row = _collapse_closed_two_level(final)

    assert np.isclose(c_e_2x2, c_e_row, atol=1e-10, rtol=0)
    assert np.isclose(c_g_2x2, c_g_row, atol=1e-10, rtol=0)


def test_composite_arp_pi_pulse_single_branching():
    """T3: an adiabatic ARP pi pulse moves m=0->m=1 with ONE branching event."""
    cp = CompositePulse.arp(
        T=600e-6,
        delta_sweep_hz=3.0e4,
        omega0_hz=1.5 * RABI_FREQ,
        k=+1,
        n=400,
        sweep_shape="tanh",
        omega_shape="sin2",
    )

    final, _, _ = run_pulse_sequence_in_borde_representation(
        _single_ground_state(), [cp], initial_velocity_z=0.0, discard_threshold=0.0
    )

    # One branching event -> exactly 2 rows, not 2**n.
    assert final.amplitudes.shape[0] == 2

    pops = np.abs(final.amplitudes) ** 2
    is_g = final.internal_is_ground
    pop_excited_m1 = pops[(~is_g) & (final.m_values == 1)].sum()
    pop_ground_m0 = pops[is_g & (final.m_values == 0)].sum()
    assert pop_excited_m1 > 0.999
    assert pop_ground_m0 < 1e-3


def test_composite_arp_pi_pulse_then_clearout_keeps_excited():
    """T3 (cont.): a Clearout after the inverting pulse keeps the excited m=1 arm."""
    cp = CompositePulse.arp(
        T=600e-6,
        delta_sweep_hz=3.0e4,
        omega0_hz=1.5 * RABI_FREQ,
        k=+1,
        n=400,
    )
    rng = np.random.default_rng(0)
    final, _, _ = run_pulse_sequence_in_borde_representation(
        _single_ground_state(),
        [cp, Clearout(0.0)],
        initial_velocity_z=0.0,
        discard_threshold=1e-9,
        rng=rng,
    )
    # The clearout projects onto excited; only the excited m=1 branch survives.
    assert np.all(~final.internal_is_ground)
    assert np.all(final.m_values == 1)
    np.testing.assert_allclose(
        final.velocities[:, 2], RECOIL_VELOCITY, rtol=0, atol=1e-12
    )


def test_composite_pulse_trailing_noops_are_consistent():
    """T4: same-frequency zero-duration trailing events do not change the result."""
    subpulses = _nontrivial_subpulses()
    cp = CompositePulse(subpulses=tuple(subpulses), k=+1)
    delta_end = cp.end_detuning_hz

    def excited_fraction(sequence):
        final, _, _ = run_pulse_sequence_in_lab_frame(
            _single_ground_state(),
            sequence,
            initial_velocity_z=0.0,
            discard_threshold=0.0,
        )
        g, e = sim.calculate_ground_and_excited_probabilities(final)
        return e / (g + e)

    base = excited_fraction([cp])
    with_freefall = excited_fraction([cp, Freefall(0.0)])
    # A zero-duration pulse at the sweep's end detuning is a same-frequency no-op.
    with_pulse = excited_fraction(
        [
            cp,
            Pulse(
                k=+1,
                detuning_hz=delta_end,
                phi=0.0,
                label="noop",
                rabi_frequency=RABI_FREQ,
                duration=0.0,
            ),
        ]
    )
    assert np.isclose(base, with_freefall, atol=1e-12)
    assert np.isclose(base, with_pulse, atol=1e-12)


def test_composite_pulse_momentum_kick_fraction_position():
    """T5: the excited-branch position follows the kick-fraction split."""
    T = 600e-6
    v_recoil = RECOIL_VELOCITY
    for frac in (0.0, 0.5, 1.0):
        cp = CompositePulse.arp(
            T=T,
            delta_sweep_hz=3.0e4,
            omega0_hz=1.5 * RABI_FREQ,
            k=+1,
            n=400,
            momentum_kick_fraction=frac,
        )
        final, _, _ = run_pulse_sequence_in_borde_representation(
            _single_ground_state(), [cp], initial_velocity_z=0.0, discard_threshold=0.0
        )
        is_g = final.internal_is_ground
        z_excited = final.positions[(~is_g) & (final.m_values == 1), 2][0]
        # v_old = 0 (m=0), v_new = v_recoil (m=1): z = v_old*frac*T + v_new*(1-frac)*T.
        expected = v_recoil * (1.0 - frac) * T
        assert np.isclose(z_excited, expected, rtol=1e-12, atol=1e-18)


def test_composite_pulse_trajectory_full_flip():
    """T6: the trajectory overlay flips one cloud and uses the kick-fraction split."""
    T = 600e-6
    cp = CompositePulse.arp(
        T=T,
        delta_sweep_hz=3.0e4,
        omega0_hz=1.5 * RABI_FREQ,
        k=+1,
        n=400,
        momentum_kick_fraction=0.5,
    )
    clouds, _ = compute_spacetime_trajectory([cp], plot=False)
    assert len(clouds) == 1
    cloud = clouds[0]
    assert cloud.m[-1] == 1
    assert cloud.is_ground[-1] is False
    # z at the pulse end: v_old*frac*T + v_new*(1-frac)*T, v_old=0, v_new=v_recoil.
    expected = RECOIL_VELOCITY * 0.5 * T
    assert np.isclose(cloud.z[-1], expected, rtol=1e-12, atol=1e-18)


def test_composite_pulse_trajectory_partial_transfer_forks():
    """T6 (cont.): a partial-transfer composite forks into drifter + flipper."""
    # A short, non-adiabatic sweep gives an intermediate transfer probability.
    cp = CompositePulse.arp(
        T=80e-6,
        delta_sweep_hz=6.0e4,
        omega0_hz=1.0 * RABI_FREQ,
        k=+1,
        n=200,
        sweep_shape="tanh",
        omega_shape="sin2",
    )
    p = abs(arp.compose_arp_2x2(cp.subpulses, k_sign=+1, vz=0.0, m_ground=0)[0, 1]) ** 2
    # Guard the premise: genuinely intermediate so the heuristic forks it.
    assert 0.25 < p < 0.75
    clouds, _ = compute_spacetime_trajectory([cp], plot=False, flip_threshold=0.75)
    assert len(clouds) == 2
    finals = sorted(cloud.m[-1] for cloud in clouds)
    assert finals == [0, 1]
