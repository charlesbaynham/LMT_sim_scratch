"""Validation tests for the staircase ARP pulses in :mod:`lmt_sim.arp`.

These checks pin the 2x2 ARP composer to the production row-based composer and
to analytic limits (no-frame-change consistency, Landau-Zener diabatic tail,
adiabatic full inversion).
"""

import numpy as np

from lmt_sim import arp
from lmt_sim import lmt_simulation as sim
from lmt_sim.lmt_simulation import (
    AtomState,
    RABI_FREQ,
    RECOIL_FREQUENCY_HZ,
    T_PI,
)
from lmt_sim.lmt_sequence import Pulse, run_pulse_sequence_in_borde_representation


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
    """Sum row amplitudes onto ``(c_excited @ m=1, c_ground @ m=0)``.

    For a single co-propagating arm the only populated physical states are
    ground(m=0) and excited(m=1); the row composer just keeps redundant copies.
    """
    is_g = state.internal_is_ground
    c_ground = state.amplitudes[is_g & (state.m_values == 0)].sum()
    c_excited = state.amplitudes[(~is_g) & (state.m_values == 1)].sum()
    return c_excited, c_ground


def test_arp_2x2_matches_row_composer():
    """The 2x2 ARP composer must equal the full row-based composer at small N.

    Run with discard_threshold=0.0 so no renormalisation perturbs amplitudes,
    then collapse the redundant rows. Both paths leave the amplitudes in the
    instantaneous Bordé frame (no inter-block frame change; the row path's
    change_laser_frequency_in_borde_representation does not touch amplitudes), so
    they compare directly. This is the standing tripwire against the two
    propagation paths drifting apart.
    """
    subpulses = arp.make_arp_subpulses(
        T=60e-6,
        delta_sweep_hz=1.0e5,
        omega0_hz=RABI_FREQ,
        n=3,
        sweep_shape="linear",
        omega_shape="const",
    )

    # 2x2 path: start in ground -> [c_excited, c_ground].
    U = arp.compose_arp_2x2(subpulses)
    c_e_2x2, c_g_2x2 = U @ np.array([0.0, 1.0], dtype=complex)

    # Row path: back-to-back Pulse objects with the same per-block parameters.
    pulse_sequence = [
        Pulse(
            k=+1,
            detuning_hz=sub.detuning_hz,
            phi=0.0,
            label=f"arp_{i}",
            rabi_frequency=sub.rabi_freq_hz,
            duration=sub.duration,
        )
        for i, sub in enumerate(subpulses)
    ]
    final_state, _, _ = run_pulse_sequence_in_borde_representation(
        _single_ground_state(),
        pulse_sequence,
        initial_velocity_z=0.0,
        discard_threshold=0.0,
    )
    c_e_row, c_g_row = _collapse_closed_two_level(final_state)

    assert np.isclose(c_e_2x2, c_e_row, atol=1e-10, rtol=0)
    assert np.isclose(c_g_2x2, c_g_row, atol=1e-10, rtol=0)


def test_arp_phi_bookkeeping_no_frame_change():
    """A zero-sweep staircase (constant detuning) must equal one block of total T.

    With delta_sweep=0 every block shares one detuning, so no frame change
    happens and the staircase is exp(-iH dt)^n = exp(-iH T): it must reproduce
    the single-block detuned-Rabi propagator to machine precision. If the
    inter-block phase bookkeeping were wrong this would fail.
    """
    T = 80e-6
    n = 17
    off_resonant_centre = 5.0e4  # deliberately off true resonance
    subpulses = arp.make_arp_subpulses(
        T=T,
        delta_sweep_hz=0.0,
        omega0_hz=RABI_FREQ,
        n=n,
        sweep_shape="linear",
        omega_shape="const",
        delta_centre_hz=off_resonant_centre,
    )
    U_staircase = arp.compose_arp_2x2(subpulses)
    U_single = sim._single_pulse_propagator_2x2(
        off_resonant_centre, T, RABI_FREQ, pulse_phase=0.0, k_sign=+1, m_ground=0
    )
    np.testing.assert_allclose(U_staircase, U_single, atol=1e-12, rtol=0)


def test_arp_ref_detuning_matches_lab_boundary_convention():
    """The ``ref_detuning_hz`` correction must use the lab-boundary sign/size.

    Relative to the un-referenced (Bordé-frame) composer, passing
    ``ref_detuning_hz`` applies exactly the laser-detuning factor that
    ``transform_state_vector(inverse=True)`` puts on at the lab boundary:
    excited ``exp(-i pi (Phi - ref*T))``, ground ``exp(+i pi (Phi - ref*T))`` with
    ``Phi = sum_k detuning_k * dt_k``. This pins the convention so it can't drift
    from the core transform.
    """
    subpulses = arp.make_arp_subpulses(
        T=120e-6,
        delta_sweep_hz=7.0e4,
        omega0_hz=RABI_FREQ,
        n=23,
        sweep_shape="tanh",
        omega_shape="sin2",
        delta_centre_hz=4.0e4,  # non-trivial, non-integer Phi
    )
    ref = 1.3e4
    U_none = arp.compose_arp_2x2(subpulses)
    U_ref = arp.compose_arp_2x2(subpulses, ref_detuning_hz=ref)

    phi_cycles = sum(s.detuning_hz * s.duration for s in subpulses)
    total_time = sum(s.duration for s in subpulses)
    residual = phi_cycles - ref * total_time
    expected = np.diag([np.exp(-1j * np.pi * residual), np.exp(+1j * np.pi * residual)])
    np.testing.assert_allclose(U_ref, expected @ U_none, atol=1e-13, rtol=0)

    # Cross-check the sign against transform_state_vector itself (omega_0 and the
    # spatial term zeroed, so only the detuning factor exp(-/+ i pi Phi) remains).
    probe = AtomState(
        m_values=np.array([1, 0]),
        positions=np.zeros((2, 3)),
        velocities=np.zeros((2, 3)),
        amplitudes=np.array([1.0 + 0j, 1.0 + 0j]),
        internal_is_ground=np.array([False, True]),
    )
    boundary = sim.transform_state_vector(
        probe,
        detuning_hz=phi_cycles / total_time,  # constant-rate frame with same Phi
        t=total_time,
        z=0.0,
        vz=0.0,
        omega_0=0.0,
        inverse=True,
    )
    np.testing.assert_allclose(
        boundary.amplitudes,
        [np.exp(-1j * np.pi * phi_cycles), np.exp(+1j * np.pi * phi_cycles)],
        atol=1e-12,
        rtol=0,
    )


def _arp_excited_population_and_phase(
    *, T, delta_sweep_hz, omega0_hz, n, sweep_shape, omega_shape, delta_centre_hz=None
):
    subpulses = arp.make_arp_subpulses(
        T=T,
        delta_sweep_hz=delta_sweep_hz,
        omega0_hz=omega0_hz,
        n=n,
        sweep_shape=sweep_shape,
        omega_shape=omega_shape,
        delta_centre_hz=delta_centre_hz,
    )
    # Reference to a fixed (error-independent) frame so the phase is meaningful.
    c_e, _ = arp.arp_excited_ground_amplitudes(subpulses, ref_detuning_hz=0.0)
    return np.abs(c_e) ** 2, np.angle(c_e)


def test_arp_staircase_convergence():
    """P_e and the imprinted phase must be stable as the staircase is refined."""
    common = dict(
        T=200e-6,
        delta_sweep_hz=2.0e4,
        omega0_hz=RABI_FREQ,
        sweep_shape="tanh",
        omega_shape="sin2",
    )
    pe_200, ph_200 = _arp_excited_population_and_phase(n=200, **common)
    pe_400, ph_400 = _arp_excited_population_and_phase(n=400, **common)

    assert abs(pe_400 - pe_200) < 1e-4
    # Phase difference, wrapped into (-pi, pi].
    dphi = (ph_400 - ph_200 + np.pi) % (2 * np.pi) - np.pi
    assert abs(dphi) < 1e-2


def test_arp_adiabatic_inversion():
    """A strongly adiabatic ARP pulse must fully invert the population.

    These parameters sit on a robust adiabatic plateau (P_e ~ 1.0, insensitive to
    +/-10% changes in T / delta_sweep / omega0), not a diabatic-residual fringe --
    a genuine test of adiabatic passage. (The narrower, shorter sweep the finding
    doc cites tops out at P_e ~ 0.998, i.e. not fully adiabatic.)
    """
    pe, _ = _arp_excited_population_and_phase(
        T=600e-6,
        delta_sweep_hz=3.0e4,
        omega0_hz=1.5 * RABI_FREQ,
        n=400,
        sweep_shape="tanh",
        omega_shape="sin2",
    )
    assert pe > 0.999


def test_arp_landau_zener_diabatic_tail():
    """A constant-Omega wide linear sweep approaches the Landau-Zener transfer.

    Mapping the code's conventions (omega_ab = pi*rabi, Delta = 2*pi*detuning)
    onto the standard LZ Hamiltonian gives gap 2*omega_ab = 2*pi*rabi and bias
    rate alpha = 2*pi*delta_sweep/T, so the single-pass transfer probability is
    1 - exp(-pi**2 * omega0**2 * T / delta_sweep).
    """
    T = 200e-6
    omega0_hz = 1.19e4
    delta_sweep_hz = 4.0e5
    pe, _ = _arp_excited_population_and_phase(
        T=T,
        delta_sweep_hz=delta_sweep_hz,
        omega0_hz=omega0_hz,
        n=3000,
        sweep_shape="linear",
        omega_shape="const",
    )
    expected = 1.0 - np.exp(-(np.pi**2) * omega0_hz**2 * T / delta_sweep_hz)
    assert abs(pe - expected) < 0.03


def test_arp_symmetric_about_resonance():
    """Auto-centred ARP is symmetric: P_e is even in the static detuning error."""
    resonant = arp.resonant_centre_detuning_hz()
    delta_err = 3.0e3
    common = dict(
        T=200e-6,
        delta_sweep_hz=2.0e4,
        omega0_hz=RABI_FREQ,
        n=400,
        sweep_shape="tanh",
        omega_shape="sin2",
    )
    pe_plus, _ = _arp_excited_population_and_phase(
        delta_centre_hz=resonant + delta_err, **common
    )
    pe_minus, _ = _arp_excited_population_and_phase(
        delta_centre_hz=resonant - delta_err, **common
    )
    assert abs(pe_plus - pe_minus) < 5e-3


def test_resonant_centre_is_one_recoil():
    """For m=0, k=+1, vz=0 the resonant centre is the recoil frequency."""
    assert np.isclose(arp.resonant_centre_detuning_hz(), RECOIL_FREQUENCY_HZ)


def test_plain_pi_pulse_via_n1_composer():
    """A single-block ARP composer reproduces a resonant pi pulse (full inversion)."""
    subpulses = arp.make_arp_subpulses(
        T=T_PI,
        delta_sweep_hz=0.0,
        omega0_hz=RABI_FREQ,
        n=1,
        omega_shape="const",
    )
    c_e, _ = arp.arp_excited_ground_amplitudes(subpulses)
    assert np.abs(c_e) ** 2 > 0.999
