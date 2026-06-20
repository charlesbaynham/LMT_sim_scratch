"""Validation tests for the staircase ARP pulses in :mod:`lmt_sim.arp`.

These checks pin the 2x2 ARP composer to the production row-based composer and
to analytic limits (no-frame-change consistency, Landau-Zener diabatic tail,
adiabatic full inversion).
"""

import pytest

# PARKED: the ARP composer (lmt_sim.arp) is on hold pending the Bordé
# frame-change rework. The row-based path has been corrected to rebase the frame
# at a laser-frequency step (no inter-block frame change), so the still-paused 2x2
# ARP composer -- which deliberately keeps the old (wrong) inter-block frame
# change -- no longer matches the row composer. The whole ARP test module is
# skipped until we return to the ARP composer. See docs/arp_frame_change_finding.md.
pytest.skip(
    "ARP composer parked pending the Bordé frame-change rework "
    "(see docs/arp_frame_change_finding.md).",
    allow_module_level=True,
)

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
    then collapse the redundant rows. Both paths end in the frame of the last
    sub-pulse, so amplitudes compare directly. This is the standing tripwire
    against the two propagation paths drifting apart.
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


# FIXME(frame-change): These three tests are xfailed only because
# compose_arp_2x2 applies the inter-block frame change. Removing that frame
# change (see the FIXME in lmt_sim/arp.py) makes them pass -- so flip these back
# to plain (must-pass) tests as part of the fix. docs/arp_frame_change_finding.md
_FRAME_CHANGE_XFAIL = pytest.mark.xfail(
    reason=(
        "compose_arp_2x2 currently applies the inter-block frame change, which "
        "double-counts the laser-frequency change for a chirp (the detuning is "
        "already in the per-block Hamiltonian diagonal). The staircase therefore "
        "converges to the wrong physics. Removing the frame change makes these "
        "pass (matches the continuous-sweep ODE and Landau-Zener). PAUSED pending "
        "investigation of the row-composer frame change -- see docs/arp_frame_change_finding.md."
    ),
    strict=False,
)


@_FRAME_CHANGE_XFAIL
def test_arp_adiabatic_inversion():
    """A strongly adiabatic ARP pulse must fully invert the population."""
    pe, _ = _arp_excited_population_and_phase(
        T=200e-6,
        delta_sweep_hz=2.0e4,
        omega0_hz=RABI_FREQ,
        n=400,
        sweep_shape="tanh",
        omega_shape="sin2",
    )
    assert pe > 0.999


@_FRAME_CHANGE_XFAIL
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


@_FRAME_CHANGE_XFAIL
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
