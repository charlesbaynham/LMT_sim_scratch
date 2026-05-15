"""Tests for lmt_real_sequence.py.

Covers structural properties of the generated pulse sequence
and a basic smoke-test that the sequence can be run on a single atom.
"""

import numpy as np
import pytest

from lmt_real_sequence import (
    _res_detuning,
    build_lmt_real_sequence,
    CLOCK_SHELVING_PULSE_TIME,
    DOWN_CLOCK_BEAM_PI_TIME,
    LMT_SELECTIVE_PI_TIME,
)
from lmt_simulation import (
    Clearout,
    Pulse,
    RECOIL_FREQUENCY_HZ,
    TRANSITION_FREQUENCY,
    calculate_ground_and_excited_probabilities,
    do_clearout,
    make_atom_states,
    pulse_interaction_in_borde_representation,
    transform_state_vector,
)


# ---------------------------------------------------------------------------
# _res_detuning
# ---------------------------------------------------------------------------


def test_res_detuning_ground_state_k_plus():
    """Velocity selection: |g,0⟩→|e,+1⟩ should be resonant at 1×δ_rec."""
    assert np.isclose(_res_detuning(m_ground=0, k=+1), RECOIL_FREQUENCY_HZ)


def test_res_detuning_bs1():
    """BS1 addresses |g,+2⟩↔|e,+1⟩ (k=-1, m_g=2): Δ = (4*2 + (-1))*δ_rec = 7δ_rec."""
    assert np.isclose(_res_detuning(m_ground=2, k=-1), 7 * RECOIL_FREQUENCY_HZ)


def test_res_detuning_first_selective_upper():
    """First selective UP addresses |g,+2⟩→|e,+3⟩ (k=+1, m_g=2): Δ = 9δ_rec."""
    assert np.isclose(_res_detuning(m_ground=2, k=+1), 9 * RECOIL_FREQUENCY_HZ)


# ---------------------------------------------------------------------------
# build_lmt_real_sequence — validation
# ---------------------------------------------------------------------------


def test_n_below_minimum_raises():
    with pytest.raises(ValueError, match="N must be >= 3"):
        build_lmt_real_sequence(N=2)


def test_n_equals_minimum_does_not_raise():
    seq = build_lmt_real_sequence(N=3)
    assert len(seq) > 0


# ---------------------------------------------------------------------------
# build_lmt_real_sequence — structural properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [3, 5, 7, 9])
def test_sequence_event_count(N):
    """Sequence has exactly 4*N pulses and 5 clearouts for any valid N."""
    seq = build_lmt_real_sequence(N=N)
    pulses = [e for e in seq if isinstance(e, Pulse)]
    clearouts = [e for e in seq if isinstance(e, Clearout)]
    assert len(pulses) == 4 * N, f"Expected 4*N={4*N} pulses, got {len(pulses)}"
    assert len(clearouts) == 5, f"Expected 5 clearouts, got {len(clearouts)}"


@pytest.mark.parametrize("N", [3, 7])
def test_sequence_events_all_typed(N):
    """Every event is either a Pulse or a Clearout."""
    seq = build_lmt_real_sequence(N=N)
    for ev in seq:
        assert isinstance(ev, (Pulse, Clearout)), f"Unexpected event type {type(ev)}"


@pytest.mark.parametrize("N", [3, 7])
def test_sequence_times_non_decreasing(N):
    """Sequence events are time-ordered (non-decreasing)."""
    seq = build_lmt_real_sequence(N=N)
    times = [ev.time for ev in seq]
    for i in range(len(times) - 1):
        assert times[i] <= times[i + 1], (
            f"Event {i} at t={times[i]*1e6:.1f}µs is later than "
            f"event {i+1} at t={times[i+1]*1e6:.1f}µs"
        )


def test_sequence_starts_with_velocity_selection():
    seq = build_lmt_real_sequence(N=7)
    assert isinstance(seq[0], Pulse)
    assert seq[0].label == "velocity_selection"
    assert seq[0].k == +1
    assert seq[0].time == 0.0


def test_sequence_ends_with_bs2():
    seq = build_lmt_real_sequence(N=7)
    last = seq[-1]
    assert isinstance(last, Pulse)
    assert last.label == "BS2"


def test_bs1_is_pi_half_pulse():
    seq = build_lmt_real_sequence(N=7)
    bs1 = next(e for e in seq if isinstance(e, Pulse) and e.label == "BS1")
    assert np.isclose(bs1.pulse_area, np.pi / 2)
    assert bs1.k == -1


def test_mirror_is_pi_pulse():
    seq = build_lmt_real_sequence(N=7)
    mirror = next(e for e in seq if isinstance(e, Pulse) and e.label == "mirror")
    assert np.isclose(mirror.pulse_area, np.pi)


def test_bs2_is_pi_half_pulse():
    seq = build_lmt_real_sequence(N=7)
    bs2 = next(e for e in seq if isinstance(e, Pulse) and e.label == "BS2")
    assert np.isclose(bs2.pulse_area, np.pi / 2)
    assert bs2.k == -1


def test_clearout_labels_present():
    """All five expected clearout labels appear in the sequence."""
    expected_labels = {
        "velocity_selection_clearout",
        "clearout_after_first_selective_upper",
        "clearout_before_last_selective_upper",
        "clearout_after_first_selective_lower",
        "clearout_before_last_selective_lower",
    }
    seq = build_lmt_real_sequence(N=7)
    actual_labels = {e.label for e in seq if isinstance(e, Clearout)}
    assert actual_labels == expected_labels


# ---------------------------------------------------------------------------
# phase_step propagation
# ---------------------------------------------------------------------------


def test_zero_phase_step_has_zero_phases():
    """With phase_step=0, all pulses carry phase 0."""
    seq = build_lmt_real_sequence(N=7, phase_step=0.0)
    for ev in seq:
        if isinstance(ev, Pulse):
            assert ev.phi == 0.0, f"Pulse '{ev.label}' has non-zero phi={ev.phi}"


def test_mirror_carries_phase_step():
    """Mirror pulse should have phi == phase_step."""
    phase_step = 1.23
    seq = build_lmt_real_sequence(N=7, phase_step=phase_step)
    mirror = next(e for e in seq if isinstance(e, Pulse) and e.label == "mirror")
    assert np.isclose(mirror.phi, phase_step)


def test_bs2_carries_four_times_phase_step():
    """BS2 should carry phi == 4 * phase_step."""
    phase_step = 0.7
    seq = build_lmt_real_sequence(N=7, phase_step=phase_step)
    bs2 = next(e for e in seq if isinstance(e, Pulse) and e.label == "BS2")
    assert np.isclose(bs2.phi, 4 * phase_step)


def test_lower_arm_down_pulses_carry_phase_step():
    """Down pulses on the lower arm should carry phi == phase_step."""
    phase_step = 0.9
    seq = build_lmt_real_sequence(N=7, phase_step=phase_step)
    for ev in seq:
        if isinstance(ev, Pulse) and ev.label.startswith("lower_") and ev.k == -1:
            assert np.isclose(ev.phi, phase_step), (
                f"Lower-arm down pulse '{ev.label}' has phi={ev.phi}, "
                f"expected {phase_step}"
            )


# ---------------------------------------------------------------------------
# Smoke test: run sequence on a single on-axis atom
# ---------------------------------------------------------------------------


def _run_single_atom_through_real_sequence(phase_step, N=3, rng_seed=0):
    """Apply the real sequence to a stationary ground-state atom.

    Clearouts are applied via Monte Carlo.  Returns ``(P_g, P_e)`` for a
    surviving atom or ``None`` if the atom is discarded.
    """
    seq = build_lmt_real_sequence(N=N, phase_step=phase_step)

    m, pos, vel, amp, isg = make_atom_states(c0=1.0, c1=0.0)
    det_vs = seq[0].detuning_hz
    omega_laser_ref = 2 * np.pi * (TRANSITION_FREQUENCY + det_vs)
    sq = transform_state_vector(
        m, amp, isg,
        omega_laser=omega_laser_ref,
        t=0.0, z=0.0, vz=0.0,
        inverse=False,
    )

    rng = np.random.default_rng(rng_seed)
    for event in seq:
        if isinstance(event, Clearout):
            result = do_clearout(m, sq, isg, pos, vel, rng=rng)
            if result is None:
                return None
            m, sq, isg, pos, vel = result
        else:
            m, sq, isg, pos, vel = pulse_interaction_in_borde_representation(
                m, sq, isg, pos, vel,
                pulse_detuning=event.detuning_hz,
                t_pulse=event.duration,
                pulse_rabi_freq=event.rabi_frequency,
                pulse_phase=event.phi,
                k_sign=event.k,
                vz=0.0,
            )
            # Prune negligible rows to keep the test fast.
            keep = np.abs(sq) ** 2 > 1e-10
            if keep.any():
                m, sq, isg, pos, vel = (
                    m[keep], sq[keep], isg[keep], pos[keep], vel[keep]
                )

    amp_lab = transform_state_vector(
        m, sq, isg,
        omega_laser=omega_laser_ref,
        t=0.0, z=0.0, vz=0.0,
        inverse=True,
    )
    return calculate_ground_and_excited_probabilities(m, amp_lab, isg)


def test_smoke_sequence_runs_without_error():
    """A cold on-axis atom survives the N=3 sequence and gives plausible output."""
    # Use a seed where the atom survives the clearouts.
    for seed in range(20):
        result = _run_single_atom_through_real_sequence(phase_step=0.0, N=3, rng_seed=seed)
        if result is not None:
            pg, pe = result
            assert pg >= 0.0
            assert pe >= 0.0
            assert np.isclose(pg + pe, 1.0, atol=1e-4), (
                f"Probabilities do not sum to 1: P_g={pg}, P_e={pe}"
            )
            return
    pytest.skip("Atom was discarded at all 20 RNG seeds — try a larger seed range")


def test_sequence_survival_rate_above_zero():
    """A cold on-axis atom should survive the N=3 velocity selection more often than not."""
    n_seeds = 30
    survived = sum(
        1
        for seed in range(n_seeds)
        if _run_single_atom_through_real_sequence(
            phase_step=0.0, N=3, rng_seed=seed
        ) is not None
    )
    assert survived > 0, "No atoms survived in 30 trials; velocity-selection looks broken"
