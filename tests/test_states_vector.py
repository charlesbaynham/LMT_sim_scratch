import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lmt_sim.lmt_simulation as sim


def _total_population(state):
    ground_prob, excited_prob = sim.calculate_ground_and_excited_probabilities(state)
    return ground_prob + excited_prob


def run_clearout_trials(sequence_fn, n_trials, rng=None):
    """Run ``sequence_fn(rng)`` ``n_trials`` times.

    The closure must return either ``None`` (atom discarded mid-sequence)
    or the final state.

    For surviving runs, each contributes its quantum-mechanical
    :math:`P_\\text{g}` and :math:`P_\\text{e}` weighted by
    :math:`1/n_\\text{trials}` (so the result equals the deterministic
    population breakdown in the limit :math:`n_\\text{trials} \\to \\infty`).

    Parameters
    ----------
    sequence_fn : callable
        ``sequence_fn(rng)`` -> ``None`` or AtomState.
    n_trials : int
        Number of Monte-Carlo trials to run.
    rng : np.random.Generator, optional
        Random-number generator.  If ``None``, ``np.random.default_rng()``
        is used.

    Returns
    -------
    tuple
        ``(p_ground, p_excited, p_discarded)`` -- population fractions.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    ground_tally = 0.0
    excited_tally = 0.0
    discard_tally = 0.0

    for _ in range(n_trials):
        result = sequence_fn(rng)
        if result is None:
            discard_tally += 1.0
        else:
            p_g, p_e = sim.calculate_ground_and_excited_probabilities(result)
            ground_tally += p_g
            excited_tally += p_e

    return (
        ground_tally / n_trials,
        excited_tally / n_trials,
        discard_tally / n_trials,
    )


@pytest.mark.parametrize("seed", range(100))
def test_mz_randomized_population_conserved_every_step(seed):
    rng = np.random.default_rng(seed)

    c0 = rng.normal() + 1j * rng.normal()
    c1 = rng.normal() + 1j * rng.normal()
    norm = np.sqrt(np.abs(c0) ** 2 + np.abs(c1) ** 2)
    c0 = c0 / norm
    c1 = c1 / norm

    phi = rng.uniform(0.0, 2.0 * np.pi)
    detuning_hz = rng.uniform(-300e3, 300e3)
    pulse_1 = rng.uniform(0.1 * sim.T_PI, 1.9 * sim.T_PI)
    pulse_2 = rng.uniform(0.1 * sim.T_PI, 1.9 * sim.T_PI)
    pulse_3 = rng.uniform(0.1 * sim.T_PI, 1.9 * sim.T_PI)
    time_between = rng.uniform(0.0, 300e-6)

    state = sim.make_atom_states(
        position_z=0.0,
        initial_velocity_z=0.0,
        c0=c0,
        c1=c1,
    )

    assert _total_population(state) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    transform_detuning_hz = detuning_hz
    current_time = 0.0

    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=current_time,
        z=0.0,
        vz=0.0,
        inverse=False,
    )

    assert _total_population(state) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    state = sim.pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=pulse_1,
        pulse_rabi_freq=sim.RABI_FREQ,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=sim.K_WAVEVECTOR,
        vz=0.0,
    )
    current_time += pulse_1

    assert _total_population(state) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    state = sim.propagate_states_in_borde_representation(
        state,
        time_of_propegation=time_between,
        detuning_hz=detuning_hz,
        vz=0.0,
        k_wavevector=sim.K_WAVEVECTOR,
    )
    current_time += time_between

    assert _total_population(state) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    state = sim.pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=pulse_2,
        pulse_rabi_freq=sim.RABI_FREQ,
        pulse_phase=phi,
        k_sign=+1,
        k_wavevector=sim.K_WAVEVECTOR,
        vz=0.0,
    )
    current_time += pulse_2

    assert _total_population(state) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    state = sim.propagate_states_in_borde_representation(
        state,
        time_of_propegation=time_between,
        detuning_hz=detuning_hz,
        vz=0.0,
        k_wavevector=sim.K_WAVEVECTOR,
    )
    current_time += time_between

    assert _total_population(state) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    state = sim.pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=pulse_3,
        pulse_rabi_freq=sim.RABI_FREQ,
        pulse_phase=4.0 * phi,
        k_sign=+1,
        k_wavevector=sim.K_WAVEVECTOR,
        vz=0.0,
    )
    current_time += pulse_3

    assert _total_population(state) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=current_time,
        z=0.0,
        vz=0.0,
        inverse=True,
    )

    assert _total_population(state) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )


# ---------------------------------------------------------------------------
# Tests for gaussian_rabi (pure-physics primitive)
# ---------------------------------------------------------------------------


def test_gaussian_rabi_on_axis():
    """gaussian_rabi at r=0 returns exactly Omega_0."""
    omega_0 = 1000.0
    w = 5e-3
    pos = np.array([[0.0, 0.0, 0.0]])
    result = sim.gaussian_rabi(pos, omega_0, w)
    assert result[0] == pytest.approx(omega_0, rel=1e-12)


def test_gaussian_rabi_at_waist():
    """gaussian_rabi at r=w returns Omega_0 / e (within 0.1%)."""
    omega_0 = 1000.0
    w = 5e-3
    pos = np.array([[w, 0.0, 0.0]])
    result = sim.gaussian_rabi(pos, omega_0, w)
    assert result[0] == pytest.approx(omega_0 / np.e, rel=1e-6)


def test_gaussian_rabi_far_from_axis():
    """gaussian_rabi at r=5w is negligible compared to Omega_0."""
    omega_0 = 1000.0
    w = 5e-3
    pos = np.array([[5 * w, 0.0, 0.0]])
    result = sim.gaussian_rabi(pos, omega_0, w)
    assert result[0] < 1e-10 * omega_0


# ---------------------------------------------------------------------------
# Test array-Rabi path matches scalar-Rabi path
# ---------------------------------------------------------------------------


def test_array_rabi_matches_scalar():
    """pulse_interaction_in_borde_representation with array Rabi == scalar Rabi."""
    state = sim.make_atom_states(c0=1.0, c1=0.0)
    transform_detuning_hz = 0.0
    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=0.0,
        z=0.0,
        vz=0.0,
        inverse=False,
    )

    # Scalar path
    out_scalar = sim.pulse_interaction_in_borde_representation(
        state,
        pulse_rabi_freq=sim.RABI_FREQ,
        pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
        t_pulse=sim.T_PI,
        pulse_phase=0.0,
        k_sign=int(+1),
        k_wavevector=sim.K_WAVEVECTOR,
        vz=0.0,
    )

    # Array path: same value broadcast to (N,)
    N = len(state.m_values)
    rabi_array = np.full(N, sim.RABI_FREQ)
    out_array = sim.pulse_interaction_in_borde_representation(
        state,
        pulse_rabi_freq=rabi_array,
        pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
        t_pulse=sim.T_PI,
        pulse_phase=0.0,
        k_sign=int(+1),
        k_wavevector=sim.K_WAVEVECTOR,
        vz=0.0,
    )

    np.testing.assert_allclose(out_scalar.amplitudes, out_array.amplitudes, rtol=1e-12)
    np.testing.assert_array_equal(out_scalar.m_values, out_array.m_values)


# ---------------------------------------------------------------------------
# Gaussian-pulse end-to-end tests
# ---------------------------------------------------------------------------


def test_gaussian_pulse_on_axis_pi():
    """do_gaussian_pulse with atom at beam centre gives full population transfer."""
    state = sim.make_atom_states(c0=1.0, c1=0.0)
    transform_detuning_hz = 0.0
    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=0.0,
        z=0.0,
        vz=0.0,
        inverse=False,
    )

    state = sim.do_gaussian_pulse(
        state,
        pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
        t_pulse=sim.T_PI,
        on_axis_rabi_freq=sim.RABI_FREQ,
        beam_waist=1.0,  # large beam: essentially flat-top at r=0
        vz=0.0,
    )

    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=sim.T_PI,
        z=0.0,
        vz=0.0,
        inverse=True,
    )
    _, excited_prob = sim.calculate_ground_and_excited_probabilities(state)
    assert excited_prob == pytest.approx(1.0, abs=1e-4)


def test_gaussian_pulse_at_waist():
    """do_gaussian_pulse with atom at r=w gives excitation sin^2(pi / (2e)) ~ 0.310."""
    w = 1e-3
    state = sim.make_atom_states(position_x=w, c0=1.0, c1=0.0)
    transform_detuning_hz = 0.0
    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=0.0,
        z=0.0,
        vz=0.0,
        inverse=False,
    )

    state = sim.do_gaussian_pulse(
        state,
        pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
        t_pulse=sim.T_PI,
        on_axis_rabi_freq=sim.RABI_FREQ,
        beam_waist=w,
        vz=0.0,
    )

    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=sim.T_PI,
        z=0.0,
        vz=0.0,
        inverse=True,
    )
    _, excited_prob = sim.calculate_ground_and_excited_probabilities(state)
    # At r=w: Rabi = Omega_0/e, pulse area = Omega_0/e * T_pi = pi/e
    # excitation = sin^2(pi / (2e))
    expected = np.sin(np.pi / (2 * np.e)) ** 2
    assert excited_prob == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# Ballistic propagation test
# ---------------------------------------------------------------------------


def test_ballistic_propagation_3d():
    """propagate_states_in_borde_representation updates x, y, z from velocities."""
    vx, vy, vz = 0.01, 0.02, 0.5
    t = 0.1
    state = sim.make_atom_states(
        position_x=1.0,
        position_y=2.0,
        position_z=3.0,
        velocity_x=vx,
        velocity_y=vy,
        initial_velocity_z=vz,
        c0=1.0,
        c1=0.0,
    )

    transform_detuning_hz = 0.0
    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=0.0,
        z=0.0,
        vz=vz,
        inverse=False,
    )

    out_state = sim.propagate_states_in_borde_representation(
        state,
        time_of_propegation=t,
        detuning_hz=0.0,
        vz=vz,
    )

    for idx in range(len(state.m_values)):
        expected = state.positions[idx] + state.velocities[idx] * t
        np.testing.assert_allclose(out_state.positions[idx], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Velocity tracking test
# ---------------------------------------------------------------------------


def test_velocity_tracking_through_pulse_and_propagation():
    """vx and vy stay constant; vz changes only via recoil during pulses."""
    vx, vy, vz0 = 0.1, 0.2, 0.0
    state = sim.make_atom_states(
        velocity_x=vx,
        velocity_y=vy,
        initial_velocity_z=vz0,
        c0=1.0,
        c1=0.0,
    )

    transform_detuning_hz = 0.0
    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=0.0,
        z=0.0,
        vz=vz0,
        inverse=False,
    )

    # Apply a pulse
    out_state = sim.pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
        t_pulse=sim.T_PI,
        pulse_rabi_freq=sim.RABI_FREQ,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=sim.K_WAVEVECTOR,
        vz=vz0,
    )

    # vx and vy must stay constant for all output rows
    np.testing.assert_allclose(out_state.velocities[:, 0], vx, rtol=1e-12)
    np.testing.assert_allclose(out_state.velocities[:, 1], vy, rtol=1e-12)

    # vz must equal vz0 + m * RECOIL_VELOCITY for each row
    for i in range(len(out_state.m_values)):
        expected_vz = vz0 + out_state.m_values[i] * sim.RECOIL_VELOCITY
        assert out_state.velocities[i, 2] == pytest.approx(expected_vz, rel=1e-12)

    # Propagation must leave velocities unchanged
    prop_state = sim.propagate_states_in_borde_representation(
        out_state,
        time_of_propegation=1e-4,
        detuning_hz=sim.RECOIL_FREQUENCY_HZ,
        vz=vz0,
    )
    np.testing.assert_allclose(prop_state.velocities, out_state.velocities, rtol=1e-12)


# ---------------------------------------------------------------------------
# Clearout tests
# ---------------------------------------------------------------------------


def test_clearout_pure_ground_always_discards():
    """c0=1, c1=0: every call returns None."""
    state = sim.make_atom_states(c0=1.0, c1=0.0)
    transform_detuning_hz = 0.0
    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=0.0,
        z=0.0,
        vz=0.0,
        inverse=False,
    )
    for seed in range(50):
        rng = np.random.default_rng(seed)
        result = sim.do_clearout(state, rng=rng)
        assert result is None


def test_clearout_pure_excited_never_discards():
    """c0=0, c1=1: every call returns a non-None tuple.

    The returned amplitudes equal the input excited row (renormalisation
    is a no-op because there is no ground population).
    """
    state = sim.make_atom_states(c0=0.0, c1=1.0)
    transform_detuning_hz = 0.0
    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=0.0,
        z=0.0,
        vz=0.0,
        inverse=False,
    )
    for seed in range(50):
        rng = np.random.default_rng(seed)
        result = sim.do_clearout(state, rng=rng)
        assert result is not None
        # Only excited row remains
        assert len(result.m_values) == 1
        assert not result.internal_is_ground[0]
        np.testing.assert_allclose(
            result.amplitudes,
            state.amplitudes[~state.internal_is_ground],
            rtol=1e-12,
        )


def test_clearout_renormalises_to_unit_norm():
    """After survive, the state has p_g=0, p_e=1."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        c0 = rng.normal() + 1j * rng.normal()
        c1 = rng.normal() + 1j * rng.normal()
        norm = np.sqrt(np.abs(c0) ** 2 + np.abs(c1) ** 2)
        c0 /= norm
        c1 /= norm

        state = sim.make_atom_states(c0=c0, c1=c1)
        transform_detuning_hz = 0.0
        state = sim.transform_state_vector(
            state,
            detuning_hz=transform_detuning_hz,
            t=0.0,
            z=0.0,
            vz=0.0,
            inverse=False,
        )

        # Use a fixed rng that yields "survive" (u >= p_g)
        # We force survival by using a rng seeded such that the uniform draw
        # is >= p_g.  Instead of guessing the seed, we loop until we get a survive.
        for trial in range(1000):
            rng_trial = np.random.default_rng(seed * 1000 + trial)
            result = sim.do_clearout(state, rng=rng_trial)
            if result is not None:
                p_g, p_e = sim.calculate_ground_and_excited_probabilities(result)
                assert p_g == pytest.approx(0.0, abs=1e-12)
                assert p_e == pytest.approx(1.0, rel=1e-6)
                break
        else:
            pytest.fail("Never got a survive outcome in 1000 trials")


def test_clearout_discard_rate_matches_initial_population():
    """Discard fraction over many trials equals |c0|^2."""
    seed = 42
    rng = np.random.default_rng(seed)
    c0 = rng.normal() + 1j * rng.normal()
    c1 = rng.normal() + 1j * rng.normal()
    norm = np.sqrt(np.abs(c0) ** 2 + np.abs(c1) ** 2)
    c0 /= norm
    c1 /= norm
    p_g_expected = np.abs(c0) ** 2

    state = sim.make_atom_states(c0=c0, c1=c1)
    transform_detuning_hz = 0.0
    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=0.0,
        z=0.0,
        vz=0.0,
        inverse=False,
    )

    n_trials = 5000
    n_discards = 0
    for trial in range(n_trials):
        rng_trial = np.random.default_rng(seed * 10000 + trial)
        result = sim.do_clearout(state, rng=rng_trial)
        if result is None:
            n_discards += 1

    discard_fraction = n_discards / n_trials
    tolerance = 4.0 / np.sqrt(n_trials)
    assert discard_fraction == pytest.approx(p_g_expected, abs=tolerance)


def test_clearout_drops_ground_rows():
    """After a pi/2 pulse, clearout removes all ground rows on survive."""
    state = sim.make_atom_states(c0=1.0, c1=0.0)
    transform_detuning_hz = sim.RECOIL_FREQUENCY_HZ
    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=0.0,
        z=0.0,
        vz=0.0,
        inverse=False,
    )

    # pi/2 pulse creates both ground and excited rows with various m
    state = sim.pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
        t_pulse=sim.T_PI / 2,
        pulse_rabi_freq=sim.RABI_FREQ,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=sim.K_WAVEVECTOR,
        vz=0.0,
    )

    # Loop until we get a survive
    for trial in range(1000):
        rng = np.random.default_rng(12345 + trial)
        result = sim.do_clearout(state, rng=rng)
        if result is not None:
            n_excited_before = np.sum(~state.internal_is_ground)
            assert len(result.m_values) == n_excited_before
            assert not result.internal_is_ground.any()
            break
    else:
        pytest.fail("Never got a survive outcome in 1000 trials")


def test_clearout_then_pulse_consistent():
    """After clearout survive, a subsequent pi pulse keeps total population 1."""
    state = sim.make_atom_states(c0=1.0, c1=0.0)
    transform_detuning_hz = sim.RECOIL_FREQUENCY_HZ
    state = sim.transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=0.0,
        z=0.0,
        vz=0.0,
        inverse=False,
    )

    # pi/2 pulse
    state = sim.pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
        t_pulse=sim.T_PI / 2,
        pulse_rabi_freq=sim.RABI_FREQ,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=sim.K_WAVEVECTOR,
        vz=0.0,
    )

    # Loop until survive
    for trial in range(1000):
        rng = np.random.default_rng(99999 + trial)
        result = sim.do_clearout(state, rng=rng)
        if result is not None:
            # Apply a pi pulse
            result = sim.pulse_interaction_in_borde_representation(
                result,
                pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
                t_pulse=sim.T_PI,
                pulse_rabi_freq=sim.RABI_FREQ,
                pulse_phase=0.0,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
                vz=0.0,
            )
            total = _total_population(result)
            assert total == pytest.approx(1.0, rel=1e-6)
            assert np.isfinite(total)
            break
    else:
        pytest.fail("Never got a survive outcome in 1000 trials")


def test_clearout_empty_state_returns_none():
    """Feeding zero-length arrays returns None."""
    rng = np.random.default_rng(42)
    result = sim.do_clearout(
        sim.AtomState(
            m_values=np.array([], dtype=int),
            positions=np.empty((0, 3), dtype=float),
            velocities=np.empty((0, 3), dtype=float),
            amplitudes=np.array([], dtype=complex),
            internal_is_ground=np.array([], dtype=bool),
        ),
        rng=rng,
    )
    assert result is None


def test_clearout_mc_matches_deterministic_dropground():
    """MC population fractions (including discard channel) match deterministic
    drop-ground-no-renormalise baseline."""
    phi = 0.3
    detuning_hz = sim.RECOIL_FREQUENCY_HZ
    time_between = 200e-6
    n_trials = 5000

    def sequence_fn(rng):
        state = sim.make_atom_states(c0=1.0, c1=0.0)
        transform_detuning_hz = detuning_hz
        current_time = 0.0

        state = sim.transform_state_vector(
            state,
            detuning_hz=transform_detuning_hz,
            t=current_time,
            z=0.0,
            vz=0.0,
            inverse=False,
        )

        # pi/2 pulse (phase=0)
        state = sim.pulse_interaction_in_borde_representation(
            state,
            pulse_detuning=detuning_hz,
            t_pulse=sim.T_PI / 2,
            pulse_rabi_freq=sim.RABI_FREQ,
            pulse_phase=0.0,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
            vz=0.0,
        )
        current_time += sim.T_PI / 2

        # Propagate
        state = sim.propagate_states_in_borde_representation(
            state,
            time_of_propegation=time_between,
            detuning_hz=detuning_hz,
            vz=0.0,
            k_wavevector=sim.K_WAVEVECTOR,
        )
        current_time += time_between

        # pi pulse (phase=phi)
        state = sim.pulse_interaction_in_borde_representation(
            state,
            pulse_detuning=detuning_hz,
            t_pulse=sim.T_PI,
            pulse_rabi_freq=sim.RABI_FREQ,
            pulse_phase=phi,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
            vz=0.0,
        )
        current_time += sim.T_PI

        # Clearout
        result = sim.do_clearout(state, rng=rng)
        if result is None:
            return None

        # Propagate
        state = sim.propagate_states_in_borde_representation(
            result,
            time_of_propegation=time_between,
            detuning_hz=detuning_hz,
            vz=0.0,
            k_wavevector=sim.K_WAVEVECTOR,
        )
        current_time += time_between

        # Final pi/2 pulse (phase=4*phi)
        state = sim.pulse_interaction_in_borde_representation(
            state,
            pulse_detuning=detuning_hz,
            t_pulse=sim.T_PI / 2,
            pulse_rabi_freq=sim.RABI_FREQ,
            pulse_phase=4.0 * phi,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
            vz=0.0,
        )
        current_time += sim.T_PI / 2

        # Transform back to lab frame
        state = sim.transform_state_vector(
            state,
            detuning_hz=transform_detuning_hz,
            t=current_time,
            z=0.0,
            vz=0.0,
            inverse=True,
        )

        return state

    # Deterministic baseline: same sequence but replace clearout with
    # "drop ground rows, do NOT renormalise"
    def deterministic_sequence():
        state = sim.make_atom_states(c0=1.0, c1=0.0)
        transform_detuning_hz = detuning_hz
        current_time = 0.0

        state = sim.transform_state_vector(
            state,
            detuning_hz=transform_detuning_hz,
            t=current_time,
            z=0.0,
            vz=0.0,
            inverse=False,
        )

        # pi/2 pulse (phase=0)
        state = sim.pulse_interaction_in_borde_representation(
            state,
            pulse_detuning=detuning_hz,
            t_pulse=sim.T_PI / 2,
            pulse_rabi_freq=sim.RABI_FREQ,
            pulse_phase=0.0,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
            vz=0.0,
        )
        current_time += sim.T_PI / 2

        # Propagate
        state = sim.propagate_states_in_borde_representation(
            state,
            time_of_propegation=time_between,
            detuning_hz=detuning_hz,
            vz=0.0,
            k_wavevector=sim.K_WAVEVECTOR,
        )
        current_time += time_between

        # pi pulse (phase=phi)
        state = sim.pulse_interaction_in_borde_representation(
            state,
            pulse_detuning=detuning_hz,
            t_pulse=sim.T_PI,
            pulse_rabi_freq=sim.RABI_FREQ,
            pulse_phase=phi,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
            vz=0.0,
        )
        current_time += sim.T_PI

        # Drop ground rows, do NOT renormalise
        keep = ~state.internal_is_ground
        state = sim.AtomState(
            m_values=state.m_values[keep],
            positions=state.positions[keep],
            velocities=state.velocities[keep],
            amplitudes=state.amplitudes[keep],
            internal_is_ground=state.internal_is_ground[keep],
        )

        # Propagate
        state = sim.propagate_states_in_borde_representation(
            state,
            time_of_propegation=time_between,
            detuning_hz=detuning_hz,
            vz=0.0,
            k_wavevector=sim.K_WAVEVECTOR,
        )
        current_time += time_between

        # Final pi/2 pulse (phase=4*phi)
        state = sim.pulse_interaction_in_borde_representation(
            state,
            pulse_detuning=detuning_hz,
            t_pulse=sim.T_PI / 2,
            pulse_rabi_freq=sim.RABI_FREQ,
            pulse_phase=4.0 * phi,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
            vz=0.0,
        )
        current_time += sim.T_PI / 2

        # Transform back to lab frame
        state = sim.transform_state_vector(
            state,
            detuning_hz=transform_detuning_hz,
            t=current_time,
            z=0.0,
            vz=0.0,
            inverse=True,
        )

        p_g, p_e = sim.calculate_ground_and_excited_probabilities(state)
        p_d = 1.0 - (p_g + p_e)
        return p_g, p_e, p_d

    p_g_det, p_e_det, p_d_det = deterministic_sequence()

    seed = 777
    rng = np.random.default_rng(seed)
    p_g_mc, p_e_mc, p_d_mc = run_clearout_trials(sequence_fn, n_trials, rng=rng)

    tolerance = 4.0 / np.sqrt(n_trials)
    assert p_g_mc == pytest.approx(p_g_det, abs=tolerance)
    assert p_e_mc == pytest.approx(p_e_det, abs=tolerance)
    assert p_d_mc == pytest.approx(p_d_det, abs=tolerance)


def test_clearout_frame_independence():
    """do_clearout gives same outcome on Bordé-frame and lab-frame amplitudes."""
    seed = 123
    rng = np.random.default_rng(seed)
    c0 = rng.normal() + 1j * rng.normal()
    c1 = rng.normal() + 1j * rng.normal()
    norm = np.sqrt(np.abs(c0) ** 2 + np.abs(c1) ** 2)
    c0 /= norm
    c1 /= norm

    # Use non-zero t, z, vz so the Bordé transform is non-trivial.
    t_ref = 1e-6
    z_ref = 1e-6
    vz_ref = 1.0

    # --- Bordé frame test ---
    borde_state = sim.make_atom_states(c0=c0, c1=c1)
    transform_detuning_hz = 0.0
    borde_state = sim.transform_state_vector(
        borde_state,
        detuning_hz=transform_detuning_hz,
        t=t_ref,
        z=z_ref,
        vz=vz_ref,
        inverse=False,
    )

    rng_borde = np.random.default_rng(42)
    result_borde = sim.do_clearout(borde_state, rng=rng_borde)

    # --- Lab frame test ---
    lab_state = sim.make_atom_states(c0=c0, c1=c1)
    # Transform to Bordé, then back to lab (non-trivial parameters)
    lab_state = sim.transform_state_vector(
        lab_state,
        detuning_hz=transform_detuning_hz,
        t=t_ref,
        z=z_ref,
        vz=vz_ref,
        inverse=False,
    )
    lab_state = sim.transform_state_vector(
        lab_state,
        detuning_hz=transform_detuning_hz,
        t=t_ref,
        z=z_ref,
        vz=vz_ref,
        inverse=True,
    )

    rng_lab = np.random.default_rng(42)
    result_lab = sim.do_clearout(lab_state, rng=rng_lab)

    # Both should discard or both should survive
    if result_borde is None:
        assert result_lab is None
    else:
        assert result_lab is not None
        p_g_b, p_e_b = sim.calculate_ground_and_excited_probabilities(result_borde)
        p_g_l, p_e_l = sim.calculate_ground_and_excited_probabilities(result_lab)
        assert p_g_b == pytest.approx(p_g_l, abs=1e-12)
        assert p_e_b == pytest.approx(p_e_l, abs=1e-12)


# ---------------------------------------------------------------------------
# change_laser_frequency_in_borde_representation
# ---------------------------------------------------------------------------


def _make_synthetic_borde_state(rng, t_ref=0.0, f_ref=0.0):
    m_values = np.array([-1, 0, 0, 1, 2], dtype=int)
    internal_is_ground = np.array([True, True, False, False, True], dtype=bool)
    sq = rng.normal(size=5) + 1j * rng.normal(size=5)
    sq = sq / np.linalg.norm(sq)
    positions = rng.normal(size=(5, 3)) * 1e-3
    velocities = rng.normal(size=(5, 3)) * 0.1
    return sim.AtomState(
        m_values=m_values,
        positions=positions,
        velocities=velocities,
        amplitudes=sq,
        internal_is_ground=internal_is_ground,
        t_ref=t_ref,
        f_ref=f_ref,
    )


@pytest.mark.parametrize("seed", range(20))
def test_change_laser_frequency_accumulates_closed_segment(seed):
    """A frequency change closes the open segment into the integral, untouched amps.

    ``change_laser_frequency_in_borde_representation`` must NOT modify the
    amplitudes (the laser phase is continuous, so the instantaneous Bordé frame is
    unchanged at the step). It only advances the carried integral
    ``accumulated_detuning_cycles += f_ref * (time - t_ref)`` and rebases
    ``(t_ref, f_ref)``.
    """
    rng = np.random.default_rng(seed)

    old_detuning = rng.uniform(-50e3, 50e3)
    new_detuning = rng.uniform(-50e3, 50e3)
    t_ref = rng.uniform(0.0, 1e-5)
    phi0 = rng.uniform(-1.0, 1.0)
    time = t_ref + rng.uniform(1e-7, 1e-5)

    state = _make_synthetic_borde_state(rng, t_ref=t_ref, f_ref=old_detuning)
    state = replace(state, accumulated_detuning_cycles=phi0)

    out = sim.change_laser_frequency_in_borde_representation(
        state,
        new_detuning_hz=new_detuning,
        time=time,
    )

    # Amplitudes are untouched.
    np.testing.assert_array_equal(out.amplitudes, state.amplitudes)

    # The integral advanced by the just-closed segment, and the frame is rebased.
    assert out.accumulated_detuning_cycles == pytest.approx(
        phi0 + old_detuning * (time - t_ref), rel=1e-12
    )
    assert out.t_ref == time
    assert out.f_ref == new_detuning

    # Pass-through invariants.
    np.testing.assert_array_equal(out.m_values, state.m_values)
    np.testing.assert_array_equal(out.internal_is_ground, state.internal_is_ground)
    np.testing.assert_array_equal(out.positions, state.positions)
    np.testing.assert_array_equal(out.velocities, state.velocities)


def test_change_laser_frequency_identity_when_time_equals_t_ref():
    """A zero-length open segment (time == t_ref) leaves the integral unchanged."""
    rng = np.random.default_rng(0)
    state = _make_synthetic_borde_state(rng, t_ref=2.0e-4, f_ref=1.0e3)
    state = replace(state, accumulated_detuning_cycles=0.5)

    out_state = sim.change_laser_frequency_in_borde_representation(
        state,
        new_detuning_hz=-7.5e3,
        time=2.0e-4,
    )
    np.testing.assert_array_equal(out_state.amplitudes, state.amplitudes)
    assert out_state.accumulated_detuning_cycles == 0.5
    # The frame is still rebased to the new detuning.
    assert out_state.t_ref == 2.0e-4
    assert out_state.f_ref == -7.5e3


def test_change_laser_frequency_same_frequency_is_noop_on_lab_output():
    """Rebasing to the SAME detuning has no effect on the lab-frame output.

    This is the headline guarantee. Inserting an extra same-frequency rebase at an
    interior time and then transforming back to the lab frame gives the identical
    result as transforming straight to the lab frame -- the rebase only repartitions
    the integral between ``accumulated_detuning_cycles`` and the open segment, and
    never touches the amplitudes. (Exact up to the eps*omega_0*t float noise of the
    optical-frequency phase; we bound t to the microsecond scale to keep that noise
    below the tolerance, as in the other transform tests.)
    """
    rng = np.random.default_rng(1)
    f_ref = 4.2e3
    state = _make_synthetic_borde_state(rng, t_ref=0.0, f_ref=f_ref)
    t_end = 8e-6
    vz = 0.27

    # Straight to lab.
    lab_direct = sim.transform_state_vector(
        state,
        detuning_hz=state.f_ref,
        t=t_end,
        t_ref=state.t_ref,
        accumulated_detuning_cycles=state.accumulated_detuning_cycles,
        z=0.0,
        vz=vz,
        inverse=True,
    )
    # Same-frequency rebase at an interior time, then to lab.
    rebased = sim.change_laser_frequency_in_borde_representation(
        state, new_detuning_hz=f_ref, time=3e-6
    )
    # The rebase leaves the amplitudes untouched...
    np.testing.assert_array_equal(rebased.amplitudes, state.amplitudes)
    # ...but does move the integral into accumulated_detuning_cycles.
    assert rebased.accumulated_detuning_cycles != 0.0
    lab_rebased = sim.transform_state_vector(
        rebased,
        detuning_hz=rebased.f_ref,
        t=t_end,
        t_ref=rebased.t_ref,
        accumulated_detuning_cycles=rebased.accumulated_detuning_cycles,
        z=0.0,
        vz=vz,
        inverse=True,
    )
    # The lab-frame output is unchanged.
    np.testing.assert_allclose(
        lab_rebased.amplitudes, lab_direct.amplitudes, rtol=1e-5, atol=1e-6
    )


def test_change_laser_frequency_independent_of_position_and_velocity():
    """Guard: the rebase must not develop any k*z or v_z dependence."""
    rng = np.random.default_rng(2)
    state = _make_synthetic_borde_state(rng, t_ref=0.0, f_ref=1.0e3)
    zero_state = sim.AtomState(
        m_values=state.m_values,
        positions=np.zeros_like(state.positions),
        velocities=np.zeros_like(state.velocities),
        amplitudes=state.amplitudes,
        internal_is_ground=state.internal_is_ground,
        t_ref=state.t_ref,
        f_ref=state.f_ref,
        accumulated_detuning_cycles=state.accumulated_detuning_cycles,
    )

    state_with_pos = sim.change_laser_frequency_in_borde_representation(
        state,
        new_detuning_hz=-3.0e3,
        time=2e-4,
    )
    state_without_pos = sim.change_laser_frequency_in_borde_representation(
        zero_state,
        new_detuning_hz=-3.0e3,
        time=2e-4,
    )
    np.testing.assert_array_equal(
        state_with_pos.amplitudes, state_without_pos.amplitudes
    )


def test_frequency_change_sequence_matches_no_frame_change_reference(seed=0):
    """A real frequency-change sequence matches the description-A composition.

    For a single co-propagating arm {|g,0>, |e,1>}, the physically correct
    propagator for ``pulse(d1) - freefall(tau) - pulse(d2)`` is the plain product
    of the per-block propagators with NO inter-block frame change (the chirp's
    detuning already lives in each block's Omega_3; see
    docs/arp_frame_change_finding.md). We build that reference directly from the
    single-source primitives and check the full row/Bordé sequence reproduces it.

    This is the test the OLD ``exp(+/-i pi Df t)`` frame change fails: it inserts a
    diagonal phase before pulse 2, which a non-diagonal pulse turns into a
    population error.
    """
    import lmt_sim.lmt_sequence as seq

    rabi = sim.RABI_FREQ
    d1 = sim.RECOIL_FREQUENCY_HZ + 3.0e3
    d2 = sim.RECOIL_FREQUENCY_HZ - 5.0e3
    tp1 = 0.3 / (2 * rabi)
    tau = 200e-6
    tp2 = 0.7 / (2 * rabi)

    # Reference: P2 @ F(d1) @ P1 in basis [excited(m=1), ground(m=0)], no frame change.
    p1 = sim._single_pulse_propagator_2x2(d1, tp1, rabi, k_sign=+1, m_ground=0)
    p2 = sim._single_pulse_propagator_2x2(d2, tp2, rabi, k_sign=+1, m_ground=0)
    _, _, _, omega_3 = sim._borde_frame_constants(d1, k_sign=+1, vz=0.0, m_ground=0)
    free = np.diag([np.exp(-1j * omega_3 * tau / 2), np.exp(+1j * omega_3 * tau / 2)])
    c_e_ref, c_g_ref = p2 @ free @ p1 @ np.array([0.0, 1.0], dtype=complex)

    # Row/Bordé path through the real sequence machinery.
    state = sim.AtomState(
        m_values=np.array([0], dtype=int),
        positions=np.zeros((1, 3)),
        velocities=np.zeros((1, 3)),
        amplitudes=np.array([1.0 + 0.0j]),
        internal_is_ground=np.array([True]),
    )
    sequence = [
        seq.Pulse(k=+1, detuning_hz=d1, phi=0.0, label="p1", rabi_frequency=rabi, duration=tp1),
        seq.Freefall(duration=tau),
        seq.Pulse(k=+1, detuning_hz=d2, phi=0.0, label="p2", rabi_frequency=rabi, duration=tp2),
    ]
    final, _, _ = seq.run_pulse_sequence_in_borde_representation(
        state, sequence, initial_velocity_z=0.0, discard_threshold=0.0
    )
    p_g, p_e = sim.calculate_ground_and_excited_probabilities(final)

    assert p_e == pytest.approx(abs(c_e_ref) ** 2, abs=1e-12)
    assert p_g == pytest.approx(abs(c_g_ref) ** 2, abs=1e-12)
