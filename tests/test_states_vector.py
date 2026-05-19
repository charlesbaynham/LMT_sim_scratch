import sys
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

    omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + detuning_hz)
    current_time = 0.0

    state = sim.transform_state_vector(
        state,
        omega_laser=omega_laser,
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
        omega_laser=omega_laser,
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
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    state = sim.transform_state_vector(
        state,
        omega_laser=omega_laser,
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
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    state = sim.transform_state_vector(
        state,
        omega_laser=omega_laser,
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
        omega_laser=omega_laser,
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
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    state = sim.transform_state_vector(
        state,
        omega_laser=omega_laser,
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
        omega_laser=omega_laser,
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

    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    state = sim.transform_state_vector(
        state,
        omega_laser=omega_laser,
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

    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    state = sim.transform_state_vector(
        state,
        omega_laser=omega_laser,
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
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    state = sim.transform_state_vector(
        state,
        omega_laser=omega_laser,
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
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    state = sim.transform_state_vector(
        state,
        omega_laser=omega_laser,
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
        omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
        state = sim.transform_state_vector(
            state,
            omega_laser=omega_laser,
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
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    state = sim.transform_state_vector(
        state,
        omega_laser=omega_laser,
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
    omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + sim.RECOIL_FREQUENCY_HZ)
    state = sim.transform_state_vector(
        state,
        omega_laser=omega_laser,
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
    omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + sim.RECOIL_FREQUENCY_HZ)
    state = sim.transform_state_vector(
        state,
        omega_laser=omega_laser,
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
        omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + detuning_hz)
        current_time = 0.0

        state = sim.transform_state_vector(
            state,
            omega_laser=omega_laser,
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
            omega_laser=omega_laser,
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
        omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + detuning_hz)
        current_time = 0.0

        state = sim.transform_state_vector(
            state,
            omega_laser=omega_laser,
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
            omega_laser=omega_laser,
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
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    borde_state = sim.transform_state_vector(
        borde_state,
        omega_laser=omega_laser,
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
        omega_laser=omega_laser,
        t=t_ref,
        z=z_ref,
        vz=vz_ref,
        inverse=False,
    )
    lab_state = sim.transform_state_vector(
        lab_state,
        omega_laser=omega_laser,
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


def _make_synthetic_borde_state(rng):
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
    )


@pytest.mark.parametrize("seed", range(20))
def test_change_laser_frequency_matches_lab_roundtrip(seed):
    """Direct Bordé(old)→Bordé(new) equals routing through the lab frame.

    The reference path multiplies by exp(i * omega_laser * t / 2) and its
    conjugate-with-different-omega_laser, where omega_laser is dominated by
    the optical transition frequency (~10^15 rad/s).  np.exp of such a large
    argument has ~eps * |arg| precision in the resulting phase, so the
    reference path is noisy at the eps * omega_0 * t / 2 level.  We bound
    t to the microsecond scale to keep that noise well below the
    detuning-induced phase we are checking.
    """
    rng = np.random.default_rng(seed)
    state = _make_synthetic_borde_state(rng)

    old_detuning = rng.uniform(-50e3, 50e3)
    new_detuning = rng.uniform(-50e3, 50e3)
    t_frame_change = rng.uniform(1e-7, 1e-5)

    omega_L_old = 2 * np.pi * (sim.TRANSITION_FREQUENCY + old_detuning)
    omega_L_new = 2 * np.pi * (sim.TRANSITION_FREQUENCY + new_detuning)

    # Reference: Bordé(old) -> lab -> Bordé(new), evaluated at the same
    # (t, z, vz) on both legs so the m / k z / v_z t cancellations are exact.
    state_lab = sim.transform_state_vector(
        state,
        omega_laser=omega_L_old,
        t=t_frame_change,
        z=0.0,
        vz=0.0,
        inverse=True,
    )
    state_reference = sim.transform_state_vector(
        state_lab,
        omega_laser=omega_L_new,
        t=t_frame_change,
        z=0.0,
        vz=0.0,
        inverse=False,
    )

    # Direct.
    state_direct = sim.change_laser_frequency_in_borde_representation(
        state,
        old_detuning_hz=old_detuning,
        new_detuning_hz=new_detuning,
        time=t_frame_change,
    )

    np.testing.assert_allclose(
        state_direct.amplitudes,
        state_reference.amplitudes,
        rtol=1e-5,
        atol=1e-6,
    )

    # Pass-through invariants.
    np.testing.assert_array_equal(state_direct.m_values, state.m_values)
    np.testing.assert_array_equal(
        state_direct.internal_is_ground, state.internal_is_ground
    )
    np.testing.assert_array_equal(state_direct.positions, state.positions)
    np.testing.assert_array_equal(state_direct.velocities, state.velocities)

    # Norm preservation (each row's phase factor is unit-modulus).
    np.testing.assert_allclose(
        np.abs(state_direct.amplitudes), np.abs(state.amplitudes), rtol=1e-12
    )


def test_change_laser_frequency_identity_when_time_zero():
    rng = np.random.default_rng(0)
    state = _make_synthetic_borde_state(rng)

    out_state = sim.change_laser_frequency_in_borde_representation(
        state,
        old_detuning_hz=1.0e3,
        new_detuning_hz=-7.5e3,
        time=0.0,
    )
    np.testing.assert_array_equal(out_state.amplitudes, state.amplitudes)


def test_change_laser_frequency_identity_when_old_equals_new():
    rng = np.random.default_rng(1)
    state = _make_synthetic_borde_state(rng)

    out_state = sim.change_laser_frequency_in_borde_representation(
        state,
        old_detuning_hz=4.2e3,
        new_detuning_hz=4.2e3,
        time=1.5e-4,
    )
    np.testing.assert_array_equal(out_state.amplitudes, state.amplitudes)


def test_change_laser_frequency_independent_of_position_and_velocity():
    """Guard: the transform must not develop any k*z or v_z dependence."""
    rng = np.random.default_rng(2)
    state = _make_synthetic_borde_state(rng)
    zero_state = sim.AtomState(
        m_values=state.m_values,
        positions=np.zeros_like(state.positions),
        velocities=np.zeros_like(state.velocities),
        amplitudes=state.amplitudes,
        internal_is_ground=state.internal_is_ground,
    )

    state_with_pos = sim.change_laser_frequency_in_borde_representation(
        state,
        old_detuning_hz=1.0e3,
        new_detuning_hz=-3.0e3,
        time=2e-4,
    )
    state_without_pos = sim.change_laser_frequency_in_borde_representation(
        zero_state,
        old_detuning_hz=1.0e3,
        new_detuning_hz=-3.0e3,
        time=2e-4,
    )
    np.testing.assert_array_equal(
        state_with_pos.amplitudes, state_without_pos.amplitudes
    )
