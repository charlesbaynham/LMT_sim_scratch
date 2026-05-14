import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lmt_simulation as sim


def _total_population(m_values, amplitudes, is_ground):
    ground_prob, excited_prob = sim.calculate_ground_and_excited_probabilities(
        m_values,
        amplitudes,
        is_ground,
    )
    return ground_prob + excited_prob


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

    m_values, positions, velocities, internal_amplitude, internal_is_ground = sim.make_atom_states(
        position_z=0.0,
        initial_velocity_z=0.0,
        c0=c0,
        c1=c1,
    )

    assert _total_population(
        m_values, internal_amplitude, internal_is_ground
    ) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + detuning_hz)
    current_time = 0.0

    squiggly_amplitudes = sim.transform_state_vector(
        m_values,
        internal_amplitude,
        internal_is_ground,
        omega_laser=omega_laser,
        t=current_time,
        z=0.0,
        vz=0.0,
        inverse=False,
    )

    assert _total_population(
        m_values, squiggly_amplitudes, internal_is_ground
    ) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
        sim.pulse_interaction_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
            velocities,
            pulse_detuning=detuning_hz,
            t_pulse=pulse_1,
            pulse_rabi_freq=sim.RABI_FREQ,
            pulse_phase=0.0,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
            vz=0.0,
        )
    )
    current_time += pulse_1

    assert _total_population(
        m_values, squiggly_amplitudes, internal_is_ground
    ) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
        sim.propagate_states_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
            velocities,
            time_of_propegation=time_between,
            omega_laser=omega_laser,
            vz=0.0,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
        )
    )
    current_time += time_between

    assert _total_population(
        m_values, squiggly_amplitudes, internal_is_ground
    ) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
        sim.pulse_interaction_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
            velocities,
            pulse_detuning=detuning_hz,
            t_pulse=pulse_2,
            pulse_rabi_freq=sim.RABI_FREQ,
            pulse_phase=phi,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
            vz=0.0,
        )
    )
    current_time += pulse_2

    assert _total_population(
        m_values, squiggly_amplitudes, internal_is_ground
    ) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
        sim.propagate_states_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
            velocities,
            time_of_propegation=time_between,
            omega_laser=omega_laser,
            vz=0.0,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
        )
    )
    current_time += time_between

    assert _total_population(
        m_values, squiggly_amplitudes, internal_is_ground
    ) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
        sim.pulse_interaction_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
            velocities,
            pulse_detuning=detuning_hz,
            t_pulse=pulse_3,
            pulse_rabi_freq=sim.RABI_FREQ,
            pulse_phase=4.0 * phi,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
            vz=0.0,
        )
    )
    current_time += pulse_3

    assert _total_population(
        m_values, squiggly_amplitudes, internal_is_ground
    ) == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )

    final_amplitude = sim.transform_state_vector(
        m_values,
        squiggly_amplitudes,
        internal_is_ground,
        omega_laser=omega_laser,
        t=current_time,
        z=0.0,
        vz=0.0,
        inverse=True,
    )

    assert _total_population(
        m_values, final_amplitude, internal_is_ground
    ) == pytest.approx(
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
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        sim.make_atom_states(c0=1.0, c1=0.0)
    )
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    squiggly_amplitudes = sim.transform_state_vector(
        m_values, internal_amplitude, internal_is_ground,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=False,
    )

    common_kwargs = dict(
        pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
        t_pulse=sim.T_PI,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=sim.K_WAVEVECTOR,
        vz=0.0,
    )

    # Scalar path
    out_scalar = sim.pulse_interaction_in_borde_representation(
        m_values, squiggly_amplitudes, internal_is_ground,
        positions, velocities,
        pulse_rabi_freq=sim.RABI_FREQ,
        **common_kwargs,
    )

    # Array path: same value broadcast to (N,)
    N = len(m_values)
    rabi_array = np.full(N, sim.RABI_FREQ)
    out_array = sim.pulse_interaction_in_borde_representation(
        m_values, squiggly_amplitudes, internal_is_ground,
        positions, velocities,
        pulse_rabi_freq=rabi_array,
        **common_kwargs,
    )

    np.testing.assert_allclose(out_scalar[1], out_array[1], rtol=1e-12)
    np.testing.assert_array_equal(out_scalar[0], out_array[0])


# ---------------------------------------------------------------------------
# Gaussian-pulse end-to-end tests
# ---------------------------------------------------------------------------

def test_gaussian_pulse_on_axis_pi():
    """do_gaussian_pulse with atom at beam centre gives full population transfer."""
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        sim.make_atom_states(c0=1.0, c1=0.0)
    )
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    squiggly_amplitudes = sim.transform_state_vector(
        m_values, internal_amplitude, internal_is_ground,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=False,
    )

    m_out, amp_out, isg_out, pos_out, vel_out = sim.do_gaussian_pulse(
        m_values, squiggly_amplitudes, internal_is_ground,
        positions, velocities,
        pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
        t_pulse=sim.T_PI,
        on_axis_rabi_freq=sim.RABI_FREQ,
        beam_waist=1.0,  # large beam: essentially flat-top at r=0
        vz=0.0,
    )

    amp_lab = sim.transform_state_vector(
        m_out, amp_out, isg_out,
        omega_laser=omega_laser, t=sim.T_PI, z=0.0, vz=0.0, inverse=True,
    )
    _, excited_prob = sim.calculate_ground_and_excited_probabilities(m_out, amp_lab, isg_out)
    assert excited_prob == pytest.approx(1.0, abs=1e-4)


def test_gaussian_pulse_at_waist():
    """do_gaussian_pulse with atom at r=w gives excitation sin^2(pi / (2e)) ~ 0.310."""
    w = 1e-3
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        sim.make_atom_states(position_x=w, c0=1.0, c1=0.0)
    )
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    squiggly_amplitudes = sim.transform_state_vector(
        m_values, internal_amplitude, internal_is_ground,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=False,
    )

    m_out, amp_out, isg_out, pos_out, vel_out = sim.do_gaussian_pulse(
        m_values, squiggly_amplitudes, internal_is_ground,
        positions, velocities,
        pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
        t_pulse=sim.T_PI,
        on_axis_rabi_freq=sim.RABI_FREQ,
        beam_waist=w,
        vz=0.0,
    )

    amp_lab = sim.transform_state_vector(
        m_out, amp_out, isg_out,
        omega_laser=omega_laser, t=sim.T_PI, z=0.0, vz=0.0, inverse=True,
    )
    _, excited_prob = sim.calculate_ground_and_excited_probabilities(m_out, amp_lab, isg_out)
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
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        sim.make_atom_states(
            position_x=1.0, position_y=2.0, position_z=3.0,
            velocity_x=vx, velocity_y=vy, initial_velocity_z=vz,
            c0=1.0, c1=0.0,
        )
    )

    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    squiggly_amplitudes = sim.transform_state_vector(
        m_values, internal_amplitude, internal_is_ground,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=vz, inverse=False,
    )

    _, _, _, pos_out, vel_out = sim.propagate_states_in_borde_representation(
        m_values, squiggly_amplitudes, internal_is_ground,
        positions, velocities,
        time_of_propegation=t,
        omega_laser=omega_laser,
        vz=vz,
    )

    for idx in range(len(m_values)):
        expected = positions[idx] + velocities[idx] * t
        np.testing.assert_allclose(pos_out[idx], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Velocity tracking test
# ---------------------------------------------------------------------------

def test_velocity_tracking_through_pulse_and_propagation():
    """vx and vy stay constant; vz changes only via recoil during pulses."""
    vx, vy, vz0 = 0.1, 0.2, 0.0
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        sim.make_atom_states(
            velocity_x=vx, velocity_y=vy, initial_velocity_z=vz0,
            c0=1.0, c1=0.0,
        )
    )

    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    squiggly_amplitudes = sim.transform_state_vector(
        m_values, internal_amplitude, internal_is_ground,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=vz0, inverse=False,
    )

    # Apply a pulse
    m_out, amp_out, isg_out, pos_out, vel_out = (
        sim.pulse_interaction_in_borde_representation(
            m_values, squiggly_amplitudes, internal_is_ground,
            positions, velocities,
            pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
            t_pulse=sim.T_PI,
            pulse_rabi_freq=sim.RABI_FREQ,
            pulse_phase=0.0,
            k_sign=+1,
            k_wavevector=sim.K_WAVEVECTOR,
            vz=vz0,
        )
    )

    # vx and vy must stay constant for all output rows
    np.testing.assert_allclose(vel_out[:, 0], vx, rtol=1e-12)
    np.testing.assert_allclose(vel_out[:, 1], vy, rtol=1e-12)

    # vz must equal vz0 + m * RECOIL_VELOCITY for each row
    for i in range(len(m_out)):
        expected_vz = vz0 + m_out[i] * sim.RECOIL_VELOCITY
        assert vel_out[i, 2] == pytest.approx(expected_vz, rel=1e-12)

    # Propagation must leave velocities unchanged
    _, _, _, pos_prop, vel_prop = sim.propagate_states_in_borde_representation(
        m_out, amp_out, isg_out, pos_out, vel_out,
        time_of_propegation=1e-4,
        omega_laser=omega_laser,
        vz=vz0,
    )
    np.testing.assert_allclose(vel_prop, vel_out, rtol=1e-12)


# ---------------------------------------------------------------------------
# Clearout tests
# ---------------------------------------------------------------------------

def test_clearout_pure_ground_always_discards():
    """c0=1, c1=0: every call returns None."""
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        sim.make_atom_states(c0=1.0, c1=0.0)
    )
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    squiggly_amplitudes = sim.transform_state_vector(
        m_values, internal_amplitude, internal_is_ground,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=False,
    )
    for seed in range(50):
        rng = np.random.default_rng(seed)
        result = sim.do_clearout(
            m_values, squiggly_amplitudes, internal_is_ground,
            positions, velocities, rng=rng,
        )
        assert result is None


def test_clearout_pure_excited_never_discards():
    """c0=0, c1=1: every call returns a non-None tuple.

    The returned amplitudes equal the input excited row (renormalisation
    is a no-op because there is no ground population).
    """
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        sim.make_atom_states(c0=0.0, c1=1.0)
    )
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    squiggly_amplitudes = sim.transform_state_vector(
        m_values, internal_amplitude, internal_is_ground,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=False,
    )
    for seed in range(50):
        rng = np.random.default_rng(seed)
        result = sim.do_clearout(
            m_values, squiggly_amplitudes, internal_is_ground,
            positions, velocities, rng=rng,
        )
        assert result is not None
        m_out, amp_out, isg_out, pos_out, vel_out = result
        # Only excited row remains
        assert len(m_out) == 1
        assert not isg_out[0]
        np.testing.assert_allclose(amp_out, squiggly_amplitudes[~internal_is_ground], rtol=1e-12)


def test_clearout_renormalises_to_unit_norm():
    """After survive, the state has p_g=0, p_e=1."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        c0 = rng.normal() + 1j * rng.normal()
        c1 = rng.normal() + 1j * rng.normal()
        norm = np.sqrt(np.abs(c0) ** 2 + np.abs(c1) ** 2)
        c0 /= norm
        c1 /= norm

        m_values, positions, velocities, internal_amplitude, internal_is_ground = (
            sim.make_atom_states(c0=c0, c1=c1)
        )
        omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
        squiggly_amplitudes = sim.transform_state_vector(
            m_values, internal_amplitude, internal_is_ground,
            omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=False,
        )

        # Use a fixed rng that yields "survive" (u >= p_g)
        # We force survival by using a rng seeded such that the uniform draw
        # is >= p_g.  Instead of guessing the seed, we loop until we get a survive.
        for trial in range(1000):
            rng_trial = np.random.default_rng(seed * 1000 + trial)
            result = sim.do_clearout(
                m_values, squiggly_amplitudes, internal_is_ground,
                positions, velocities, rng=rng_trial,
            )
            if result is not None:
                m_out, amp_out, isg_out, _, _ = result
                p_g, p_e = sim.calculate_ground_and_excited_probabilities(
                    m_out, amp_out, isg_out,
                )
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

    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        sim.make_atom_states(c0=c0, c1=c1)
    )
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    squiggly_amplitudes = sim.transform_state_vector(
        m_values, internal_amplitude, internal_is_ground,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=False,
    )

    n_trials = 5000
    n_discards = 0
    for trial in range(n_trials):
        rng_trial = np.random.default_rng(seed * 10000 + trial)
        result = sim.do_clearout(
            m_values, squiggly_amplitudes, internal_is_ground,
            positions, velocities, rng=rng_trial,
        )
        if result is None:
            n_discards += 1

    discard_fraction = n_discards / n_trials
    tolerance = 4.0 / np.sqrt(n_trials)
    assert discard_fraction == pytest.approx(p_g_expected, abs=tolerance)


def test_clearout_drops_ground_rows():
    """After a pi/2 pulse, clearout removes all ground rows on survive."""
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        sim.make_atom_states(c0=1.0, c1=0.0)
    )
    omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + sim.RECOIL_FREQUENCY_HZ)
    squiggly_amplitudes = sim.transform_state_vector(
        m_values, internal_amplitude, internal_is_ground,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=False,
    )

    # pi/2 pulse creates both ground and excited rows with various m
    m_vals, amps, isg, pos, vel = sim.pulse_interaction_in_borde_representation(
        m_values, squiggly_amplitudes, internal_is_ground,
        positions, velocities,
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
        result = sim.do_clearout(m_vals, amps, isg, pos, vel, rng=rng)
        if result is not None:
            m_out, amp_out, isg_out, _, _ = result
            n_excited_before = np.sum(~isg)
            assert len(m_out) == n_excited_before
            assert not isg_out.any()
            break
    else:
        pytest.fail("Never got a survive outcome in 1000 trials")


def test_clearout_then_pulse_consistent():
    """After clearout survive, a subsequent pi pulse keeps total population 1."""
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        sim.make_atom_states(c0=1.0, c1=0.0)
    )
    omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + sim.RECOIL_FREQUENCY_HZ)
    squiggly_amplitudes = sim.transform_state_vector(
        m_values, internal_amplitude, internal_is_ground,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=False,
    )

    # pi/2 pulse
    m_vals, amps, isg, pos, vel = sim.pulse_interaction_in_borde_representation(
        m_values, squiggly_amplitudes, internal_is_ground,
        positions, velocities,
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
        result = sim.do_clearout(m_vals, amps, isg, pos, vel, rng=rng)
        if result is not None:
            m_out, amp_out, isg_out, pos_out, vel_out = result
            # Apply a pi pulse
            m2, a2, i2, p2, v2 = sim.pulse_interaction_in_borde_representation(
                m_out, amp_out, isg_out,
                pos_out, vel_out,
                pulse_detuning=sim.RECOIL_FREQUENCY_HZ,
                t_pulse=sim.T_PI,
                pulse_rabi_freq=sim.RABI_FREQ,
                pulse_phase=0.0,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
                vz=0.0,
            )
            total = _total_population(m2, a2, i2)
            assert total == pytest.approx(1.0, rel=1e-6)
            assert np.isfinite(total)
            break
    else:
        pytest.fail("Never got a survive outcome in 1000 trials")


def test_clearout_empty_state_returns_none():
    """Feeding zero-length arrays returns None."""
    rng = np.random.default_rng(42)
    result = sim.do_clearout(
        np.array([], dtype=int),
        np.array([], dtype=complex),
        np.array([], dtype=bool),
        np.empty((0, 3), dtype=float),
        np.empty((0, 3), dtype=float),
        rng=rng,
    )
    assert result is None


def test_clearout_mc_matches_deterministic_dropground():
    """Mach-Zehnder with clearout: MC results match deterministic baseline."""
    phi = 0.3
    detuning_hz = sim.RECOIL_FREQUENCY_HZ
    time_between = 200e-6
    n_trials = 5000

    def sequence_fn(rng):
        m_values, positions, velocities, internal_amplitude, internal_is_ground = (
            sim.make_atom_states(c0=1.0, c1=0.0)
        )
        omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + detuning_hz)
        current_time = 0.0

        squiggly_amplitudes = sim.transform_state_vector(
            m_values, internal_amplitude, internal_is_ground,
            omega_laser=omega_laser, t=current_time, z=0.0, vz=0.0, inverse=False,
        )

        # pi/2 pulse (phase=0)
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            sim.pulse_interaction_in_borde_representation(
                m_values, squiggly_amplitudes, internal_is_ground,
                positions, velocities,
                pulse_detuning=detuning_hz,
                t_pulse=sim.T_PI / 2,
                pulse_rabi_freq=sim.RABI_FREQ,
                pulse_phase=0.0,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
                vz=0.0,
            )
        )
        current_time += sim.T_PI / 2

        # Propagate
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            sim.propagate_states_in_borde_representation(
                m_values, squiggly_amplitudes, internal_is_ground,
                positions, velocities,
                time_of_propegation=time_between,
                omega_laser=omega_laser,
                vz=0.0,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
            )
        )
        current_time += time_between

        # pi pulse (phase=phi)
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            sim.pulse_interaction_in_borde_representation(
                m_values, squiggly_amplitudes, internal_is_ground,
                positions, velocities,
                pulse_detuning=detuning_hz,
                t_pulse=sim.T_PI,
                pulse_rabi_freq=sim.RABI_FREQ,
                pulse_phase=phi,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
                vz=0.0,
            )
        )
        current_time += sim.T_PI

        # Clearout
        result = sim.do_clearout(
            m_values, squiggly_amplitudes, internal_is_ground,
            positions, velocities, rng=rng,
        )
        if result is None:
            return None
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = result
        current_time += 0.0  # clearout is instantaneous

        # Propagate
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            sim.propagate_states_in_borde_representation(
                m_values, squiggly_amplitudes, internal_is_ground,
                positions, velocities,
                time_of_propegation=time_between,
                omega_laser=omega_laser,
                vz=0.0,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
            )
        )
        current_time += time_between

        # Final pi/2 pulse (phase=4*phi)
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            sim.pulse_interaction_in_borde_representation(
                m_values, squiggly_amplitudes, internal_is_ground,
                positions, velocities,
                pulse_detuning=detuning_hz,
                t_pulse=sim.T_PI / 2,
                pulse_rabi_freq=sim.RABI_FREQ,
                pulse_phase=4.0 * phi,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
                vz=0.0,
            )
        )
        current_time += sim.T_PI / 2

        # Transform back to lab frame
        internal_amplitude_final = sim.transform_state_vector(
            m_values, squiggly_amplitudes, internal_is_ground,
            omega_laser=omega_laser, t=current_time, z=0.0, vz=0.0, inverse=True,
        )

        return (
            m_values,
            internal_amplitude_final,
            internal_is_ground,
            positions,
            velocities,
        )

    # Deterministic baseline: same sequence but replace clearout with
    # "drop ground rows, do NOT renormalise"
    def deterministic_sequence():
        m_values, positions, velocities, internal_amplitude, internal_is_ground = (
            sim.make_atom_states(c0=1.0, c1=0.0)
        )
        omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + detuning_hz)
        current_time = 0.0

        squiggly_amplitudes = sim.transform_state_vector(
            m_values, internal_amplitude, internal_is_ground,
            omega_laser=omega_laser, t=current_time, z=0.0, vz=0.0, inverse=False,
        )

        # pi/2 pulse (phase=0)
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            sim.pulse_interaction_in_borde_representation(
                m_values, squiggly_amplitudes, internal_is_ground,
                positions, velocities,
                pulse_detuning=detuning_hz,
                t_pulse=sim.T_PI / 2,
                pulse_rabi_freq=sim.RABI_FREQ,
                pulse_phase=0.0,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
                vz=0.0,
            )
        )
        current_time += sim.T_PI / 2

        # Propagate
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            sim.propagate_states_in_borde_representation(
                m_values, squiggly_amplitudes, internal_is_ground,
                positions, velocities,
                time_of_propegation=time_between,
                omega_laser=omega_laser,
                vz=0.0,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
            )
        )
        current_time += time_between

        # pi pulse (phase=phi)
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            sim.pulse_interaction_in_borde_representation(
                m_values, squiggly_amplitudes, internal_is_ground,
                positions, velocities,
                pulse_detuning=detuning_hz,
                t_pulse=sim.T_PI,
                pulse_rabi_freq=sim.RABI_FREQ,
                pulse_phase=phi,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
                vz=0.0,
            )
        )
        current_time += sim.T_PI

        # Drop ground rows, do NOT renormalise
        keep = ~internal_is_ground
        m_values = m_values[keep]
        squiggly_amplitudes = squiggly_amplitudes[keep]
        internal_is_ground = internal_is_ground[keep]
        positions = positions[keep]
        velocities = velocities[keep]

        # Propagate
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            sim.propagate_states_in_borde_representation(
                m_values, squiggly_amplitudes, internal_is_ground,
                positions, velocities,
                time_of_propegation=time_between,
                omega_laser=omega_laser,
                vz=0.0,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
            )
        )
        current_time += time_between

        # Final pi/2 pulse (phase=4*phi)
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            sim.pulse_interaction_in_borde_representation(
                m_values, squiggly_amplitudes, internal_is_ground,
                positions, velocities,
                pulse_detuning=detuning_hz,
                t_pulse=sim.T_PI / 2,
                pulse_rabi_freq=sim.RABI_FREQ,
                pulse_phase=4.0 * phi,
                k_sign=+1,
                k_wavevector=sim.K_WAVEVECTOR,
                vz=0.0,
            )
        )
        current_time += sim.T_PI / 2

        # Transform back to lab frame
        internal_amplitude_final = sim.transform_state_vector(
            m_values, squiggly_amplitudes, internal_is_ground,
            omega_laser=omega_laser, t=current_time, z=0.0, vz=0.0, inverse=True,
        )

        p_g, p_e = sim.calculate_ground_and_excited_probabilities(
            m_values, internal_amplitude_final, internal_is_ground,
        )
        p_d = 1.0 - (p_g + p_e)
        return p_g, p_e, p_d

    p_g_det, p_e_det, p_d_det = deterministic_sequence()

    seed = 777
    rng = np.random.default_rng(seed)
    p_g_mc, p_e_mc, p_d_mc = sim.run_clearout_trials(sequence_fn, n_trials, rng=rng)

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

    # --- Bordé frame test ---
    m_b, pos_b, vel_b, amp_b, isg_b = sim.make_atom_states(c0=c0, c1=c1)
    omega_laser = 2 * np.pi * sim.TRANSITION_FREQUENCY
    squiggly_b = sim.transform_state_vector(
        m_b, amp_b, isg_b,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=False,
    )

    rng_borde = np.random.default_rng(42)
    result_borde = sim.do_clearout(m_b, squiggly_b, isg_b, pos_b, vel_b, rng=rng_borde)

    # --- Lab frame test ---
    m_l, pos_l, vel_l, amp_l, isg_l = sim.make_atom_states(c0=c0, c1=c1)
    # Transform to Bordé, then back to lab
    squiggly_tmp = sim.transform_state_vector(
        m_l, amp_l, isg_l,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=False,
    )
    amp_lab = sim.transform_state_vector(
        m_l, squiggly_tmp, isg_l,
        omega_laser=omega_laser, t=0.0, z=0.0, vz=0.0, inverse=True,
    )

    rng_lab = np.random.default_rng(42)
    result_lab = sim.do_clearout(m_l, amp_lab, isg_l, pos_l, vel_l, rng=rng_lab)

    # Both should discard or both should survive
    if result_borde is None:
        assert result_lab is None
    else:
        assert result_lab is not None
        m_b_out, amp_b_out, isg_b_out, _, _ = result_borde
        m_l_out, amp_l_out, isg_l_out, _, _ = result_lab
        p_g_b, p_e_b = sim.calculate_ground_and_excited_probabilities(
            m_b_out, amp_b_out, isg_b_out,
        )
        p_g_l, p_e_l = sim.calculate_ground_and_excited_probabilities(
            m_l_out, amp_l_out, isg_l_out,
        )
        assert p_g_b == pytest.approx(p_g_l, abs=1e-12)
        assert p_e_b == pytest.approx(p_e_l, abs=1e-12)
