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
