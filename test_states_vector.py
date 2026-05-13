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

    m_values, positions, internal_amplitude, internal_is_ground = sim.make_atom_states(
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

    m_values, squiggly_amplitudes, internal_is_ground, positions = (
        sim.pulse_interaction_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
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

    m_values, squiggly_amplitudes, internal_is_ground, positions = (
        sim.propagate_states_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
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

    m_values, squiggly_amplitudes, internal_is_ground, positions = (
        sim.pulse_interaction_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
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

    m_values, squiggly_amplitudes, internal_is_ground, positions = (
        sim.propagate_states_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
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

    m_values, squiggly_amplitudes, internal_is_ground, positions = (
        sim.pulse_interaction_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
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
