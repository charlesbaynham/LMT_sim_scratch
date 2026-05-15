import numpy as np
import pytest

from lmt_simulation import (
    K_WAVEVECTOR,
    Pulse,
    RABI_FREQ,
    RECOIL_FREQUENCY_HZ,
    T_PI,
    TRANSITION_FREQUENCY,
    build_mach_zehnder_pulse_sequence,
    calc_mz_excitation,
    calculate_excited_fraction_for_pulse_sequence,
    calculate_ground_and_excited_probabilities,
    make_atom_states,
    propagate_states_in_borde_representation,
    pulse_interaction_in_borde_representation,
    transform_state_vector,
)


def legacy_calc_mz_excitation(
    phi,
    detuning_hz=RECOIL_FREQUENCY_HZ,
    initial_velocity_z=0.0,
    time_between_pulses=200e-6,
):
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        make_atom_states(initial_velocity_z=initial_velocity_z)
    )

    omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)
    current_time = 0.0

    squiggly_amplitudes = transform_state_vector(
        m_values,
        internal_amplitude,
        internal_is_ground,
        omega_laser=omega_laser,
        t=current_time,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )

    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
        pulse_interaction_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
            velocities,
            pulse_detuning=detuning_hz,
            t_pulse=T_PI / 2,
            pulse_rabi_freq=RABI_FREQ,
            pulse_phase=0.0,
            k_sign=+1,
            k_wavevector=K_WAVEVECTOR,
            vz=initial_velocity_z,
        )
    )
    current_time += T_PI / 2

    if time_between_pulses > 0.0:
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            propagate_states_in_borde_representation(
                m_values,
                squiggly_amplitudes,
                internal_is_ground,
                positions,
                velocities,
                time_of_propegation=time_between_pulses,
                omega_laser=omega_laser,
                vz=initial_velocity_z,
                k_sign=+1,
                k_wavevector=K_WAVEVECTOR,
            )
        )
        current_time += time_between_pulses

    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
        pulse_interaction_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
            velocities,
            pulse_detuning=detuning_hz,
            t_pulse=T_PI,
            pulse_rabi_freq=RABI_FREQ,
            pulse_phase=phi,
            k_sign=+1,
            k_wavevector=K_WAVEVECTOR,
            vz=initial_velocity_z,
        )
    )
    current_time += T_PI

    if time_between_pulses > 0.0:
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            propagate_states_in_borde_representation(
                m_values,
                squiggly_amplitudes,
                internal_is_ground,
                positions,
                velocities,
                time_of_propegation=time_between_pulses,
                omega_laser=omega_laser,
                vz=initial_velocity_z,
                k_sign=+1,
                k_wavevector=K_WAVEVECTOR,
            )
        )
        current_time += time_between_pulses

    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
        pulse_interaction_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
            velocities,
            pulse_detuning=detuning_hz,
            t_pulse=T_PI / 2,
            pulse_rabi_freq=RABI_FREQ,
            pulse_phase=4 * phi,
            k_sign=+1,
            k_wavevector=K_WAVEVECTOR,
            vz=initial_velocity_z,
        )
    )
    current_time += T_PI / 2

    internal_amplitude_final = transform_state_vector(
        m_values,
        squiggly_amplitudes,
        internal_is_ground,
        omega_laser=omega_laser,
        t=current_time,
        z=0.0,
        vz=initial_velocity_z,
        inverse=True,
    )

    ground_prob, excited_prob = calculate_ground_and_excited_probabilities(
        m_values,
        internal_amplitude_final,
        internal_is_ground,
    )
    return excited_prob / (ground_prob + excited_prob)


def test_build_mach_zehnder_pulse_sequence_returns_pulse_objects():
    pulse_sequence = build_mach_zehnder_pulse_sequence(
        detuning_hz=RECOIL_FREQUENCY_HZ,
        time_between_pulses=200e-6,
    )

    assert all(isinstance(pulse, Pulse) for pulse in pulse_sequence)
    assert [pulse.label for pulse in pulse_sequence] == [
        "beam_splitter_1",
        "mirror",
        "beam_splitter_2",
    ]
    assert [pulse.phi for pulse in pulse_sequence] == [0.0, 0.0, 0.0]
    assert [pulse.k for pulse in pulse_sequence] == [+1, +1, +1]
    assert np.isclose(pulse_sequence[0].time, 0.0)
    assert np.isclose(pulse_sequence[0].duration, T_PI / 2)
    assert np.isclose(pulse_sequence[1].time, T_PI / 2 + 200e-6)
    assert np.isclose(pulse_sequence[1].duration, T_PI)
    assert np.isclose(pulse_sequence[2].time, T_PI / 2 + 200e-6 + T_PI + 200e-6)
    assert np.isclose(pulse_sequence[2].duration, T_PI / 2)


@pytest.mark.parametrize(
    ("phi", "detuning_hz", "initial_velocity_z", "time_between_pulses"),
    [
        (0.0, RECOIL_FREQUENCY_HZ, 0.0, 200e-6),
        (0.37 * np.pi, RECOIL_FREQUENCY_HZ, 0.0, 0.0),
        (0.91 * np.pi, 0.0, 1.2e-3, 200e-6),
        (1.5 * np.pi, 1.3 * RECOIL_FREQUENCY_HZ, -8.0e-4, 350e-6),
    ],
)
def test_pulse_sequence_interface_matches_legacy_results(
    phi,
    detuning_hz,
    initial_velocity_z,
    time_between_pulses,
):
    pulse_sequence = build_mach_zehnder_pulse_sequence(
        phi=phi,
        detuning_hz=detuning_hz,
        time_between_pulses=time_between_pulses,
    )

    new_result = calculate_excited_fraction_for_pulse_sequence(
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
    )
    legacy_result = legacy_calc_mz_excitation(
        phi=phi,
        detuning_hz=detuning_hz,
        initial_velocity_z=initial_velocity_z,
        time_between_pulses=time_between_pulses,
    )

    assert np.isclose(new_result, legacy_result)
    assert np.isclose(
        calc_mz_excitation(
            phi=phi,
            detuning_hz=detuning_hz,
            initial_velocity_z=initial_velocity_z,
            time_between_pulses=time_between_pulses,
        ),
        legacy_result,
    )


def test_build_mach_zehnder_pulse_sequence_stores_phase_on_pulses():
    phi = 0.37 * np.pi
    pulse_sequence = build_mach_zehnder_pulse_sequence(phi=phi)

    assert np.isclose(pulse_sequence[0].phi, 0.0)
    assert np.isclose(pulse_sequence[1].phi, phi)
    assert np.isclose(pulse_sequence[2].phi, 4 * phi)
