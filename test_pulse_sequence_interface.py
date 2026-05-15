import numpy as np
import pytest

from lmt_simulation import (
    Clearout,
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
    do_clearout,
    make_atom_states,
    propagate_states_in_borde_representation,
    pulse_interaction_in_borde_representation,
    run_pulse_sequence_in_borde_representation,
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


def legacy_run_mz_sequence_in_borde_representation(
    phi,
    detuning_hz=RECOIL_FREQUENCY_HZ,
    initial_velocity_z=0.0,
    time_between_pulses=200e-6,
):
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        make_atom_states(initial_velocity_z=initial_velocity_z)
    )
    pulse_sequence = build_mach_zehnder_pulse_sequence(
        phi=phi,
        detuning_hz=detuning_hz,
        time_between_pulses=time_between_pulses,
    )

    omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)
    squiggly_amplitudes = transform_state_vector(
        m_values,
        internal_amplitude,
        internal_is_ground,
        omega_laser=omega_laser,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )
    current_time = 0.0

    for pulse in pulse_sequence:
        free_evolution_time = pulse.time - current_time
        if free_evolution_time > 0.0:
            m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
                propagate_states_in_borde_representation(
                    m_values,
                    squiggly_amplitudes,
                    internal_is_ground,
                    positions,
                    velocities,
                    time_of_propegation=free_evolution_time,
                    omega_laser=omega_laser,
                    vz=initial_velocity_z,
                    k_sign=pulse.k,
                    k_wavevector=K_WAVEVECTOR,
                )
            )
            current_time += free_evolution_time

        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            pulse_interaction_in_borde_representation(
                m_values,
                squiggly_amplitudes,
                internal_is_ground,
                positions,
                velocities,
                pulse_detuning=pulse.detuning_hz,
                t_pulse=pulse.duration,
                pulse_rabi_freq=pulse.rabi_frequency,
                pulse_phase=pulse.phi,
                k_sign=pulse.k,
                k_wavevector=K_WAVEVECTOR,
                vz=initial_velocity_z,
            )
        )
        current_time += pulse.duration

    return (
        m_values,
        squiggly_amplitudes,
        internal_is_ground,
        positions,
        velocities,
        omega_laser,
        current_time,
    )


def legacy_run_mz_sequence_with_clearout_in_borde_representation(
    phi,
    rng,
    detuning_hz=RECOIL_FREQUENCY_HZ,
    initial_velocity_z=0.0,
    time_between_pulses=200e-6,
):
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        make_atom_states(initial_velocity_z=initial_velocity_z)
    )
    omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)
    squiggly_amplitudes = transform_state_vector(
        m_values,
        internal_amplitude,
        internal_is_ground,
        omega_laser=omega_laser,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )
    current_time = 0.0

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

    result = do_clearout(
        m_values,
        squiggly_amplitudes,
        internal_is_ground,
        positions,
        velocities,
        rng=rng,
    )
    if result is None:
        return None
    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = result

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

    return (
        m_values,
        squiggly_amplitudes,
        internal_is_ground,
        positions,
        velocities,
        omega_laser,
        current_time,
    )


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


@pytest.mark.parametrize(
    ("phi", "detuning_hz", "initial_velocity_z", "time_between_pulses"),
    [
        (0.0, RECOIL_FREQUENCY_HZ, 0.0, 200e-6),
        (0.37 * np.pi, RECOIL_FREQUENCY_HZ, 0.0, 0.0),
        (1.5 * np.pi, 1.3 * RECOIL_FREQUENCY_HZ, -8.0e-4, 350e-6),
    ],
)
def test_run_pulse_sequence_in_borde_representation_preserves_representation(
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
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        make_atom_states(initial_velocity_z=initial_velocity_z)
    )
    omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)
    squiggly_amplitudes = transform_state_vector(
        m_values,
        internal_amplitude,
        internal_is_ground,
        omega_laser=omega_laser,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )

    actual = run_pulse_sequence_in_borde_representation(
        m_values,
        positions,
        velocities,
        squiggly_amplitudes,
        internal_is_ground,
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
    )
    expected = legacy_run_mz_sequence_in_borde_representation(
        phi=phi,
        detuning_hz=detuning_hz,
        initial_velocity_z=initial_velocity_z,
        time_between_pulses=time_between_pulses,
    )

    for actual_value, expected_value in zip(actual, expected):
        if isinstance(actual_value, np.ndarray):
            assert np.allclose(actual_value, expected_value)
        else:
            assert np.isclose(actual_value, expected_value)


@pytest.mark.parametrize("seed", [0, 1, 4, 7])
def test_run_pulse_sequence_in_borde_representation_handles_clearout(seed):
    phi = 0.37 * np.pi
    detuning_hz = RECOIL_FREQUENCY_HZ
    time_between_pulses = 200e-6
    initial_velocity_z = 0.0
    clearout_duration = 37e-6
    pulse_sequence = [
        Pulse(
            time=0.0,
            k=+1,
            detuning_hz=detuning_hz,
            phi=0.0,
            label="beam_splitter_1",
            rabi_frequency=RABI_FREQ,
            pulse_area=np.pi / 2,
        ),
        Pulse(
            time=T_PI / 2 + time_between_pulses,
            k=+1,
            detuning_hz=detuning_hz,
            phi=phi,
            label="mirror",
            rabi_frequency=RABI_FREQ,
            pulse_area=np.pi,
        ),
        Clearout(
            time=T_PI / 2 + time_between_pulses + T_PI,
            duration=clearout_duration,
            label="mid_sequence_clearout",
        ),
        Pulse(
            time=T_PI / 2 + time_between_pulses + T_PI + time_between_pulses,
            k=+1,
            detuning_hz=detuning_hz,
            phi=4 * phi,
            label="beam_splitter_2",
            rabi_frequency=RABI_FREQ,
            pulse_area=np.pi / 2,
        ),
    ]
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        make_atom_states(initial_velocity_z=initial_velocity_z)
    )
    omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)
    squiggly_amplitudes = transform_state_vector(
        m_values,
        internal_amplitude,
        internal_is_ground,
        omega_laser=omega_laser,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )

    actual_rng = np.random.default_rng(seed)
    actual = run_pulse_sequence_in_borde_representation(
        m_values,
        positions,
        velocities,
        squiggly_amplitudes,
        internal_is_ground,
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
        rng=actual_rng,
    )
    expected_rng = np.random.default_rng(seed)
    expected = legacy_run_mz_sequence_with_clearout_in_borde_representation(
        phi=phi,
        rng=expected_rng,
        detuning_hz=detuning_hz,
        initial_velocity_z=initial_velocity_z,
        time_between_pulses=time_between_pulses,
    )

    if expected is None:
        assert actual is None
        return

    assert actual is not None
    for actual_value, expected_value in zip(actual, expected):
        if isinstance(actual_value, np.ndarray):
            assert np.allclose(actual_value, expected_value)
        else:
            assert np.isclose(actual_value, expected_value)


def test_clearout_duration_is_metadata_only_in_pulse_sequence_runner():
    detuning_hz = RECOIL_FREQUENCY_HZ
    phi = 0.37 * np.pi
    time_between_pulses = 200e-6
    base_sequence = [
        Pulse(
            time=0.0,
            k=+1,
            detuning_hz=detuning_hz,
            phi=0.0,
            label="beam_splitter_1",
            rabi_frequency=RABI_FREQ,
            pulse_area=np.pi / 2,
        ),
        Pulse(
            time=T_PI / 2 + time_between_pulses,
            k=+1,
            detuning_hz=detuning_hz,
            phi=phi,
            label="mirror",
            rabi_frequency=RABI_FREQ,
            pulse_area=np.pi,
        ),
        Pulse(
            time=T_PI / 2 + time_between_pulses + T_PI + time_between_pulses,
            k=+1,
            detuning_hz=detuning_hz,
            phi=4 * phi,
            label="beam_splitter_2",
            rabi_frequency=RABI_FREQ,
            pulse_area=np.pi / 2,
        ),
    ]
    sequences = [
        [
            base_sequence[0],
            base_sequence[1],
            Clearout(
                time=T_PI / 2 + time_between_pulses + T_PI,
                duration=0.0,
                label="mid_sequence_clearout",
            ),
            base_sequence[2],
        ],
        [
            base_sequence[0],
            base_sequence[1],
            Clearout(
                time=T_PI / 2 + time_between_pulses + T_PI,
                duration=123e-6,
                label="mid_sequence_clearout",
            ),
            base_sequence[2],
        ],
    ]
    state = make_atom_states()
    omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)
    squiggly = transform_state_vector(
        state[0],
        state[3],
        state[4],
        omega_laser=omega_laser,
        t=0.0,
        z=0.0,
        vz=0.0,
        inverse=False,
    )

    results = []
    for sequence in sequences:
        rng = np.random.default_rng(7)
        result = run_pulse_sequence_in_borde_representation(
            state[0].copy(),
            state[1].copy(),
            state[2].copy(),
            squiggly.copy(),
            state[4].copy(),
            sequence,
            initial_velocity_z=0.0,
            rng=rng,
        )
        results.append(result)

    if results[0] is None:
        assert results[1] is None
        return

    assert results[1] is not None
    for actual_value, expected_value in zip(results[0], results[1]):
        if isinstance(actual_value, np.ndarray):
            assert np.allclose(actual_value, expected_value)
        else:
            assert np.isclose(actual_value, expected_value)


def test_calculate_excited_fraction_for_pulse_sequence_rejects_clearout_events():
    pulse_sequence = [
        Pulse(
            time=0.0,
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="beam_splitter_1",
            rabi_frequency=RABI_FREQ,
            pulse_area=np.pi / 2,
        ),
        Clearout(time=T_PI / 2, duration=10e-6),
    ]

    with pytest.raises(ValueError, match="does not support Clearout"):
        calculate_excited_fraction_for_pulse_sequence(pulse_sequence)
