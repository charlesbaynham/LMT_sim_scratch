import numpy as np
import pytest

from lmt_sim.lmt_sequence import (
    compute_spacetime_trajectory,
    Clearout,
    Freefall,
    Pulse,
    calculate_excited_fraction_for_pulse_sequence,
    run_pulse_sequence_in_lab_frame,
    run_pulse_sequence_in_borde_representation,
    build_mach_zehnder_pulse_sequence,
)
from lmt_sim.lmt_simulation import (
    K_WAVEVECTOR,
    RABI_FREQ,
    RECOIL_FREQUENCY_HZ,
    T_PI,
    TRANSITION_FREQUENCY,
    calculate_ground_and_excited_probabilities,
    do_clearout,
    make_atom_states,
    propagate_states_in_borde_representation,
    pulse_interaction_in_borde_representation,
    transform_state_vector,
)
from lmt_sim.lmt_real_sequence import build_lmt_real_sequence


def calc_mz_excitation(
    phi,
    detuning_hz=RECOIL_FREQUENCY_HZ,
    initial_velocity_z=0.0,
    time_between_pulses=200e-6,
):
    """Compute excitation fraction for Mach-Zehnder sequence.

    Pulse sequence: π/2 (phase=0) - π (phase=φ) - π/2 (phase=4φ)
    Uses proper quantum state evolution with coherent sum over paths.

    Parameters
    ----------
    phi : float
        Phase parameter in radians
    detuning_hz : float
        Laser detuning from resonance in Hz (default: recoil shift)
    initial_velocity_z : float
        Initial atom velocity in m/s
    time_between_pulses : float
        Time between pulses in seconds (default: 0.0)

    Returns
    -------
    float
        Excitation fraction after full sequence
    """

    pulse_sequence = build_mach_zehnder_pulse_sequence(
        phi=phi,
        detuning_hz=detuning_hz,
        time_between_pulses=time_between_pulses,
        rabi_frequency=RABI_FREQ,
        pulse_area_multiplier=1.0,
        k=+1,
    )

    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        make_atom_states()
    )

    result = run_pulse_sequence_in_lab_frame(
        m_values,
        positions,
        velocities,
        internal_amplitude,
        internal_is_ground,
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
    )

    if result is None:
        return result
    else:
        (
            m_values,
            positions,
            velocities,
            amplitudes,
            internal_is_ground,
            current_detuning_hz,
            current_time,
        ) = result
        ground_prob, excited_prob = calculate_ground_and_excited_probabilities(
            m_values,
            amplitudes,
            internal_is_ground,
        )
        return excited_prob / (ground_prob + excited_prob)


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
                detuning_hz=detuning_hz,
                vz=initial_velocity_z,
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
                detuning_hz=detuning_hz,
                vz=initial_velocity_z,
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

    for event in pulse_sequence:
        if isinstance(event, Freefall):
            m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
                propagate_states_in_borde_representation(
                    m_values,
                    squiggly_amplitudes,
                    internal_is_ground,
                    positions,
                    velocities,
                    time_of_propegation=event.duration,
                    detuning_hz=detuning_hz,
                    vz=initial_velocity_z,
                    k_wavevector=K_WAVEVECTOR,
                )
            )
            current_time += event.duration
            continue

        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
            pulse_interaction_in_borde_representation(
                m_values,
                squiggly_amplitudes,
                internal_is_ground,
                positions,
                velocities,
                pulse_detuning=event.detuning_hz,
                t_pulse=event.duration,
                pulse_rabi_freq=event.rabi_frequency,
                pulse_phase=event.phi,
                k_sign=event.k,
                k_wavevector=K_WAVEVECTOR,
                vz=initial_velocity_z,
            )
        )
        current_time += event.duration

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
                detuning_hz=detuning_hz,
                vz=initial_velocity_z,
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
                detuning_hz=detuning_hz,
                vz=initial_velocity_z,
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

    assert [type(event) for event in pulse_sequence] == [
        Pulse,
        Freefall,
        Pulse,
        Freefall,
        Pulse,
    ]
    pulses = [event for event in pulse_sequence if isinstance(event, Pulse)]
    assert [pulse.label for pulse in pulses] == [
        "beam_splitter_1",
        "mirror",
        "beam_splitter_2",
    ]
    assert [pulse.phi for pulse in pulses] == [0.0, 0.0, 0.0]
    assert [pulse.k for pulse in pulses] == [+1, +1, +1]
    assert np.isclose(pulses[0].duration, T_PI / 2)
    assert np.isclose(pulses[1].duration, T_PI)
    assert np.isclose(pulses[2].duration, T_PI / 2)
    freefalls = [event for event in pulse_sequence if isinstance(event, Freefall)]
    assert len(freefalls) == 2
    assert np.isclose(freefalls[0].duration, 200e-6)
    assert np.isclose(freefalls[1].duration, 200e-6)


def test_compute_spacetime_trajectory_returns_expected_shapes_and_clearouts():
    sequence = [
        Pulse(
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="vel sel (UP-TOP)",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        ),
        Clearout(duration=0.0, label="velocity selection clearout"),
        Pulse(
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="G1 #1 UP-TOP (BS1 pi/2)",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        ),
        Pulse(
            k=-1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="G1 #2 DOWN-BOT (acc pi)",
            rabi_frequency=RABI_FREQ,
            duration=T_PI,
        ),
    ]

    outputs = compute_spacetime_trajectory(sequence)
    (
        times,
        z_top,
        z_bot,
        v_top,
        v_bot,
        m_top,
        m_bot,
        s_top,
        s_bot,
        labels,
        clearout_times,
    ) = outputs

    expected_len = len(sequence) + 1
    assert len(times) == expected_len
    assert len(z_top) == expected_len
    assert len(z_bot) == expected_len
    assert len(v_top) == expected_len
    assert len(v_bot) == expected_len
    assert len(m_top) == expected_len
    assert len(m_bot) == expected_len
    assert len(s_top) == expected_len
    assert len(s_bot) == expected_len
    assert len(labels) == expected_len
    assert len(clearout_times) == 1
    assert labels[-1] == sequence[-1].label


def test_compute_spacetime_trajectory_plot_true_runs_without_error():
    sequence = [
        Pulse(
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="vel sel (UP-TOP)",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        ),
        Pulse(
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="G1 #1 UP-TOP (BS1 pi/2)",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        ),
    ]

    outputs = compute_spacetime_trajectory(sequence, plot=True)
    assert len(outputs[0]) == len(sequence) + 1


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
    pulse_sequence = [event for event in pulse_sequence if isinstance(event, Pulse)]

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
    clearout_duration = 0.0
    pulse_sequence = [
        Pulse(
            k=+1,
            detuning_hz=detuning_hz,
            phi=0.0,
            label="beam_splitter_1",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        ),
        Freefall(
            duration=time_between_pulses,
            label="dark_time_1",
        ),
        Pulse(
            k=+1,
            detuning_hz=detuning_hz,
            phi=phi,
            label="mirror",
            rabi_frequency=RABI_FREQ,
            duration=T_PI,
        ),
        Clearout(
            duration=clearout_duration,
            label="mid_sequence_clearout",
        ),
        Freefall(
            duration=time_between_pulses,
            label="dark_time_2",
        ),
        Pulse(
            k=+1,
            detuning_hz=detuning_hz,
            phi=4 * phi,
            label="beam_splitter_2",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
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


def test_clearout_duration_affects_timeline_in_pulse_sequence_runner():
    detuning_hz = RECOIL_FREQUENCY_HZ
    phi = 0.37 * np.pi
    time_between_pulses = 200e-6
    base_sequence = [
        Pulse(
            k=+1,
            detuning_hz=detuning_hz,
            phi=0.0,
            label="beam_splitter_1",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        ),
        Freefall(
            duration=time_between_pulses,
            label="dark_time_1",
        ),
        Pulse(
            k=+1,
            detuning_hz=detuning_hz,
            phi=phi,
            label="mirror",
            rabi_frequency=RABI_FREQ,
            duration=T_PI,
        ),
        Freefall(
            duration=time_between_pulses,
            label="dark_time_2",
        ),
        Pulse(
            k=+1,
            detuning_hz=detuning_hz,
            phi=4 * phi,
            label="beam_splitter_2",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        ),
    ]
    sequences = [
        [
            base_sequence[0],
            base_sequence[1],
            Clearout(
                duration=0.0,
                label="mid_sequence_clearout",
            ),
            base_sequence[2],
            base_sequence[3],
            base_sequence[4],
        ],
        [
            base_sequence[0],
            base_sequence[1],
            Clearout(
                duration=123e-6,
                label="mid_sequence_clearout",
            ),
            base_sequence[2],
            base_sequence[3],
            base_sequence[4],
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
    assert not np.isclose(results[0][-1], results[1][-1])


def test_calculate_excited_fraction_for_pulse_sequence_rejects_clearout_events():
    pulse_sequence = [
        Pulse(
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="beam_splitter_1",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        ),
        Clearout(duration=10e-6),
    ]

    with pytest.raises(ValueError, match="does not support Clearout"):
        calculate_excited_fraction_for_pulse_sequence(pulse_sequence)


def test_build_lmt_real_sequence_negative_gap_validation_comes_from_steps():
    with pytest.raises(ValueError, match="duration must be non-negative"):
        build_lmt_real_sequence(delay_between_interferometry_pulses=-1e-6)
    with pytest.raises(ValueError, match="duration must be non-negative"):
        build_lmt_real_sequence(vs_to_bs1_gap=-1e-6)


def test_build_lmt_real_sequence_uses_duration_based_events():
    sequence = build_lmt_real_sequence(N=7)
    assert sequence
    assert all(not hasattr(event, "time") for event in sequence)
    assert isinstance(sequence[0], Pulse)
    assert sequence[0].label == "velocity_selection"
    assert isinstance(sequence[1], Clearout)
    assert isinstance(sequence[2], Freefall)
    assert isinstance(sequence[-1], Pulse)
    assert sequence[-1].label == "BS2"


def test_build_lmt_real_sequence_has_positive_total_duration():
    sequence = build_lmt_real_sequence(N=7)
    total_duration = sum(event.duration for event in sequence)
    assert total_duration > 0.0
    assert any(isinstance(event, Freefall) for event in sequence)
    assert any(isinstance(event, Clearout) for event in sequence)
