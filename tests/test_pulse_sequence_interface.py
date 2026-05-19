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
    AtomState,
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


def assert_states_close(actual, expected):
    np.testing.assert_array_equal(actual.m_values, expected.m_values)
    np.testing.assert_allclose(actual.positions, expected.positions)
    np.testing.assert_allclose(actual.velocities, expected.velocities)
    np.testing.assert_allclose(actual.amplitudes, expected.amplitudes)
    np.testing.assert_array_equal(
        actual.internal_is_ground, expected.internal_is_ground
    )


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

    state = make_atom_states()

    result = run_pulse_sequence_in_lab_frame(
        state,
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
    )

    if result is None:
        return result
    else:
        final_state, current_detuning_hz, current_time = result
        ground_prob, excited_prob = calculate_ground_and_excited_probabilities(
            final_state
        )
        return excited_prob / (ground_prob + excited_prob)


def legacy_calc_mz_excitation(
    phi,
    detuning_hz=RECOIL_FREQUENCY_HZ,
    initial_velocity_z=0.0,
    time_between_pulses=200e-6,
):
    state = make_atom_states(initial_velocity_z=initial_velocity_z)

    omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)
    current_time = 0.0

    state = transform_state_vector(
        state,
        omega_laser=omega_laser,
        t=current_time,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )

    state = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=T_PI / 2,
        pulse_rabi_freq=RABI_FREQ,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=initial_velocity_z,
    )
    current_time += T_PI / 2

    if time_between_pulses > 0.0:
        state = propagate_states_in_borde_representation(
            state,
            time_of_propegation=time_between_pulses,
            detuning_hz=detuning_hz,
            vz=initial_velocity_z,
            k_wavevector=K_WAVEVECTOR,
        )
        current_time += time_between_pulses

    state = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=T_PI,
        pulse_rabi_freq=RABI_FREQ,
        pulse_phase=phi,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=initial_velocity_z,
    )
    current_time += T_PI

    if time_between_pulses > 0.0:
        state = propagate_states_in_borde_representation(
            state,
            time_of_propegation=time_between_pulses,
            detuning_hz=detuning_hz,
            vz=initial_velocity_z,
            k_wavevector=K_WAVEVECTOR,
        )
        current_time += time_between_pulses

    state = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=T_PI / 2,
        pulse_rabi_freq=RABI_FREQ,
        pulse_phase=4 * phi,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=initial_velocity_z,
    )
    current_time += T_PI / 2

    state = transform_state_vector(
        state,
        omega_laser=omega_laser,
        t=current_time,
        z=0.0,
        vz=initial_velocity_z,
        inverse=True,
    )

    ground_prob, excited_prob = calculate_ground_and_excited_probabilities(state)
    return excited_prob / (ground_prob + excited_prob)


def legacy_run_mz_sequence_in_borde_representation(
    phi,
    detuning_hz=RECOIL_FREQUENCY_HZ,
    initial_velocity_z=0.0,
    time_between_pulses=200e-6,
):
    state = make_atom_states(initial_velocity_z=initial_velocity_z)
    pulse_sequence = build_mach_zehnder_pulse_sequence(
        phi=phi,
        detuning_hz=detuning_hz,
        time_between_pulses=time_between_pulses,
    )

    omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)
    state = transform_state_vector(
        state,
        omega_laser=omega_laser,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )
    current_time = 0.0

    for event in pulse_sequence:
        if isinstance(event, Freefall):
            state = propagate_states_in_borde_representation(
                state,
                time_of_propegation=event.duration,
                detuning_hz=detuning_hz,
                vz=initial_velocity_z,
                k_wavevector=K_WAVEVECTOR,
            )
            current_time += event.duration
            continue

        state = pulse_interaction_in_borde_representation(
            state,
            pulse_detuning=event.detuning_hz,
            t_pulse=event.duration,
            pulse_rabi_freq=event.rabi_frequency,
            pulse_phase=event.phi,
            k_sign=event.k,
            k_wavevector=K_WAVEVECTOR,
            vz=initial_velocity_z,
        )
        current_time += event.duration

    return state, detuning_hz, current_time


def legacy_run_mz_sequence_with_clearout_in_borde_representation(
    phi,
    rng,
    detuning_hz=RECOIL_FREQUENCY_HZ,
    initial_velocity_z=0.0,
    time_between_pulses=200e-6,
):
    state = make_atom_states(initial_velocity_z=initial_velocity_z)
    omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)
    state = transform_state_vector(
        state,
        omega_laser=omega_laser,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )
    current_time = 0.0

    state = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=T_PI / 2,
        pulse_rabi_freq=RABI_FREQ,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=initial_velocity_z,
    )
    current_time += T_PI / 2

    if time_between_pulses > 0.0:
        state = propagate_states_in_borde_representation(
            state,
            time_of_propegation=time_between_pulses,
            detuning_hz=detuning_hz,
            vz=initial_velocity_z,
            k_wavevector=K_WAVEVECTOR,
        )
        current_time += time_between_pulses

    state = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=T_PI,
        pulse_rabi_freq=RABI_FREQ,
        pulse_phase=phi,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=initial_velocity_z,
    )
    current_time += T_PI

    result = do_clearout(state, rng=rng)
    if result is None:
        return None
    state = result

    if time_between_pulses > 0.0:
        state = propagate_states_in_borde_representation(
            state,
            time_of_propegation=time_between_pulses,
            detuning_hz=detuning_hz,
            vz=initial_velocity_z,
            k_wavevector=K_WAVEVECTOR,
        )
        current_time += time_between_pulses

    state = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=T_PI / 2,
        pulse_rabi_freq=RABI_FREQ,
        pulse_phase=4 * phi,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=initial_velocity_z,
    )
    current_time += T_PI / 2

    return state, detuning_hz, current_time


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

    clouds, clearout_times = compute_spacetime_trajectory(sequence)

    assert len(clearout_times) == 1
    assert len(clouds) >= 1

    alive = [c for c in clouds if c.alive]
    dead = [c for c in clouds if not c.alive]

    assert len(alive) >= 1
    expected_len = len(sequence) + 1
    for cloud in alive:
        assert len(cloud.times) == expected_len
        assert cloud.labels[-1] == sequence[-1].label
    for cloud in dead:
        assert len(cloud.times) < expected_len


def test_compute_spacetime_trajectory_plot_true_runs_without_error():
    sequence = [
        Pulse(
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="bs1",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        ),
        Pulse(
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="mirror",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        ),
    ]

    clouds, _ = compute_spacetime_trajectory(sequence, plot=True)
    assert len(clouds[0].times) == len(sequence) + 1


def test_compute_spacetime_trajectory_mach_zehnder():
    sequence = build_mach_zehnder_pulse_sequence()
    clouds, clearout_times = compute_spacetime_trajectory(sequence)

    assert len(clearout_times) == 0
    assert all(c.alive for c in clouds)
    # BS1 splits 1→2, mirror flips both, BS2 splits each 2→4
    assert len(clouds) == 4
    # Final m values should be two of each: 0 and 1
    assert sorted(c.m[-1] for c in clouds) == [0, 0, 1, 1]


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
    state = make_atom_states(initial_velocity_z=initial_velocity_z)
    omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)
    state = transform_state_vector(
        state,
        omega_laser=omega_laser,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )

    actual = run_pulse_sequence_in_borde_representation(
        state,
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
    )
    expected = legacy_run_mz_sequence_in_borde_representation(
        phi=phi,
        detuning_hz=detuning_hz,
        initial_velocity_z=initial_velocity_z,
        time_between_pulses=time_between_pulses,
    )

    actual_state, actual_detuning_hz, actual_time = actual
    expected_state, expected_detuning_hz, expected_time = expected
    assert_states_close(actual_state, expected_state)
    assert np.isclose(actual_detuning_hz, expected_detuning_hz)
    assert np.isclose(actual_time, expected_time)


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
    state = make_atom_states(initial_velocity_z=initial_velocity_z)
    omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)
    state = transform_state_vector(
        state,
        omega_laser=omega_laser,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )

    actual_rng = np.random.default_rng(seed)
    actual = run_pulse_sequence_in_borde_representation(
        state,
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
    actual_state, actual_detuning_hz, actual_time = actual
    expected_state, expected_detuning_hz, expected_time = expected
    assert_states_close(actual_state, expected_state)
    assert np.isclose(actual_detuning_hz, expected_detuning_hz)
    assert np.isclose(actual_time, expected_time)


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
    state = transform_state_vector(
        state,
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
            AtomState(
                m_values=state.m_values.copy(),
                positions=state.positions.copy(),
                velocities=state.velocities.copy(),
                amplitudes=state.amplitudes.copy(),
                internal_is_ground=state.internal_is_ground.copy(),
            ),
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
