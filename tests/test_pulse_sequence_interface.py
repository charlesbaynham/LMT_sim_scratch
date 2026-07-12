import numpy as np
import pytest

from lmt_sim.lmt_sequence import (
    _addressed_momentum_classes,
    _transition_probability,
    compute_spacetime_trajectory,
    Clearout,
    Freefall,
    Pulse,
    calculate_excited_fraction_for_pulse_sequence,
    run_pulse_sequence_in_lab_frame,
    run_pulse_sequence_in_borde_representation,
    build_mach_zehnder_pulse_sequence,
    build_sequence_from_lab_pulse_dump,
    calibrate_probe_shift_and_velocity_from_dump,
    decode_pulse_record_flat,
    LabPulseDump,
    PULSE_RECORD_SAME_AS_LAST_SENTINEL,
    PULSE_RECORD_DISABLED_SENTINEL,
)
from lmt_sim.lmt_simulation import (
    AtomState,
    K_WAVEVECTOR,
    RABI_FREQ,
    RECOIL_FREQUENCY_HZ,
    T_PI,
    calculate_ground_and_excited_probabilities,
    discard_and_renormalise_state_vector,
    do_clearout,
    make_atom_states,
    propagate_states_in_borde_representation,
    pulse_interaction_in_borde_representation,
    transform_state_vector,
)


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


# ---------------------------------------------------------------------------
# Low-level Bordé-frame primitives shared by the hand-rolled "legacy" baselines
# below. These deliberately wrap only the low-level simulation primitives (never
# the production runner) so the baselines remain an independent reference to
# compare the high-level API against.
# ---------------------------------------------------------------------------


def _legacy_to_borde(state, detuning_hz, vz, t=0.0, inverse=False):
    return transform_state_vector(
        state, detuning_hz=detuning_hz, t=t, z=0.0, vz=vz, inverse=inverse
    )


def _legacy_pulse(
    state, detuning_hz, t_pulse, phase, vz, rabi_freq=RABI_FREQ, k_sign=+1
):
    return pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=t_pulse,
        pulse_rabi_freq=rabi_freq,
        pulse_phase=phase,
        k_sign=k_sign,
        k_wavevector=K_WAVEVECTOR,
        vz=vz,
    )


def _legacy_propagate(state, dt, detuning_hz, vz):
    return propagate_states_in_borde_representation(
        state,
        time_of_propegation=dt,
        detuning_hz=detuning_hz,
        vz=vz,
        k_wavevector=K_WAVEVECTOR,
    )


def legacy_calc_mz_excitation(
    phi,
    detuning_hz=RECOIL_FREQUENCY_HZ,
    initial_velocity_z=0.0,
    time_between_pulses=200e-6,
):
    state = make_atom_states(initial_velocity_z=initial_velocity_z)
    current_time = 0.0

    state = _legacy_to_borde(state, detuning_hz, initial_velocity_z, t=current_time)

    state = _legacy_pulse(state, detuning_hz, T_PI / 2, 0.0, initial_velocity_z)
    current_time += T_PI / 2

    if time_between_pulses > 0.0:
        state = _legacy_propagate(
            state, time_between_pulses, detuning_hz, initial_velocity_z
        )
        current_time += time_between_pulses

    state = _legacy_pulse(state, detuning_hz, T_PI, phi, initial_velocity_z)
    current_time += T_PI

    if time_between_pulses > 0.0:
        state = _legacy_propagate(
            state, time_between_pulses, detuning_hz, initial_velocity_z
        )
        current_time += time_between_pulses

    state = _legacy_pulse(state, detuning_hz, T_PI / 2, 4 * phi, initial_velocity_z)
    current_time += T_PI / 2

    state = _legacy_to_borde(
        state, detuning_hz, initial_velocity_z, t=current_time, inverse=True
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

    state = _legacy_to_borde(state, detuning_hz, initial_velocity_z)
    current_time = 0.0

    for event in pulse_sequence:
        if isinstance(event, Freefall):
            state = _legacy_propagate(
                state, event.duration, detuning_hz, initial_velocity_z
            )
            current_time += event.duration
            continue

        state = _legacy_pulse(
            state,
            event.detuning_hz,
            event.duration,
            event.phi,
            initial_velocity_z,
            rabi_freq=event.rabi_frequency,
            k_sign=event.k,
        )
        current_time += event.duration

    # Match the production runner, which discards near-zero branches after each
    # pulse. The pruned branches carry ~1e-12 of the probability, so a single
    # discard at the end reproduces the same surviving rows.
    state = discard_and_renormalise_state_vector(state, 1e-9)
    return state, detuning_hz, current_time


def legacy_run_mz_sequence_with_clearout_in_borde_representation(
    phi,
    rng,
    detuning_hz=RECOIL_FREQUENCY_HZ,
    initial_velocity_z=0.0,
    time_between_pulses=200e-6,
):
    state = make_atom_states(initial_velocity_z=initial_velocity_z)
    state = _legacy_to_borde(state, detuning_hz, initial_velocity_z)
    current_time = 0.0

    state = _legacy_pulse(state, detuning_hz, T_PI / 2, 0.0, initial_velocity_z)
    current_time += T_PI / 2

    if time_between_pulses > 0.0:
        state = _legacy_propagate(
            state, time_between_pulses, detuning_hz, initial_velocity_z
        )
        current_time += time_between_pulses

    state = _legacy_pulse(state, detuning_hz, T_PI, phi, initial_velocity_z)
    current_time += T_PI

    result = do_clearout(state, rng=rng)
    if result is None:
        return None
    state = result

    if time_between_pulses > 0.0:
        state = _legacy_propagate(
            state, time_between_pulses, detuning_hz, initial_velocity_z
        )
        current_time += time_between_pulses

    state = _legacy_pulse(state, detuning_hz, T_PI / 2, 4 * phi, initial_velocity_z)
    current_time += T_PI / 2

    # Match the production runner's per-pulse discard (see note above).
    state = discard_and_renormalise_state_vector(state, 1e-9)
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


def test_compute_spacetime_trajectory_plot_places_pulse_step_at_midpoint(monkeypatch):
    sequence = [
        Pulse(
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="midpoint-check",
            rabi_frequency=RABI_FREQ,
            duration=T_PI,
        )
    ]

    class DummyAxis:
        def __init__(self):
            self.plot_calls = []
            self.vline_calls = []
            self.broken_barh_calls = []

        def plot(self, x, y, *args, **kwargs):
            self.plot_calls.append((np.asarray(x), np.asarray(y), args, kwargs))
            return []

        def broken_barh(self, *args, **kwargs):
            self.broken_barh_calls.append((args, kwargs))
            return []

        def axvline(self, *args, **kwargs):
            self.vline_calls.append((args, kwargs))

        def set_ylabel(self, *args, **kwargs):
            pass

        def set_xlabel(self, *args, **kwargs):
            pass

        def set_title(self, *args, **kwargs):
            pass

        def legend(self, *args, **kwargs):
            pass

        def grid(self, *args, **kwargs):
            pass

        def axhline(self, *args, **kwargs):
            pass

        def axvspan(self, *args, **kwargs):
            pass

    ax_z = DummyAxis()
    ax_m = DummyAxis()

    def fake_subplots(*args, **kwargs):
        return object(), (ax_z, ax_m)

    monkeypatch.setattr("matplotlib.pyplot.subplots", fake_subplots)

    compute_spacetime_trajectory(sequence, plot=True)

    midpoint_us = T_PI * 1e6 / 2
    end_us = T_PI * 1e6

    z_segment_xs = [call[0] for call in ax_z.plot_calls if len(call[0]) == 2]
    assert any(np.allclose(xs, [0.0, midpoint_us]) for xs in z_segment_xs)
    assert any(np.allclose(xs, [midpoint_us, end_us]) for xs in z_segment_xs)

    m_segment_xs = [call[0] for call in ax_m.plot_calls if len(call[0]) == 2]
    assert any(np.allclose(xs, [0.0, midpoint_us]) for xs in m_segment_xs)
    assert any(np.allclose(xs, [midpoint_us, midpoint_us]) for xs in m_segment_xs)
    assert any(np.allclose(xs, [midpoint_us, end_us]) for xs in m_segment_xs)

    z_vline_xs = sorted(float(args[0]) for args, _ in ax_z.vline_calls)
    m_vline_xs = sorted(float(args[0]) for args, _ in ax_m.vline_calls)
    assert np.allclose(z_vline_xs, [0.0, end_us])
    assert np.allclose(m_vline_xs, [0.0, end_us])

    addressed_ranges = [args[0] for args, _ in ax_m.broken_barh_calls]
    addressed_y_ranges = [args[1] for args, _ in ax_m.broken_barh_calls]
    assert addressed_ranges == [[(0.0, end_us)]]
    assert np.allclose(addressed_y_ranges, [(-0.05, 1.1)])


class _CapturingAxis:
    """Minimal matplotlib-axis stand-in that records plotted segments."""

    def __init__(self):
        self.plot_calls = []

    def plot(self, x, y, *args, **kwargs):
        self.plot_calls.append((np.asarray(x), np.asarray(y)))
        return []

    def broken_barh(self, *args, **kwargs):
        return []

    def axvline(self, *args, **kwargs):
        pass

    def axhline(self, *args, **kwargs):
        pass

    def axvspan(self, *args, **kwargs):
        pass

    def set_ylabel(self, *args, **kwargs):
        pass

    def set_xlabel(self, *args, **kwargs):
        pass

    def set_title(self, *args, **kwargs):
        pass

    def legend(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass


def _capture_spacetime_axes(monkeypatch, sequence, *, include_gravity):
    ax_z = _CapturingAxis()
    ax_m = _CapturingAxis()
    monkeypatch.setattr(
        "matplotlib.pyplot.subplots",
        lambda *a, **k: (object(), (ax_z, ax_m)),
    )
    compute_spacetime_trajectory(sequence, plot=True, include_gravity=include_gravity)
    return ax_z, ax_m


def test_compute_spacetime_trajectory_include_gravity_leaves_data_unchanged():
    # The trajectory is always solved in the freely-falling frame; the gravity
    # flag must not touch the returned Cloud data, only the plot.
    sequence = build_mach_zehnder_pulse_sequence(time_between_pulses=5e-3)
    clouds_ff, ct_ff = compute_spacetime_trajectory(sequence, include_gravity=False)
    clouds_g, ct_g = compute_spacetime_trajectory(sequence, include_gravity=True)

    np.testing.assert_array_equal(ct_ff, ct_g)
    assert len(clouds_ff) == len(clouds_g)
    for cloud_ff, cloud_g in zip(clouds_ff, clouds_g):
        assert cloud_ff.z == cloud_g.z
        assert cloud_ff.m == cloud_g.m
        assert cloud_ff.times == cloud_g.times


def test_compute_spacetime_trajectory_include_gravity_shifts_plot_to_lab_frame(
    monkeypatch,
):
    from lmt_sim.lmt_simulation import GRAVITY_G, RECOIL_VELOCITY

    sequence = build_mach_zehnder_pulse_sequence(time_between_pulses=5e-3)

    ax_z_ff, ax_m_ff = _capture_spacetime_axes(
        monkeypatch, sequence, include_gravity=False
    )
    ax_z_g, ax_m_g = _capture_spacetime_axes(
        monkeypatch, sequence, include_gravity=True
    )

    def segments_by_x(axis):
        # Key each 2-point segment by its (start, end) time in us so the
        # gravity / no-gravity runs can be compared segment-for-segment.
        out = {}
        for xs, ys in axis.plot_calls:
            if len(xs) == 2:
                out[(round(float(xs[0]), 6), round(float(xs[1]), 6))] = ys
        return out

    # z axis: lab frame subtracts 1/2 g t^2 (metres -> mm in the plot).
    z_ff = segments_by_x(ax_z_ff)
    z_g = segments_by_x(ax_z_g)
    shared_z = set(z_ff) & set(z_g)
    assert shared_z
    for key in shared_z:
        t_s = np.asarray(key) * 1e-6
        expected = z_ff[key] - 0.5 * GRAVITY_G * t_s**2 * 1e3
        np.testing.assert_allclose(z_g[key], expected, rtol=0, atol=1e-9)

    # velocity axis: lab frame subtracts g t, expressed in recoil units.
    v_ff = segments_by_x(ax_m_ff)
    v_g = segments_by_x(ax_m_g)
    shared_v = set(v_ff) & set(v_g)
    assert shared_v
    for key in shared_v:
        t_s = np.asarray(key) * 1e-6
        expected = v_ff[key] - GRAVITY_G * t_s / RECOIL_VELOCITY
        np.testing.assert_allclose(v_g[key], expected, rtol=0, atol=1e-9)


def test_addressed_momentum_classes_follow_beam_sign():
    pulse = Pulse(
        k=-1,
        detuning_hz=3 * RECOIL_FREQUENCY_HZ,
        phi=0.0,
        label="down-resonant",
        rabi_frequency=RABI_FREQ,
        duration=T_PI,
    )

    assert np.allclose(_addressed_momentum_classes(pulse), [-1.0, -2.0])


def test_addressed_momentum_classes_use_effective_detuning():
    probe_shift_coefficient = 1.0e-5
    pulse = Pulse(
        k=+1,
        detuning_hz=RECOIL_FREQUENCY_HZ + probe_shift_coefficient * RABI_FREQ**2,
        phi=0.0,
        label="probe-shifted",
        rabi_frequency=RABI_FREQ,
        duration=T_PI,
        probe_shift_coefficient=probe_shift_coefficient,
    )

    assert np.allclose(_addressed_momentum_classes(pulse), [0.0, 1.0])


def test_compute_spacetime_trajectory_max_branches_plots_before_raising(monkeypatch):
    sequence = [
        Pulse(
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="split",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        )
    ]
    plot_calls = []

    def record_plot(
        sequence_arg, clouds_arg, clearout_times_arg, *, include_gravity=False
    ):
        plot_calls.append((sequence_arg, clouds_arg, clearout_times_arg))

    monkeypatch.setattr("lmt_sim.lmt_sequence._plot_spacetime", record_plot)

    with pytest.raises(RuntimeError, match="exceeded max_branches"):
        compute_spacetime_trajectory(sequence, max_branches=1, plot=True)

    assert len(plot_calls) == 1
    plotted_sequence, plotted_clouds, plotted_clearout_times = plot_calls[0]
    assert plotted_sequence is sequence
    assert len([cloud for cloud in plotted_clouds if cloud.alive]) == 2
    assert plotted_clearout_times == []


def test_compute_spacetime_trajectory_mach_zehnder():
    sequence = build_mach_zehnder_pulse_sequence()
    clouds, clearout_times = compute_spacetime_trajectory(sequence)

    assert len(clearout_times) == 0
    assert all(c.alive for c in clouds)
    # BS1 splits 1→2, mirror flips both, BS2 splits each 2→4
    assert len(clouds) == 4
    # Final m values should be two of each: 0 and 1
    assert sorted(c.m[-1] for c in clouds) == [0, 0, 1, 1]


def test_compute_spacetime_trajectory_alternating_k_lmt():
    """Successive pi pulses with alternating k must each be resonant on the
    accumulated momentum state, so the cloud should ratchet up in |m| and
    alternate between |g> and |e>."""

    def resonant_detuning(i):
        return (2 * i + 1) * (-1) ** i * RECOIL_FREQUENCY_HZ

    n = 5
    sequence = [
        Pulse(
            k=+1 if i % 2 == 0 else -1,
            detuning_hz=resonant_detuning(i),
            phi=0.0,
            label=f"lmt {i}",
            rabi_frequency=RABI_FREQ,
            duration=T_PI,
        )
        for i in range(n)
    ]

    clouds, _ = compute_spacetime_trajectory(sequence)

    assert len(clouds) == 1, "pi pulses should never split the cloud"
    cloud = clouds[0]
    # Each pi pulse should alternate the internal state and accumulate +1 in m
    expected_m = [0] + [i + 1 for i in range(n)]
    expected_is_ground = [True] + [(i % 2 == 1) for i in range(n)]
    assert cloud.m == expected_m
    assert cloud.is_ground == expected_is_ground


def test_compute_spacetime_trajectory_pi_over_two_split_is_pulse_area_invariant_on_resonance():
    sequence_fast = [
        Pulse(
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="fast bs1",
            rabi_frequency=RABI_FREQ,
            duration=T_PI / 2,
        )
    ]
    sequence_slow = [
        Pulse(
            k=+1,
            detuning_hz=RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="slow bs1",
            rabi_frequency=RABI_FREQ / 100,
            duration=T_PI * 50,
        )
    ]

    fast_clouds, _ = compute_spacetime_trajectory(sequence_fast)
    slow_clouds, _ = compute_spacetime_trajectory(sequence_slow)

    assert len(fast_clouds) == 2
    assert len(slow_clouds) == 2
    assert sorted((cloud.m[-1], cloud.is_ground[-1]) for cloud in fast_clouds) == [
        (0, True),
        (1, False),
    ]
    assert sorted((cloud.m[-1], cloud.is_ground[-1]) for cloud in slow_clouds) == [
        (0, True),
        (1, False),
    ]


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
        velocity=(0.0, 0.0, initial_velocity_z),
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
    state = transform_state_vector(
        state,
        detuning_hz=detuning_hz,
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
    state = transform_state_vector(
        state,
        detuning_hz=detuning_hz,
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
    state = transform_state_vector(
        state,
        detuning_hz=detuning_hz,
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


def _minimal_lab_dump(is_up):
    """A minimal valid lab pulse dump (two pulses) for builder tests."""
    return dict(
        is_up=is_up,
        start_times_s=np.array([0.0, 1e-3]),
        durations_s=np.array([95e-6, 95e-6]),
        opll_hz=np.array([80e6, 80e6]),
        switch_hz=np.zeros(2),
        delivery_hz=np.zeros(2),
        delivery_setpoint=np.array([2.0, 2.0]),
    )


@pytest.mark.parametrize(
    "is_up",
    [[1, 0], np.array([1, 0]), np.array([True, False])],
)
def test_build_sequence_accepts_boolean_like_is_up(is_up):
    """0/1 ints, integer arrays and boolean arrays all map to the same beams."""
    _, sequence = build_sequence_from_lab_pulse_dump(**_minimal_lab_dump(is_up))
    pulse_ks = [event.k for event in sequence if isinstance(event, Pulse)]
    assert pulse_ks == [1, -1]


def test_build_sequence_rejects_non_boolean_is_up():
    """``~`` on an integer array yields [-2, -1]; this must fail loudly rather
    than silently coercing to all-True (all up beams)."""
    bad_is_up = ~np.array([1, 0])  # -> [-2, -1]
    with pytest.raises(ValueError, match="boolean array"):
        build_sequence_from_lab_pulse_dump(**_minimal_lab_dump(bad_is_up))


def test_build_sequence_logical_not_flips_beams():
    """The supported beam flip (boolean array / np.logical_not) swaps up<->down."""
    is_up = np.array([True, False])
    _, base = build_sequence_from_lab_pulse_dump(**_minimal_lab_dump(is_up))
    _, flipped = build_sequence_from_lab_pulse_dump(
        **_minimal_lab_dump(np.logical_not(is_up))
    )
    base_ks = [event.k for event in base if isinstance(event, Pulse)]
    flipped_ks = [event.k for event in flipped if isinstance(event, Pulse)]
    assert base_ks == [1, -1]
    assert flipped_ks == [-1, 1]


def _second_pulse_detuning(dump):
    _, sequence = build_sequence_from_lab_pulse_dump(**dump)
    pulses = [event for event in sequence if isinstance(event, Pulse)]
    return pulses[1].detuning_hz


def test_opll_and_aom_frequencies_are_all_subtracted():
    """The OPLL offsets the Sirah from the ECDL with the lock on the NEGATIVE
    side, and the switch and delivery AOMs use the -1st order, so a per-pulse
    increase in ANY of the three must DECREASE that pulse's detuning:
    total = -opll - switch - delivery (lab-confirmed convention)."""
    # Use two up pulses so beam direction is identical and the first-pulse
    # anchor cancels: only the pulse-2 offset survives.
    base = dict(
        is_up=np.array([True, True]),
        start_times_s=np.array([0.0, 1e-3]),
        durations_s=np.array([95e-6, 95e-6]),
        opll_hz=np.array([80e6, 80e6]),
        switch_hz=np.array([200e6, 200e6]),
        delivery_hz=np.array([99e6, 99e6]),
        delivery_setpoint=np.array([2.0, 2.0]),
    )
    ref = _second_pulse_detuning(base)

    bump_opll = dict(base, opll_hz=np.array([80e6, 80e6 + 1e3]))
    bump_switch = dict(base, switch_hz=np.array([200e6, 200e6 + 1e3]))
    bump_delivery = dict(base, delivery_hz=np.array([99e6, 99e6 + 1e3]))

    # All three subtract: +1 kHz on pulse 2 -> -1 kHz detuning.
    assert _second_pulse_detuning(bump_opll) - ref == pytest.approx(-1e3)
    assert _second_pulse_detuning(bump_switch) - ref == pytest.approx(-1e3)
    assert _second_pulse_detuning(bump_delivery) - ref == pytest.approx(-1e3)


def _synthesize_ladder_dump(anchor_beam_sign, alpha, v0, n_pairs=3):
    """A lab dump from a perfectly tuned LMT launch with known (alpha, v0).

    The first pulse (beam ``anchor_beam_sign``) is a long velocity-selection pi
    pulse on m=0 -> m=anchor_beam_sign; then alternating opposite/anchor-beam
    pi pulses climb the ladder. Every laser frequency is exactly what the lab
    would set so each pulse is resonant on its intended rung, using the same
    bookkeeping as build_sequence_from_lab_pulse_dump (gravity Doppler
    evaluated at pulse centres, as the lab tunes it).

    Returns ``(dump_kwargs, intended_rungs)``.
    """
    from lmt_sim.lmt_simulation import (
        GRAVITY_G,
        TRANSITION_WAVELENGTH,
    )

    beams = [anchor_beam_sign] + [-anchor_beam_sign, anchor_beam_sign] * n_pairs
    durations = [380e-6] + [68e-6, 54.999e-6] * n_pairs
    starts = [0.0] + [1e-3 + 0.15e-3 * i for i in range(2 * n_pairs)]

    # Velocity selection puts the atom at m = anchor_beam_sign (excited); each
    # subsequent pulse climbs one more recoil in that direction. The rung of a
    # pulse addressing ground-state class m_g with beam k is 2*k*m_g + 1.
    rungs = [1.0]
    recoils_from_origin = 1
    for k in beams[1:]:
        m_g = (recoils_from_origin + 1) * anchor_beam_sign
        rungs.append(2 * k * m_g + 1)
        recoils_from_origin += 1

    rabis = [1 / (2 * d) for d in durations]
    total_laser_hz = [
        rung * RECOIL_FREQUENCY_HZ
        + alpha * rabi**2
        - (
            -v0 / TRANSITION_WAVELENGTH
            + GRAVITY_G * (t0 + dur / 2) / TRANSITION_WAVELENGTH
        )
        * k
        for k, t0, dur, rung, rabi in zip(beams, starts, durations, rungs, rabis)
    ]

    dump = dict(
        is_up=np.array([k > 0 for k in beams]),
        start_times_s=np.array(starts),
        durations_s=np.array(durations),
        # total = -opll - switch - delivery
        opll_hz=-np.array(total_laser_hz),
        switch_hz=np.zeros(len(beams)),
        delivery_hz=np.zeros(len(beams)),
        delivery_setpoint=np.full(len(beams), 2.0),
    )
    return dump, np.array(rungs)


@pytest.mark.parametrize("anchor_beam_sign", [+1, -1])
def test_calibrate_probe_shift_and_velocity_warns_and_lands_on_ladder(
    anchor_beam_sign,
):
    """The auto-calibration is a hacky self-consistent fit: it must warn, and
    on a perfectly tuned dump it must recover the true (alpha, v0) and place
    every pulse on its intended recoil-ladder rung -- whichever beam fires the
    first (anchor) pulse."""
    alpha_true = -1.282e-5
    v0_true = -1.6e-3
    dump, intended_rungs = _synthesize_ladder_dump(
        anchor_beam_sign, alpha_true, v0_true
    )

    with pytest.warns(UserWarning, match="HACKY"):
        alpha, v0 = calibrate_probe_shift_and_velocity_from_dump(**dump)

    assert alpha == pytest.approx(alpha_true, rel=1e-6)
    assert v0 == pytest.approx(v0_true, rel=1e-6)

    _, sequence = build_sequence_from_lab_pulse_dump(
        **dump,
        probe_induced_alpha_up=alpha,
        probe_induced_alpha_down=alpha,
        initial_velocity_z=v0,
    )
    pulses = [event for event in sequence if isinstance(event, Pulse)]
    for pulse, intended_rung in zip(pulses, intended_rungs):
        effective = pulse.detuning_hz - alpha * pulse.rabi_frequency**2
        assert effective / RECOIL_FREQUENCY_HZ == pytest.approx(intended_rung, abs=1e-6)


# --- pulse_record_flat decoding ---------------------------------------------


def _encode_regular_record(
    is_up,
    start_times_s,
    durations_s,
    opll_hz,
    switch_hz,
    delivery_hz,
    delivery_setpoint,
    interferometry_phase_turns=None,
):
    """Encode one shot into a flat float64 record, mirroring the producer.

    Passing ``interferometry_phase_turns`` appends the 8th (phase) row, giving
    the current-format record; omitting it produces a legacy 7-row record.
    """
    rows = [
        np.asarray(is_up, dtype=float),
        np.asarray(start_times_s, dtype=float),
        np.asarray(durations_s, dtype=float),
        np.asarray(opll_hz, dtype=float),
        np.asarray(switch_hz, dtype=float),
        np.asarray(delivery_hz, dtype=float),
        np.asarray(delivery_setpoint, dtype=float),
    ]
    if interferometry_phase_turns is not None:
        rows.append(np.asarray(interferometry_phase_turns, dtype=float))
    num_pulses = len(rows[0])
    return np.concatenate([[float(num_pulses)]] + rows)


def test_decode_pulse_record_flat_round_trips_si_values():
    """A regular record decodes back to exactly the SI values it encoded, with
    no machine-unit conversion applied to the time columns."""
    is_up = [1.0, 0.0, 1.0]
    start_times_s = [0.0, 1.5e-3, 4.2e-3]
    durations_s = [380e-6, 68e-6, 55e-6]
    opll_hz = [80e6, 79.9e6, 80.1e6]
    switch_hz = [200e6, 200.005e6, 200.001e6]
    delivery_hz = [99e6, 99e6, 99e6]
    # Full-precision, distinct, sub-volt setpoints -- the thing the old int64
    # format truncated away.
    delivery_setpoint = [1.234, 2.718, 0.577]

    record = _encode_regular_record(
        is_up,
        start_times_s,
        durations_s,
        opll_hz,
        switch_hz,
        delivery_hz,
        delivery_setpoint,
    )
    flat = np.asarray(record, dtype=np.float64)
    offsets = np.array([0], dtype=np.int64)

    decoded = decode_pulse_record_flat(flat, offsets)
    assert len(decoded) == 1
    dump = decoded[0]
    assert isinstance(dump, LabPulseDump)

    assert dump.is_up.dtype == bool
    np.testing.assert_array_equal(dump.is_up, [True, False, True])
    np.testing.assert_allclose(dump.start_times_s, start_times_s)
    np.testing.assert_allclose(dump.durations_s, durations_s)
    np.testing.assert_allclose(dump.opll_hz, opll_hz)
    np.testing.assert_allclose(dump.switch_hz, switch_hz)
    np.testing.assert_allclose(dump.delivery_hz, delivery_hz)
    np.testing.assert_allclose(dump.delivery_setpoint, delivery_setpoint)
    # Legacy 7-row records have no phase row -> zero phase for every pulse.
    np.testing.assert_array_equal(
        dump.interferometry_phase_turns, np.zeros(len(is_up))
    )


def test_decode_pulse_record_flat_decodes_eighth_phase_row():
    """A current-format 8-row record round-trips the interferometry phase
    (in turns) into LabPulseDump, while the 7 documented rows are unaffected."""
    is_up = [1.0, 0.0, 1.0]
    start_times_s = [0.0, 1.5e-3, 4.2e-3]
    durations_s = [380e-6, 68e-6, 55e-6]
    opll_hz = [80e6, 79.9e6, 80.1e6]
    switch_hz = [200e6, 200.005e6, 200.001e6]
    delivery_hz = [99e6, 99e6, 99e6]
    delivery_setpoint = [1.234, 2.718, 0.577]
    phase_turns = [0.0, 0.25, 1.5]

    record = _encode_regular_record(
        is_up,
        start_times_s,
        durations_s,
        opll_hz,
        switch_hz,
        delivery_hz,
        delivery_setpoint,
        interferometry_phase_turns=phase_turns,
    )
    # Sanity: this really is the 8-row layout (1 + 8 * num_pulses).
    assert len(record) == 1 + 8 * len(is_up)

    decoded = decode_pulse_record_flat(np.asarray(record, dtype=np.float64), [0])
    dump = decoded[0]
    assert isinstance(dump, LabPulseDump)
    # The seven original rows still decode exactly as before.
    np.testing.assert_allclose(dump.start_times_s, start_times_s)
    np.testing.assert_allclose(dump.delivery_setpoint, delivery_setpoint)
    # ...and the new 8th row is carried through verbatim (still in turns).
    np.testing.assert_allclose(dump.interferometry_phase_turns, phase_turns)


def test_decode_pulse_record_flat_rejects_ambiguous_row_count():
    """A record whose length matches neither the 7- nor 8-row layout raises."""
    # num_pulses=3 -> valid lengths are 22 (7-row) or 25 (8-row); make it 23.
    bad = np.concatenate([[3.0], np.zeros(22)])
    with pytest.raises(ValueError, match="7-row|8-row"):
        decode_pulse_record_flat(bad, [0])


def test_build_sequence_threads_interferometry_phase_turns_to_radians():
    """The recorded phase (turns) reaches Pulse.phi as radians (x 2*pi)."""
    dump = _minimal_lab_dump([True, False])
    dump["interferometry_phase_turns"] = np.array([0.0, 0.25])
    _, sequence = build_sequence_from_lab_pulse_dump(**dump)
    phis = [event.phi for event in sequence if isinstance(event, Pulse)]
    np.testing.assert_allclose(phis, [0.0, np.pi / 2])


def test_build_sequence_defaults_phase_to_zero_when_omitted():
    """Callers that don't supply the phase (legacy dumps / hand-built dicts)
    get zero phase on every pulse, matching the old hard-coded behaviour."""
    dump = _minimal_lab_dump([True, False])
    assert "interferometry_phase_turns" not in dump
    _, sequence = build_sequence_from_lab_pulse_dump(**dump)
    phis = [event.phi for event in sequence if isinstance(event, Pulse)]
    assert phis == [0.0, 0.0]


def test_decode_pulse_record_flat_feeds_build_sequence():
    """A decoded dump splats straight into build_sequence_from_lab_pulse_dump."""
    import dataclasses

    record = _encode_regular_record(
        is_up=[1.0, 0.0],
        start_times_s=[0.0, 1e-3],
        durations_s=[95e-6, 95e-6],
        opll_hz=[80e6, 80e6],
        switch_hz=[0.0, 0.0],
        delivery_hz=[0.0, 0.0],
        delivery_setpoint=[1.5, 2.5],
    )
    decoded = decode_pulse_record_flat(record, [0])
    _, sequence = build_sequence_from_lab_pulse_dump(**dataclasses.asdict(decoded[0]))
    pulse_ks = [event.k for event in sequence if isinstance(event, Pulse)]
    assert pulse_ks == [1, -1]


def test_decode_pulse_record_flat_multiple_records_and_offsets():
    """Several records concatenated are split by the offsets array."""
    rec_a = _encode_regular_record(
        is_up=[1.0],
        start_times_s=[0.0],
        durations_s=[380e-6],
        opll_hz=[80e6],
        switch_hz=[0.0],
        delivery_hz=[0.0],
        delivery_setpoint=[1.0],
    )
    rec_b = _encode_regular_record(
        is_up=[1.0, 0.0],
        start_times_s=[0.0, 2e-3],
        durations_s=[68e-6, 55e-6],
        opll_hz=[80e6, 80e6],
        switch_hz=[0.0, 0.0],
        delivery_hz=[0.0, 0.0],
        delivery_setpoint=[2.0, 3.0],
    )
    flat = np.concatenate([rec_a, rec_b])
    offsets = np.array([0, len(rec_a)], dtype=np.int64)

    decoded = decode_pulse_record_flat(flat, offsets)
    assert len(decoded) == 2
    assert len(decoded[0].is_up) == 1
    assert len(decoded[1].is_up) == 2
    np.testing.assert_allclose(decoded[1].start_times_s, [0.0, 2e-3])


def test_decode_pulse_record_flat_same_as_last_sentinel_reuses_previous():
    """A -1.0 sentinel reuses (the same object as) the previous decoded dump."""
    rec = _encode_regular_record(
        is_up=[1.0],
        start_times_s=[0.0],
        durations_s=[380e-6],
        opll_hz=[80e6],
        switch_hz=[0.0],
        delivery_hz=[0.0],
        delivery_setpoint=[1.0],
    )
    flat = np.concatenate([rec, [PULSE_RECORD_SAME_AS_LAST_SENTINEL]])
    offsets = np.array([0, len(rec)], dtype=np.int64)

    decoded = decode_pulse_record_flat(flat, offsets)
    assert len(decoded) == 2
    assert decoded[1] is decoded[0]


def test_decode_pulse_record_flat_disabled_sentinel_is_none():
    """A -2.0 sentinel decodes to None and does not become the 'previous'."""
    rec = _encode_regular_record(
        is_up=[1.0],
        start_times_s=[0.0],
        durations_s=[380e-6],
        opll_hz=[80e6],
        switch_hz=[0.0],
        delivery_hz=[0.0],
        delivery_setpoint=[1.0],
    )
    flat = np.concatenate(
        [
            rec,
            [PULSE_RECORD_DISABLED_SENTINEL],
            [PULSE_RECORD_SAME_AS_LAST_SENTINEL],
        ]
    )
    offsets = np.array([0, len(rec), len(rec) + 1], dtype=np.int64)

    decoded = decode_pulse_record_flat(flat, offsets)
    assert decoded[1] is None
    # The 'same as last' after the disabled shot still refers to the last
    # STORED record, not to the disabled one.
    assert decoded[2] is decoded[0]


def test_decode_pulse_record_flat_leading_same_as_last_raises():
    """A 'same as last' with no prior stored record is a hard error."""
    with pytest.raises(ValueError, match="no previous record"):
        decode_pulse_record_flat(
            np.array([PULSE_RECORD_SAME_AS_LAST_SENTINEL]), np.array([0])
        )


def test_decode_pulse_record_flat_rounds_noisy_num_pulses_and_directions():
    """num_pulses and the sentinels are integer-valued floats handled with a
    tolerance; tiny float noise must not break decoding."""
    record = _encode_regular_record(
        is_up=[0.9999999, 0.0000001],
        start_times_s=[0.0, 1e-3],
        durations_s=[95e-6, 95e-6],
        opll_hz=[80e6, 80e6],
        switch_hz=[0.0, 0.0],
        delivery_hz=[0.0, 0.0],
        delivery_setpoint=[1.0, 2.0],
    )
    record[0] = 2.0000001  # noisy num_pulses
    decoded = decode_pulse_record_flat(record, [0])
    np.testing.assert_array_equal(decoded[0].is_up, [True, False])


def test_decode_pulse_record_flat_bad_length_raises():
    """A record whose length disagrees with num_pulses is rejected loudly."""
    # num_pulses = 2 implies length 1 + 7*2 = 15, but give 14.
    bad = np.zeros(14, dtype=np.float64)
    bad[0] = 2.0
    with pytest.raises(ValueError, match="length"):
        decode_pulse_record_flat(bad, np.array([0]))


# --- Shaped-pulse stand-ins: arm-restricted, simultaneous pi pulses ---------


def _stand_in_pulse(**overrides):
    rabi = 9090.9
    defaults = dict(
        k=+1,
        detuning_hz=5 * RECOIL_FREQUENCY_HZ,
        phi=0.0,
        label="stand_in",
        rabi_frequency=rabi,
        duration=1 / (2 * rabi),
    )
    defaults.update(overrides)
    return Pulse(**defaults)


def test_restricted_pulse_only_addresses_its_momentum_class():
    """A restricted pulse flips its target class as usual but is a strict
    no-op -- not merely off-resonant -- for every other momentum class."""
    pulse = _stand_in_pulse(restrict_to_m_ground=2)
    # Target class: ground m=2 (and its partner, excited m=3), resonant pi.
    assert _transition_probability(2, True, pulse) == pytest.approx(1.0, abs=1e-6)
    assert _transition_probability(3, False, pulse) == pytest.approx(1.0, abs=1e-6)
    # Any other class: exactly zero, even though a plain pulse would still
    # interact off-resonantly.
    assert _transition_probability(0, True, pulse) == 0.0
    assert _transition_probability(1, False, pulse) == 0.0
    unrestricted = _stand_in_pulse()
    assert _transition_probability(0, True, unrestricted) > 0.0


def test_addressed_momentum_classes_follow_the_restriction():
    """For a restricted stand-in the restriction IS the addressing, whatever
    the detuning says."""
    pulse = _stand_in_pulse(
        detuning_hz=2.37 * RECOIL_FREQUENCY_HZ, restrict_to_m_ground=2
    )
    assert _addressed_momentum_classes(pulse) == (2.0, 3.0)


def test_simultaneous_pulse_requires_restriction():
    with pytest.raises(ValueError, match="restrict_to_m_ground"):
        _stand_in_pulse(simultaneous_with_previous=True)


def test_simultaneous_pulse_must_follow_a_pulse():
    lone = _stand_in_pulse(restrict_to_m_ground=0, simultaneous_with_previous=True)
    with pytest.raises(ValueError, match="follow"):
        compute_spacetime_trajectory([lone])
    with pytest.raises(ValueError, match="follow"):
        compute_spacetime_trajectory([Freefall(duration=1e-3), lone])


def test_quantum_runner_guards_on_shaped_pulse_stand_ins():
    """Full quantum propagation of the stand-ins is NOT implemented; the
    runner must fail loudly rather than apply wrong physics."""
    state = make_atom_states(
        velocity_x=0.0,
        velocity_y=0.0,
        initial_velocity_z=0.0,
        position_x=0.0,
        position_y=0.0,
        position_z=0.0,
    )
    restricted = _stand_in_pulse(restrict_to_m_ground=2)
    with pytest.raises(NotImplementedError, match="shaped"):
        run_pulse_sequence_in_lab_frame(state, [restricted])


def test_double_launch_stand_in_kicks_both_arms_simultaneously():
    """The rid74108 topology: velocity selection, pi/2 splitter, then ONE
    shaped pulse modelled as two simultaneous arm-restricted pi pulses that
    kick the two arms in opposite momentum directions."""
    rabi = 9090.9
    t_pi = 1 / (2 * rabi)
    sequence = [
        # VS: g m=0 -> e m=1
        Pulse(
            k=+1,
            detuning_hz=1 * RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="VS",
            rabi_frequency=rabi,
            duration=t_pi,
        ),
        Freefall(duration=1e-3),
        # pi/2 down splitter on the shelved atom: e m=1 <-> g m=2
        Pulse(
            k=-1,
            detuning_hz=-3 * RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="splitter",
            rabi_frequency=rabi,
            duration=t_pi / 2,
        ),
        # shaped-pulse stand-in: upper arm g m=2 -> e m=3 (absorption) ...
        Pulse(
            k=+1,
            detuning_hz=5 * RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="shaped_upper",
            rabi_frequency=rabi,
            duration=t_pi,
            restrict_to_m_ground=2,
        ),
        # ... and lower arm e m=1 -> g m=0 (stimulated emission), at the
        # same instant
        Pulse(
            k=+1,
            detuning_hz=1 * RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="shaped_lower",
            rabi_frequency=rabi,
            duration=t_pi,
            restrict_to_m_ground=0,
            simultaneous_with_previous=True,
        ),
        Freefall(duration=1e-3),
    ]
    clouds, _ = compute_spacetime_trajectory(sequence)
    assert len(clouds) == 2
    finals = {(cloud.m[-1], cloud.is_ground[-1]) for cloud in clouds}
    assert finals == {(3, False), (0, True)}
    for cloud in clouds:
        # One entry per event after the initial point...
        assert len(cloud.times) == len(sequence) + 1
        # ...and the simultaneous pulse must not have advanced the clock:
        # the post-upper and post-lower timestamps coincide.
        assert cloud.times[4] == cloud.times[5]
