import numpy as np

from lmt_sim.lmt_simulation import (
    RABI_FREQ,
    RECOIL_FREQUENCY_HZ,
    T_PI,
    K_WAVEVECTOR,
    calculate_ground_and_excited_probabilities,
    make_atom_states,
    pulse_interaction_in_borde_representation,
    transform_state_vector,
)


def run_pulse_sequence(c0, c1, pulse_durations, pulse_detuning_hz=RECOIL_FREQUENCY_HZ):
    state = make_atom_states(position_z=0.0, initial_velocity_z=0.0, c0=c0, c1=c1)
    state = transform_state_vector(
        state,
        detuning_hz=pulse_detuning_hz,
        t=0.0,
        z=0.0,
        vz=0.0,
        inverse=False,
    )
    for pulse_duration in pulse_durations:
        state = pulse_interaction_in_borde_representation(
            state,
            pulse_detuning=pulse_detuning_hz,
            t_pulse=pulse_duration,
            pulse_rabi_freq=RABI_FREQ,
            pulse_phase=0.0,
            k_sign=+1,
            k_wavevector=K_WAVEVECTOR,
            vz=0.0,
        )
    return calculate_ground_and_excited_probabilities(state)


def test_pi_pulse():
    """Test that a pi pulse on a stationary atom transfers all population to the excited state."""
    p_ground, p_excited = run_pulse_sequence(c0=1.0, c1=0.0, pulse_durations=[T_PI])

    assert (
        p_excited > 0.99
    ), "Pi pulse did not transfer population to excited state as expected"
    assert (
        p_excited < 1.01
    ), "Pi pulse transferred more than 100% population, which is unphysical"
    assert np.isclose(
        p_ground + p_excited, 1.0, rtol=1e-9
    ), "Population should be conserved"


def test_two_pi_pulses_return_to_ground():
    p_ground, p_excited = run_pulse_sequence(
        c0=1.0, c1=0.0, pulse_durations=[T_PI, T_PI]
    )

    assert (
        p_ground > 0.99
    ), "Two pi pulses did not transfer population back to ground state as expected"
    assert (
        p_ground < 1.01
    ), "Two pi pulses transferred more than 100% population, which is unphysical"
    assert p_excited < 1e-10, "Excited-state population should be near zero"
