import numpy as np
import pytest
from scipy import constants

from lmt_simulation import (
    MASS_ATOM,
    RABI_FREQ,
    RECOIL_FREQUENCY_HZ,
    RECOIL_VELOCITY,
    T_PI,
    TRANSITION_WAVELENGTH,
    propagate_states_pulse,
    make_atom_states,
    propagate_states_freely,
)


def do_pi_pulse_on_state(c0, c1):
    m, pos, amp, isg = make_atom_states(
        position_z=0.0, initial_velocity_z=0.0, c0=c0, c1=c1
    )
    new_m, new_pos, new_amp, new_isg = propagate_states_pulse(
        m,
        pos,
        amp,
        isg,
        initial_velocity_z=0.0,
        laser_direction=+1,
        pulse_duration=T_PI,
        pulse_phase=0.0,
        pulse_detuning=RECOIL_FREQUENCY_HZ,
    )
    return new_m, new_pos, new_amp, new_isg


def test_pi_pulse():
    """Test that a pi pulse on a stationary atom transfers all population to the excited state."""
    new_m, new_pos, new_amp, new_isg = do_pi_pulse_on_state(c0=1.0, c1=0.0)

    p_transfer = np.abs(new_amp[2]) ** 2

    print("\nTest pi pulse from ground state:")
    print("New m:", new_m)
    print("New pos:", new_pos)
    print("New amp:", new_amp)
    print("New isg:", new_isg)
    print(f"Population transferred to excited state: {p_transfer:.4f}")

    assert (
        p_transfer > 0.99
    ), "Pi pulse did not transfer population to excited state as expected"
    assert (
        p_transfer < 1.01
    ), "Pi pulse transferred more than 100% population, which is unphysical"


def test_pulse_series():
    m, pos, amp, isg = make_atom_states(
        position_z=0.0, initial_velocity_z=0.0, c0=1.0, c1=0.0
    )
    new_m, new_pos, new_amp, new_isg = propagate_states_pulse(
        m,
        pos,
        amp,
        isg,
        initial_velocity_z=0.0,
        laser_direction=+1,
        pulse_duration=T_PI / 2,
        pulse_phase=0.0,
        pulse_detuning=RECOIL_FREQUENCY_HZ,
    )

    new_m, new_pos, new_amp, new_isg = propagate_states_pulse(
        new_m,
        new_pos,
        new_amp,
        new_isg,
        initial_velocity_z=0.0,
        laser_direction=+1,
        pulse_duration=T_PI,
        pulse_phase=0.0,
        pulse_detuning=RECOIL_FREQUENCY_HZ,
    )

    p_transfer = np.abs(new_amp[2]) ** 2

    print("\nTest two pi pulses from ground state:")
    print("New m:", new_m)
    print("New pos:", new_pos)
    print("New amp:", new_amp)
    print("New isg:", new_isg)
    print(f"Population transferred back to ground state: {p_transfer:.4f}")

    sq_abs = np.abs(new_amp) ** 2
    print("Square of absolute values of new_amp:", sq_abs)

    assert (
        p_transfer > 0.99
    ), "Two pi pulses did not transfer population back to ground state as expected"
    assert (
        p_transfer < 1.01
    ), "Two pi pulses transferred more than 100% population, which is unphysical"


test_pulse_series()
