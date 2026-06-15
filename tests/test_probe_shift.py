import numpy as np

from lmt_sim.lmt_simulation import (
    RABI_FREQ,
    RECOIL_FREQUENCY_HZ,
    T_PI,
    K_WAVEVECTOR,
    make_atom_states,
    pulse_interaction_in_borde_representation,
    transform_state_vector,
)
from lmt_sim.lmt_sequence import (
    Pulse,
    calculate_excited_fraction_for_pulse_sequence,
)


# Coefficient (1/Hz) such that a 45 us pi pulse (Rabi = RABI_FREQ) produces a
# 1 kHz probe shift: shift_Hz = coefficient * Rabi_Hz**2.
PROBE_COEFF_1KHZ = 1000.0 / RABI_FREQ**2


def _prepared_state(c0=1.0, c1=0.0, pulse_detuning_hz=RECOIL_FREQUENCY_HZ):
    """Make an atom state already transformed into the Borde frame."""
    state = make_atom_states(position_z=0.0, initial_velocity_z=0.0, c0=c0, c1=c1)
    return transform_state_vector(
        state, detuning_hz=pulse_detuning_hz, t=0.0, z=0.0, vz=0.0, inverse=False
    )


def _do_pulse(pulse_detuning_hz, probe_shift_coefficient=0.0, t_pulse=T_PI):
    state = _prepared_state(pulse_detuning_hz=pulse_detuning_hz)
    return pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=pulse_detuning_hz,
        t_pulse=t_pulse,
        pulse_rabi_freq=RABI_FREQ,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=0.0,
        probe_shift_coefficient=probe_shift_coefficient,
    )


def test_calibration_constant_gives_1khz():
    """The chosen coefficient maps a 45 us pi pulse onto exactly a 1 kHz shift."""
    assert np.isclose(PROBE_COEFF_1KHZ * RABI_FREQ**2, 1000.0)


def test_zero_coefficient_is_backward_compatible():
    """probe_shift_coefficient=0 reproduces the original (no-parameter) result."""
    detuning = RECOIL_FREQUENCY_HZ

    state = _prepared_state(pulse_detuning_hz=detuning)
    baseline = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning,
        t_pulse=T_PI,
        pulse_rabi_freq=RABI_FREQ,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=0.0,
    )
    with_param = _do_pulse(detuning, probe_shift_coefficient=0.0)

    assert np.allclose(baseline.amplitudes, with_param.amplitudes)


def test_probe_shift_equivalent_to_detuning_shift():
    """A probe shift is identical to lowering the bare detuning by coeff*Rabi**2.

    The shift is subtracted (no factor of 2*pi): the recorded detuning sits above
    the bare resonance by this amount, so removing it recovers the bare ladder.
    """
    detuning = RECOIL_FREQUENCY_HZ
    coeff = PROBE_COEFF_1KHZ
    shift_hz = coeff * RABI_FREQ**2  # 1000 Hz

    shifted = _do_pulse(detuning, probe_shift_coefficient=coeff)
    bumped = _do_pulse(detuning - shift_hz, probe_shift_coefficient=0.0)

    assert np.allclose(shifted.amplitudes, bumped.amplitudes)


def test_probe_shift_detunes_a_resonant_pi_pulse():
    """Turning on the probe shift detunes an otherwise-resonant pi pulse, lowering
    the excited-state transfer."""
    # On resonance Omega_3 = 0 requires detuning that compensates the recoil
    # shift of one photon: delta = RECOIL_FREQUENCY_HZ for m_ground = 0, k = +1.
    detuning = RECOIL_FREQUENCY_HZ
    # Use a 5 kHz shift so the detuning visibly suppresses the transfer (a 1 kHz
    # shift only nudges the broad pi-pulse resonance by < 1%).
    coeff_5khz = 5000.0 / RABI_FREQ**2

    resonant = _do_pulse(detuning, probe_shift_coefficient=0.0)
    shifted = _do_pulse(detuning, probe_shift_coefficient=coeff_5khz)

    p_exc_resonant = np.abs(resonant.amplitudes[~resonant.internal_is_ground]).sum() ** 2
    p_exc_shifted = np.abs(shifted.amplitudes[~shifted.internal_is_ground]).sum() ** 2

    assert p_exc_resonant > 0.99, "pi pulse should be ~fully resonant without shift"
    assert p_exc_shifted < p_exc_resonant - 0.05, "probe shift should detune the pulse"


def test_stark_rabi_freq_overrides_shift_only():
    """stark_rabi_freq changes the light shift but not the pulse dynamics.

    A pulse with stark_rabi_freq=Omega_s must match a pulse whose detuning is
    bumped by coeff * Omega_s**2 (instead of coeff * pulse_rabi**2) but whose
    coupling is unchanged.
    """
    detuning = RECOIL_FREQUENCY_HZ
    coeff = PROBE_COEFF_1KHZ
    stark_rabi = 3.0 * RABI_FREQ
    shift_hz = coeff * stark_rabi**2

    state = _prepared_state(pulse_detuning_hz=detuning)
    shaped = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning,
        t_pulse=T_PI,
        pulse_rabi_freq=RABI_FREQ,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=0.0,
        probe_shift_coefficient=coeff,
        stark_rabi_freq=stark_rabi,
    )
    bumped = _do_pulse(detuning - shift_hz, probe_shift_coefficient=0.0)

    assert np.allclose(shaped.amplitudes, bumped.amplitudes)


def test_stark_rabi_freq_none_is_backward_compatible():
    """stark_rabi_freq=None reproduces the rabi_freq-based shift exactly."""
    detuning = RECOIL_FREQUENCY_HZ
    coeff = PROBE_COEFF_1KHZ

    state = _prepared_state(pulse_detuning_hz=detuning)
    explicit_none = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning,
        t_pulse=T_PI,
        pulse_rabi_freq=RABI_FREQ,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=0.0,
        probe_shift_coefficient=coeff,
        stark_rabi_freq=None,
    )
    default = _do_pulse(detuning, probe_shift_coefficient=coeff)

    assert np.allclose(explicit_none.amplitudes, default.amplitudes)


def test_shaped_pulse_resonant_only_with_true_stark_rabi():
    """A shaped pulse compensated in the lab for its TRUE light shift is only
    resonant in the sim if the shift is computed from the true Rabi frequency.

    The shaped pulse is several times longer than a pi pulse but keeps the full
    intensity (Rabi = RABI_FREQ). We pretend it is a plain pi pulse, so its
    fictitious dynamics Rabi frequency is 1 / (2 * duration). The lab detuning
    includes compensation for the true shift coeff * RABI_FREQ**2.
    """
    duration = 4 * T_PI
    fictitious_rabi = 1.0 / (2.0 * duration)
    # 5 kHz true shift (cf. test_probe_shift_detunes_a_resonant_pi_pulse): large
    # enough that computing it from the fictitious Rabi visibly breaks resonance.
    coeff = 5000.0 / RABI_FREQ**2
    lab_detuning = RECOIL_FREQUENCY_HZ + coeff * RABI_FREQ**2

    common = dict(
        k=+1,
        detuning_hz=lab_detuning,
        phi=0.0,
        label="shaped",
        rabi_frequency=fictitious_rabi,
        duration=duration,
        probe_shift_coefficient=coeff,
    )
    seq_marked = [Pulse(stark_rabi_frequency=RABI_FREQ, **common)]
    seq_unmarked = [Pulse(**common)]

    frac_marked = calculate_excited_fraction_for_pulse_sequence(seq_marked)
    frac_unmarked = calculate_excited_fraction_for_pulse_sequence(seq_unmarked)

    assert frac_marked > 0.99, "true-Rabi shift should restore resonance"
    assert frac_unmarked < 0.5, "fictitious-Rabi shift should leave the pulse far off-resonance"


def test_pulse_rejects_non_positive_stark_rabi():
    import pytest

    with pytest.raises(ValueError):
        Pulse(
            k=+1,
            detuning_hz=0.0,
            phi=0.0,
            label="bad",
            rabi_frequency=RABI_FREQ,
            duration=T_PI,
            stark_rabi_frequency=-1.0,
        )


def test_probe_shift_at_sequence_level():
    """Setting probe_shift_coefficient on a Pulse matches bumping its detuning."""
    detuning = RECOIL_FREQUENCY_HZ
    coeff = PROBE_COEFF_1KHZ
    shift_hz = coeff * RABI_FREQ**2

    common = dict(
        k=+1,
        phi=0.0,
        label="pulse",
        rabi_frequency=RABI_FREQ,
        duration=T_PI,
    )

    seq_shift = [Pulse(detuning_hz=detuning, probe_shift_coefficient=coeff, **common)]
    seq_bump = [Pulse(detuning_hz=detuning - shift_hz, **common)]

    frac_shift = calculate_excited_fraction_for_pulse_sequence(seq_shift)
    frac_bump = calculate_excited_fraction_for_pulse_sequence(seq_bump)

    assert np.isclose(frac_shift, frac_bump)
