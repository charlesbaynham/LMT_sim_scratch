from dataclasses import dataclass
import logging

import numpy as np

from lmt_sim.lmt_simulation import RABI_FREQ, T_PI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pulse:
    k: int
    detuning_hz: float
    phi: float
    label: str
    rabi_frequency: float
    duration: float

    def __post_init__(self):
        if self.k not in (-1, +1):
            raise ValueError("Pulse k must be either +1 or -1")
        if self.rabi_frequency <= 0.0:
            raise ValueError("Pulse rabi_frequency must be positive")
        if self.duration < 0.0:
            raise ValueError("Pulse duration must be non-negative")

    @property
    def pulse_area(self):
        return self.duration * 2 * np.pi * self.rabi_frequency


@dataclass(frozen=True)
class Clearout:
    duration: float
    label: str = "clearout"

    def __post_init__(self):
        if self.duration < 0.0:
            raise ValueError("Clearout duration must be non-negative")


@dataclass(frozen=True)
class Freefall:
    duration: float
    label: str = "freefall"

    def __post_init__(self):
        if self.duration < 0.0:
            raise ValueError("Freefall duration must be non-negative")


def build_mach_zehnder_pulse_sequence(
    phi=0.0,
    detuning_hz=None,  # FIXME make this positional
    time_between_pulses=200e-6,
    rabi_frequency=None,  # FIXME make this positional
    pulse_area_multiplier=1.0,
    k=+1,
):
    import lmt_sim.lmt_simulation as sim

    if detuning_hz is None:
        detuning_hz = sim.RECOIL_FREQUENCY_HZ
    if rabi_frequency is None:
        rabi_frequency = sim.RABI_FREQ

    t_pi = 1 / (2 * rabi_frequency)
    first_pulse = Pulse(
        k=k,
        detuning_hz=detuning_hz,
        phi=0.0,
        label="beam_splitter_1",
        rabi_frequency=rabi_frequency,
        duration=t_pi * pulse_area_multiplier / 2,
    )
    second_pulse = Pulse(
        k=k,
        detuning_hz=detuning_hz,
        phi=phi,
        label="mirror",
        rabi_frequency=rabi_frequency,
        duration=t_pi * pulse_area_multiplier,
    )
    third_pulse = Pulse(
        k=k,
        detuning_hz=detuning_hz,
        phi=4 * phi,
        label="beam_splitter_2",
        rabi_frequency=rabi_frequency,
        duration=t_pi * pulse_area_multiplier / 2,
    )

    if time_between_pulses > 0.0:
        return [
            first_pulse,
            Freefall(duration=time_between_pulses, label="dark_time_1"),
            second_pulse,
            Freefall(duration=time_between_pulses, label="dark_time_2"),
            third_pulse,
        ]
    return [first_pulse, second_pulse, third_pulse]


def run_pulse_sequence_in_borde_representation(
    m_values,
    positions,
    velocities,
    squiggly_amplitudes,
    internal_is_ground,
    pulse_sequence,
    initial_velocity_z=0.0,
    rng=None,
):
    """Run a pulse sequence on amplitudes already expressed in the Bordé frame."""
    import lmt_sim.lmt_simulation as sim

    if not pulse_sequence:
        raise ValueError("pulse_sequence must contain at least one pulse")

    for event in pulse_sequence:
        if not isinstance(event, (Pulse, Clearout, Freefall)):
            raise TypeError(f"Unsupported sequence event type: {type(event)!r}")

    detunings_hz = [
        pulse.detuning_hz for pulse in pulse_sequence if isinstance(pulse, Pulse)
    ]
    if len(detunings_hz) == 0:
        logger.warning("No pulses in sequence, defaulting to zero detuning")
        current_detuning_hz = 0.0
    else:
        current_detuning_hz = detunings_hz[0]

    current_time = 0.0

    # Process the sequence event by event
    for event in pulse_sequence:
        if isinstance(event, Pulse):
            # If it's a Pulse, apply it to the state. This includes
            # ballistically propagating the states

            # If the frequency has changed, transform the states to the new frame
            new_detuning_hz = event.detuning_hz
            if new_detuning_hz != current_detuning_hz:
                squiggly_amplitudes = (
                    sim.change_laser_frequency_in_borde_representation(
                        m_values,
                        squiggly_amplitudes,
                        internal_is_ground,
                        positions,
                        velocities,
                        new_detuning_hz=new_detuning_hz,
                        old_detuning_hz=current_detuning_hz,
                        time=current_time,
                    )
                )
                current_detuning_hz = new_detuning_hz

            # Do the pulse interaction
            m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
                sim.pulse_interaction_in_borde_representation(
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
                    k_wavevector=sim.K_WAVEVECTOR,
                    vz=initial_velocity_z,
                )
            )

        elif isinstance(event, Clearout):
            # If it's a clearout, do the projection and abort if the atom is
            # cleared out. N.B. This does not do balliastic propegation - we
            # must do it later
            result = sim.do_clearout(
                m_values,
                squiggly_amplitudes,
                internal_is_ground,
                positions,
                velocities,
                rng=rng,
            )
            if result is None:
                return None
            m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
                result
            )

        if (
            isinstance(event, Freefall) or isinstance(event, Clearout)
        ) and event.duration > 0.0:
            # Propegate the atom states ballistically during freefall or after clearout

            # for prior_event in reversed(sequence[:event_index]):
            #     if isinstance(prior_event, Pulse):
            #         return prior_event.k

            # for later_event in sequence[event_index + 1 :]:
            #     if isinstance(later_event, Pulse):
            #         return later_event.k

            # return +1

            (
                m_values,
                squiggly_amplitudes,
                internal_is_ground,
                positions,
                velocities,
            ) = sim.propagate_states_in_borde_representation(
                m_values,
                squiggly_amplitudes,
                internal_is_ground,
                positions,
                velocities,
                time_of_propegation=event.duration,
                detuning_hz=current_detuning_hz,
                vz=initial_velocity_z,
                k_wavevector=sim.K_WAVEVECTOR,
            )

        current_time += event.duration

    return (
        m_values,
        squiggly_amplitudes,
        internal_is_ground,
        positions,
        velocities,
        current_detuning_hz,
        current_time,
    )


def do_rabi_pulse(pulse_detuning, pulse_duration=T_PI, initial_velocity_z=0.0):
    """Compute excitation fraction for a single pulse.

    Parameters
    ----------
    pulse_detuning : float
        Laser detuning from resonance in Hz
    pulse_duration : float
        Pulse duration in seconds (default: T_PI for pi pulse)
    initial_velocity_z : float
        Initial atom velocity in m/s

    Returns
    -------
    float
        Excitation fraction (probability of being in excited state)
    """

    pulse_sequence = [
        Pulse(
            k=+1,
            detuning_hz=pulse_detuning,
            phi=0.0,
            label="rabi_pulse",
            rabi_frequency=RABI_FREQ,
            duration=pulse_duration,
        )
    ]

    return calculate_excited_fraction_for_pulse_sequence(
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
    )
