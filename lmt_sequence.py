from dataclasses import dataclass
import logging

import numpy as np

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


def _sequence_clock_pulses(sequence):
    return [event for event in sequence if isinstance(event, Pulse)]


def _sequence_event_k_sign(sequence, event_index):
    event = sequence[event_index]
    if isinstance(event, Pulse):
        return event.k

    for prior_event in reversed(sequence[:event_index]):
        if isinstance(prior_event, Pulse):
            return prior_event.k

    for later_event in sequence[event_index + 1 :]:
        if isinstance(later_event, Pulse):
            return later_event.k

    return +1


def build_mach_zehnder_pulse_sequence(
    phi=0.0,
    detuning_hz=None,
    time_between_pulses=200e-6,
    rabi_frequency=None,
    pulse_area_multiplier=1.0,
    k=+1,
):
    import lmt_simulation as sim

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
    import lmt_simulation as sim

    if not pulse_sequence:
        raise ValueError("pulse_sequence must contain at least one pulse")

    clock_pulses = _sequence_clock_pulses(pulse_sequence)
    if not clock_pulses:
        raise ValueError("pulse_sequence must contain at least one clock Pulse")

    for event in pulse_sequence:
        if not isinstance(event, (Pulse, Clearout, Freefall)):
            raise TypeError(f"Unsupported sequence event type: {type(event)!r}")

    detunings_hz = {pulse.detuning_hz for pulse in clock_pulses}
    if len(detunings_hz) != 1:
        raise ValueError(
            "All pulses must currently use the same detuning for Bordé-frame propagation"
        )

    omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + clock_pulses[0].detuning_hz)
    current_time = 0.0

    for event_index, event in enumerate(pulse_sequence):
        if isinstance(event, Pulse):
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
            current_time += event.duration
            continue

        if isinstance(event, Clearout):
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
            if event.duration > 0.0:
                k_sign = _sequence_event_k_sign(pulse_sequence, event_index)
                m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
                    sim.propagate_states_in_borde_representation(
                        m_values,
                        squiggly_amplitudes,
                        internal_is_ground,
                        positions,
                        velocities,
                        time_of_propegation=event.duration,
                        omega_laser=omega_laser,
                        vz=initial_velocity_z,
                        k_sign=k_sign,
                        k_wavevector=sim.K_WAVEVECTOR,
                    )
                )
                current_time += event.duration
            continue

        if event.duration > 0.0:
            k_sign = _sequence_event_k_sign(pulse_sequence, event_index)
            m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
                sim.propagate_states_in_borde_representation(
                    m_values,
                    squiggly_amplitudes,
                    internal_is_ground,
                    positions,
                    velocities,
                    time_of_propegation=event.duration,
                    omega_laser=omega_laser,
                    vz=initial_velocity_z,
                    k_sign=k_sign,
                    k_wavevector=sim.K_WAVEVECTOR,
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


def calculate_excited_fraction_for_pulse_sequence(
    pulse_sequence,
    initial_velocity_z=0.0,
):
    """Run a lab-frame pulse sequence and return the final excited-state fraction."""
    import lmt_simulation as sim

    if not pulse_sequence:
        raise ValueError("pulse_sequence must contain at least one pulse")

    for event in pulse_sequence:
        if not isinstance(event, (Pulse, Freefall, Clearout)):
            raise TypeError(f"Unsupported sequence event type: {type(event)!r}")

    if any(isinstance(event, Clearout) for event in pulse_sequence):
        raise ValueError(
            "calculate_excited_fraction_for_pulse_sequence does not support Clearout events"
        )

    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        sim.make_atom_states(initial_velocity_z=initial_velocity_z)
    )
    clock_pulses = _sequence_clock_pulses(pulse_sequence)
    if not clock_pulses:
        raise ValueError("pulse_sequence must contain at least one clock Pulse")
    omega_laser = 2 * np.pi * (sim.TRANSITION_FREQUENCY + clock_pulses[0].detuning_hz)
    squiggly_amplitudes = sim.transform_state_vector(
        m_values,
        internal_amplitude,
        internal_is_ground,
        omega_laser=omega_laser,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )

    (
        m_values,
        squiggly_amplitudes,
        internal_is_ground,
        positions,
        velocities,
        _omega_laser,
        current_time,
    ) = run_pulse_sequence_in_borde_representation(
        m_values,
        positions,
        velocities,
        squiggly_amplitudes,
        internal_is_ground,
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
    )

    internal_amplitude_final = sim.transform_state_vector(
        m_values,
        squiggly_amplitudes,
        internal_is_ground,
        omega_laser=omega_laser,
        t=current_time,
        z=0.0,
        vz=initial_velocity_z,
        inverse=True,
    )

    ground_prob, excited_prob = sim.calculate_ground_and_excited_probabilities(
        m_values,
        internal_amplitude_final,
        internal_is_ground,
    )

    total_prob = ground_prob + excited_prob
    if not np.isclose(total_prob, 1.0, rtol=1e-6):
        logger.warning("State is not normalized after pulse sequence: total_prob=%s", total_prob)

    return excited_prob / total_prob
