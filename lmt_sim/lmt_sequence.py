from dataclasses import dataclass
import logging
import warnings

import numpy as np
import lmt_sim.lmt_simulation as sim
from lmt_sim.lmt_simulation import RABI_FREQ, T_PI
from scipy import constants

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pulse:
    k: int
    detuning_hz: float
    phi: float
    label: str
    rabi_frequency: float
    duration: float
    beam_waist: float = 1e6  # "infinite" by default
    # Probe (light) shift coefficient in 1/Hz. The effective detuning during the
    # pulse is reduced by probe_shift_coefficient * rabi_frequency**2 Hz (the
    # Rabi-squared / intensity scaling of a light shift; no factor of 2*pi).
    # Default 0.0 disables it.
    probe_shift_coefficient: float = 0.0
    # True Rabi frequency (Hz) used for the probe (light) shift only. For a
    # shaped (optimal-control) pulse modelled as a plain pi pulse,
    # rabi_frequency is the fictitious value implied by the duration (so the
    # 2x2 dynamics see a pi pulse area) while the light shift scales with the
    # true intensity, which can be much higher. None (default) means an
    # ordinary square pulse: the shift uses rabi_frequency.
    stark_rabi_frequency: float | None = None
    # --- Shaped-pulse stand-in fields ------------------------------------
    # TODO: replace this stand-in with a real shaped-pulse propagator that
    # integrates the time-dependent Hamiltonian over the recorded
    # amplitude/phase profile (see docs/roadmap.md).
    #
    # Restrict this pulse to a single transition pair: only rows/clouds whose
    # GROUND-state momentum class equals this value interact with the pulse;
    # every other momentum class is passed through completely untouched, i.e.
    # the off-resonant interaction is deliberately suppressed IN CODE, not by
    # physics. This models a phase-shaped pulse that is engineered to address
    # one arm of the interferometer without disturbing the other -- something
    # a plain square pulse cannot do. None (default): the pulse addresses all
    # rows with the usual off-resonant 2x2 physics.
    restrict_to_m_ground: int | None = None
    # Fire this pulse at the same time as the previous pulse in the sequence
    # instead of after it: the sequence clock does not advance. Used to model
    # a single shaped pulse that drives several transitions at once (e.g. the
    # double-launch JessePulseLMT) as multiple simultaneous arm-restricted
    # stand-in pulses. Requires restrict_to_m_ground: simultaneous
    # UNrestricted pulses cannot be composed from sequential 2x2 interactions.
    simultaneous_with_previous: bool = False

    def __post_init__(self):
        if self.k not in (-1, +1):
            raise ValueError("Pulse k must be either +1 or -1")
        if self.rabi_frequency <= 0.0:
            raise ValueError("Pulse rabi_frequency must be positive")
        if self.duration < 0.0:
            raise ValueError("Pulse duration must be non-negative")
        if self.stark_rabi_frequency is not None and self.stark_rabi_frequency <= 0.0:
            raise ValueError("Pulse stark_rabi_frequency must be positive if given")
        if self.simultaneous_with_previous and self.restrict_to_m_ground is None:
            raise ValueError(
                "simultaneous_with_previous requires restrict_to_m_ground: "
                "unrestricted pulses cannot meaningfully overlap in time"
            )

    @property
    def pulse_area(self):
        return self.duration * 2 * np.pi * self.rabi_frequency

    @property
    def effective_stark_rabi_frequency(self):
        """Rabi frequency the probe (light) shift is computed from."""
        if self.stark_rabi_frequency is not None:
            return self.stark_rabi_frequency
        return self.rabi_frequency


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


def run_pulse_sequence_in_lab_frame(
    state,
    pulse_sequence,
    initial_velocity_z=0.0,
    discard_threshold=1e-9,
    rng=None,
):
    """
    Run a pulse sequence on a state vector
    """
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
        # Freefall-only or clearout-only sequences are valid; use the
        # zero-detuning lab/Borde frame without logging noisy warnings.
        current_detuning_hz = 0.0
    else:
        current_detuning_hz = detunings_hz[0]

    current_time = 0.0

    # Convert to the Borde representation based on the first detuning
    state = sim.transform_state_vector(
        state,
        omega_laser=2 * np.pi * (sim.TRANSITION_FREQUENCY + current_detuning_hz),
        t=current_time,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )

    # Run the sequence in the Borde representation
    result = run_pulse_sequence_in_borde_representation(
        state,
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
        discard_threshold=discard_threshold,
        rng=rng,
    )
    if result is None:
        # Atom was cleared out
        return None
    state, current_detuning_hz, current_time = result

    # Convert back to the lab frame
    state = sim.transform_state_vector(
        state,
        omega_laser=2 * np.pi * (sim.TRANSITION_FREQUENCY + current_detuning_hz),
        t=current_time,
        z=0.0,
        vz=initial_velocity_z,
        inverse=True,
    )

    return state, current_detuning_hz, current_time


def iter_pulse_sequence_in_borde_representation(
    state,
    pulse_sequence,
    initial_velocity_z=0.0,
    discard_threshold=1e-9,
    rng=None,
):
    """Yield ``(state, current_detuning_hz, current_time)`` before the first
    event and after each event of ``pulse_sequence``.

    Source of truth for the per-event loop -- ``run_pulse_sequence_in_borde_representation``
    is a thin wrapper that exhausts this generator and returns the final tuple.

    Stops early (without a final yield) if a ``Clearout`` removes the atom;
    consumers can detect this by counting yields against ``len(pulse_sequence) + 1``.
    """
    for event in pulse_sequence:
        if not isinstance(event, (Pulse, Clearout, Freefall)):
            raise TypeError(f"Unsupported sequence event type: {type(event)!r}")
        if isinstance(event, Pulse) and (
            event.restrict_to_m_ground is not None or event.simultaneous_with_previous
        ):
            raise NotImplementedError(
                "Arm-restricted / simultaneous pulses are a stand-in for shaped "
                "pulses and are only supported by compute_spacetime_trajectory. "
                "Full quantum propagation of shaped pulses is not implemented "
                "yet (see docs/roadmap.md): a restricted pulse needs the "
                "untouched rows to free-evolve coherently during the pulse, and "
                "simultaneous pulses are not sequential 2x2 interactions."
            )

    detunings_hz = [
        pulse.detuning_hz for pulse in pulse_sequence if isinstance(pulse, Pulse)
    ]
    current_detuning_hz = detunings_hz[0] if len(detunings_hz) > 0 else 0.0
    current_time = 0.0

    yield state, current_detuning_hz, current_time

    for event in pulse_sequence:
        if isinstance(event, Pulse):
            # If it's a Pulse, apply it to the state. This includes
            # ballistically propagating the states

            # If the frequency has changed, transform the states to the new frame
            new_detuning_hz = event.detuning_hz
            if new_detuning_hz != current_detuning_hz:
                state = sim.change_laser_frequency_in_borde_representation(
                    state,
                    new_detuning_hz=new_detuning_hz,
                    old_detuning_hz=current_detuning_hz,
                    time=current_time,
                )
                current_detuning_hz = new_detuning_hz

            # Do the pulse interaction
            state = sim.do_gaussian_pulse(
                state,
                beam_waist=event.beam_waist,
                pulse_detuning=event.detuning_hz,
                t_pulse=event.duration,
                on_axis_rabi_freq=event.rabi_frequency,
                pulse_phase=event.phi,
                k_sign=event.k,
                k_wavevector=sim.K_WAVEVECTOR,
                vz=initial_velocity_z,
                probe_shift_coefficient=event.probe_shift_coefficient,
                on_axis_stark_rabi_freq=event.stark_rabi_frequency,
            )

            # If any states are below the discard threshold, discard them and renormalise
            state = sim.discard_and_renormalise_state_vector(state, discard_threshold)

        elif isinstance(event, Clearout):
            # If it's a clearout, do the projection and abort if the atom is
            # cleared out. N.B. This does not do balliastic propegation - we
            # must do it later
            result = sim.do_clearout(state, rng=rng)
            if result is None:
                return
            state = result

        if (
            isinstance(event, Freefall) or isinstance(event, Clearout)
        ) and event.duration > 0.0:
            # Propegate the atom states ballistically during freefall or after clearout
            state = sim.propagate_states_in_borde_representation(
                state,
                time_of_propegation=event.duration,
                detuning_hz=current_detuning_hz,
                vz=initial_velocity_z,
                k_wavevector=sim.K_WAVEVECTOR,
            )

        current_time += event.duration

        yield state, current_detuning_hz, current_time


def run_pulse_sequence_in_borde_representation(
    state,
    pulse_sequence,
    initial_velocity_z=0.0,
    discard_threshold=1e-9,
    rng=None,
):
    """Run a pulse sequence while staying in the Borde representation.

    If a state ends up representing less then discard_threshold of the total
    (i.e. its mod(amplitude) <= sqrt(discard_threshold)), it is discarded the
    wavefunction renormalised.

    Returns the final ``(state, detuning, time)`` tuple, or ``None`` if a
    ``Clearout`` removed the atom mid-sequence.
    """
    n_expected = len(pulse_sequence) + 1
    last = None
    n = 0
    for last in iter_pulse_sequence_in_borde_representation(
        state,
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
        discard_threshold=discard_threshold,
        rng=rng,
    ):
        n += 1
    if n < n_expected:
        return None
    return last


def calculate_excited_fraction_for_pulse_sequence(
    pulse_sequence, velocity=(0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0)
):
    """Calculate final excited-state fraction for a sequence without clearout."""
    import lmt_sim.lmt_simulation as sim

    if any(isinstance(event, Clearout) for event in pulse_sequence):
        raise ValueError(
            "calculate_excited_fraction_for_pulse_sequence does not support Clearout events"
        )

    vx, vy, vz = velocity
    x, y, z = position
    state = sim.make_atom_states(
        velocity_x=vx,
        velocity_y=vy,
        initial_velocity_z=vz,
        position_x=x,
        position_y=y,
        position_z=z,
    )

    result = run_pulse_sequence_in_lab_frame(
        state,
        pulse_sequence,
        initial_velocity_z=vz,
    )

    if result is None:
        return None

    state, *_ = result
    ground_prob, excited_prob = sim.calculate_ground_and_excited_probabilities(state)
    return excited_prob / (ground_prob + excited_prob)


def _transition_probability(m, is_ground, pulse: Pulse):
    """
    Rabi transition probability for a stationary on-axis atom at momentum class m.

    "Stationary" here means that the m=0 state is stationary - obviously an atom with non-zero m is not stationary.

    Reuses the same interaction propagator as the full simulation: the
    transition probability is |B|^2 of the Bordé 2x2 matrix. The probe (light)
    shift is folded in so the inferred trajectory matches the actual pulse.
    """
    k = pulse.k
    m_ground = m if is_ground else m - k
    if (
        pulse.restrict_to_m_ground is not None
        and m_ground != pulse.restrict_to_m_ground
    ):
        # Shaped-pulse stand-in: the pulse is engineered to leave every other
        # momentum class untouched, so suppress the off-resonant interaction
        # entirely instead of computing it.
        return 0.0
    omega_ab = np.pi * pulse.rabi_frequency
    effective_detuning = sim._effective_detuning_hz(
        pulse.detuning_hz,
        pulse.probe_shift_coefficient,
        pulse.effective_stark_rabi_frequency,
    )
    _, B, _, _ = sim._calculate_interaction_constants(
        effective_detuning,
        pulse.duration,
        omega_ab,
        k_sign=k,
        vz=0.0,
        m_ground=m_ground,
    )
    return float(abs(B) ** 2)


def _addressed_momentum_classes(pulse: Pulse):
    """Return the stationary on-axis momentum classes addressed by ``pulse``.

    This uses the same effective detuning and ``vz=0`` convention as
    ``_transition_probability`` so the overlay matches the trajectory heuristic.
    The opposite Doppler slopes for the two beams enter via ``pulse.k``.
    """
    if pulse.restrict_to_m_ground is not None:
        # Shaped-pulse stand-in: the restriction IS the addressing.
        m_ground = float(pulse.restrict_to_m_ground)
        return m_ground, m_ground + pulse.k
    effective_detuning_hz = sim._effective_detuning_hz(
        pulse.detuning_hz,
        pulse.probe_shift_coefficient,
        pulse.effective_stark_rabi_frequency,
    )
    m_ground = (
        effective_detuning_hz - sim.RECOIL_FREQUENCY_HZ
    ) / (2 * pulse.k * sim.RECOIL_FREQUENCY_HZ)
    m_excited = m_ground + pulse.k
    return float(m_ground), float(m_excited)


# Sentinel values the lab pulse recorder (PulseDMARecording in icl_experiments)
# writes as length-1 records inside ``pulse_record_flat``. They are stored as
# float64 and so must be compared with a tolerance, not for exact equality.
PULSE_RECORD_SAME_AS_LAST_SENTINEL = -1.0
PULSE_RECORD_DISABLED_SENTINEL = -2.0
# Sentinels and num_pulses are integer-valued floats; anything within this of an
# integer is treated as that integer.
_PULSE_RECORD_SENTINEL_TOL = 0.5


@dataclass
class LabPulseDump:
    """One shot's worth of decoded pulse-recorder data, in SI units.

    Field names match the keyword arguments of
    :func:`build_sequence_from_lab_pulse_dump`, so a decoded dump can be fed
    straight in with ``dataclasses.asdict(dump)``.
    """

    is_up: np.ndarray  # bool, True = up beam
    start_times_s: np.ndarray  # seconds
    durations_s: np.ndarray  # seconds
    opll_hz: np.ndarray  # Hz
    switch_hz: np.ndarray  # Hz
    delivery_hz: np.ndarray  # Hz
    delivery_setpoint: np.ndarray  # volts


def decode_pulse_record_flat(pulse_record_flat, pulse_record_offsets):
    """Decode the ``pulse_record_flat`` / ``pulse_record_offsets`` datasets.

    These two archived datasets are emitted by the lab pulse recorder
    (``PulseDMARecording._archive_encoded_pulse_records`` in icl_experiments).
    This is the consumer-side counterpart and must be kept in lockstep with it.

    ``pulse_record_flat`` is a **float64** 1-D array storing every field in
    physical (SI) units; ``pulse_record_offsets`` is an int64 1-D array giving
    the start index of each per-shot record within it. Each record is one slice
    of ``pulse_record_flat``:

    * Length-1 sentinel record:
        - ``-1.0`` -> "same as the previous shot": the previous decoded dump is
          reused (returned as the same object).
        - ``-2.0`` -> "pulse-sequence storage was disabled for this shot",
          decoded as ``None``.
    * Regular record of length ``1 + 7 * num_pulses``::

        [num_pulses, dir.., start.., dur.., opll.., switch.., delivery.., setpoint..]

      i.e. ``num_pulses`` followed by seven contiguous rows, each
      ``num_pulses`` long, in the order direction, start_time, duration,
      opll_freq, switch_freq, delivery_freq, delivery_setpoint. All values are
      already in SI units (seconds for the times, Hz for the frequencies, volts
      for the setpoint); **no** machine-unit -> seconds conversion is applied.

    ``num_pulses`` and the sentinels are integer-valued floats and are handled
    defensively (rounded / compared with a tolerance).

    Returns
    -------
    list
        One entry per record (``len(pulse_record_offsets)`` entries), each
        either a :class:`LabPulseDump` (SI units) or ``None`` for a shot whose
        storage was disabled.
    """
    flat = np.asarray(pulse_record_flat, dtype=np.float64)
    offsets = np.asarray(pulse_record_offsets, dtype=np.int64)

    if flat.ndim != 1 or offsets.ndim != 1:
        raise ValueError(
            "pulse_record_flat and pulse_record_offsets must both be 1-D"
        )

    decoded = []
    previous = None
    n_records = len(offsets)
    for i in range(n_records):
        start = int(offsets[i])
        end = int(offsets[i + 1]) if i + 1 < n_records else len(flat)
        record = flat[start:end]

        if len(record) == 0:
            raise ValueError(f"Pulse record {i} is empty")

        # Sentinel records are exactly one element long.
        if len(record) == 1:
            value = float(record[0])
            if (
                abs(value - PULSE_RECORD_SAME_AS_LAST_SENTINEL)
                < _PULSE_RECORD_SENTINEL_TOL
            ):
                if previous is None:
                    raise ValueError(
                        "First pulse record is a 'same as last' sentinel; there "
                        "is no previous record to reuse."
                    )
                decoded.append(previous)
                continue
            if (
                abs(value - PULSE_RECORD_DISABLED_SENTINEL)
                < _PULSE_RECORD_SENTINEL_TOL
            ):
                # Storage disabled this shot. This does NOT update ``previous``:
                # a later 'same as last' refers to the last STORED sequence, as
                # on the producer side.
                decoded.append(None)
                continue
            raise ValueError(
                f"Unrecognised length-1 pulse record value {value!r} at record "
                f"{i} (expected {PULSE_RECORD_SAME_AS_LAST_SENTINEL} or "
                f"{PULSE_RECORD_DISABLED_SENTINEL})"
            )

        # Regular record: num_pulses followed by 7 rows of num_pulses values.
        num_pulses = int(round(float(record[0])))
        if num_pulses < 0:
            raise ValueError(
                f"Pulse record {i} has negative num_pulses ({num_pulses})"
            )
        expected_len = 1 + 7 * num_pulses
        if len(record) != expected_len:
            raise ValueError(
                f"Pulse record {i} has length {len(record)} but num_pulses="
                f"{num_pulses} implies {expected_len}"
            )

        rows = record[1:].reshape(7, num_pulses)
        (
            directions,
            start_times_s,
            durations_s,
            opll_hz,
            switch_hz,
            delivery_hz,
            delivery_setpoint,
        ) = rows

        dump = LabPulseDump(
            is_up=np.round(directions).astype(bool),
            start_times_s=start_times_s.copy(),
            durations_s=durations_s.copy(),
            opll_hz=opll_hz.copy(),
            switch_hz=switch_hz.copy(),
            delivery_hz=delivery_hz.copy(),
            delivery_setpoint=delivery_setpoint.copy(),
        )
        decoded.append(dump)
        previous = dump

    return decoded


def build_sequence_from_lab_pulse_dump(
    is_up,
    start_times_s,
    durations_s,
    opll_hz,
    switch_hz,
    delivery_hz,
    delivery_setpoint,
    probe_induced_alpha_up=1.8153e-05,
    probe_induced_alpha_down=1.8153e-05,
    pi_pulse_threshold_s=50e-6,
    initial_velocity_z=0.0,
):
    """
    Parse the "pulse record" arrays that the experiment spits out to define the
    pulse sequences that were fired

    This code is tightly coupled to icl_experiments and must be updated in
    lockstep. The rest of this repository is a physics simulator only - this
    function is the only place where experiemnt-specific knowledge is allowed to
    live.

    Note that the simulation code currently works in the freely falling frame,
    and assumes a zero starting velocity. We therefore compensate out the
    Doppler effect here when the pulse sequence is built. This is non ideal
    since the Doppler shift is very much physics, so it would be better to treat
    this with all the rest of the physics.

    Our sign convention is that positive z is upwards, gravity therefore
    accelerates in the -1 direction. The UP beam is the one that propegates from
    bottom to top. Its k vector has direction [0,0,+1], i.e. along the positive
    z direction
    """

    if pi_pulse_threshold_s <= 0.0:
        raise ValueError("pi_pulse_threshold_s must be positive")

    # is_up is a boolean beam mask (True = up beam, False = down). Validate
    # before casting so an accidental bitwise-NOT of an integer array
    # (e.g. ``~np.array([1, 0])`` -> ``[-2, -1]``) fails loudly here instead of
    # being silently coerced to all-True. To flip beams, pass a boolean array or
    # use np.logical_not -- never ``~`` on an integer array.
    is_up_input = np.asarray(is_up)
    if is_up_input.dtype != bool and not np.all(np.isin(is_up_input, (0, 1))):
        raise ValueError("is_up must be a boolean array (or contain only 0/1)")
    is_up = is_up_input.astype(bool)
    start_times_s = np.asarray(start_times_s, dtype=float)
    durations_s = np.asarray(durations_s, dtype=float)
    opll_hz = np.asarray(opll_hz, dtype=float)
    switch_hz = np.asarray(switch_hz, dtype=float)
    delivery_hz = np.asarray(delivery_hz, dtype=float)
    delivery_setpoint = np.asarray(delivery_setpoint, dtype=float)

    lengths = {
        len(is_up),
        len(start_times_s),
        len(durations_s),
        len(opll_hz),
        len(switch_hz),
        len(delivery_hz),
        len(delivery_setpoint),
    }

    # delivery_setpoint is now a full-precision per-pulse value in volts (the
    # producer no longer truncates it to whole volts). The simulator does not
    # yet model the delivery-intensity setpoint, so it is accepted and
    # length-checked but not used in the physics below.

    if len(lengths) != 1:
        raise ValueError("Lab pulse dump arrays must all have the same length")

    # start_time and duration arrive already in seconds (the producer converts
    # from machine units via mu_to_seconds), so no conversion is applied here.
    timestamps = start_times_s
    durations = durations_s

    # The OPLL offsets the Sirah from the ECDL and we lock to the negative side.
    # The delivery and switch AOMs all use the -1st order.
    # This "total laser frequency" is defined in the lab rest frame.
    # TODO: This should be handled in icl_experiments
    total_laser_frequency_hz = -opll_hz - switch_hz - delivery_hz

    # The overall offset is arbitrary, so normalise to the first pulse for convenience
    total_laser_frequency_hz -= total_laser_frequency_hz[0]

    # Convert boolean "is_up" into +-1 for the k_vector. This might become a vector later
    beam_sign = np.where(is_up, +1.0, -1.0)

    # Doppler shift seen by each beam from the atom's velocity. This number is
    # the difference between what the atom experiences and the UP beam's
    # frequency. In other words, if this number is positive, the atom is falling
    # towards the ground and blue-shifting the up beam.
    #
    # The lab tunes each pulse's frequency for resonance at the pulse CENTRE,
    # so the free-fall Doppler ramp must be evaluated there, not at the pulse
    # start. (For a 380 us velocity-selection pulse the difference is ~0.57
    # recoils -- large enough to wreck the anchor if evaluated at the start.)
    pulse_centre_times = timestamps + durations / 2
    up_beam_doppler_hz = (
        -initial_velocity_z / sim.TRANSITION_WAVELENGTH
        + sim.GRAVITY_DOPPLER_PER_SEC_HZ * pulse_centre_times
    )

    # Assume that the first pulse is on resonance
    rabi_freq_first_pulse = (
        1 / (2 * durations[0])
        if durations[0] > pi_pulse_threshold_s
        else 1 / (4 * durations[0])
    )

    # Get the shifts of the first pulse
    first_pulse_probe_shift_hz = (
        probe_induced_alpha_up if is_up[0] else probe_induced_alpha_down
    ) * rabi_freq_first_pulse**2
    first_pulse_doppler_shift_hz = up_beam_doppler_hz[0] * beam_sign[0]
    # The recoil energy is positive whichever way the photon kicks, so a
    # velocity-selection pulse on a stationary atom is resonant one recoil
    # frequency ABOVE the bare transition for BOTH beam directions.
    first_pulse_atom_frame_detuning_hz = sim.RECOIL_FREQUENCY_HZ

    # This is the unperturbed transition frequency for a hypothetical m=0 -> m=0
    # transition. To get it, we must subtract the probe shift that the atom was
    # experiencing during the pulse, and also subtract the detuning resultant
    # from us actually driving the m=0 -> m=+-1 transition. We assume that the
    # first pulse in any sequence -- whichever beam fired it -- is a pi pulse
    # that drives the ground state m=0 to the excited state m=+-1 along that
    # beam's direction, i.e. a velocity selection pulse. All frequencies are
    # therefore anchored on the first pulse, NOT on the first up pulse: the
    # beam-dependent pieces (probe-shift coefficient and Doppler sign) above are
    # selected by the first pulse's actual beam.
    centre_freq_hz = total_laser_frequency_hz[0] + first_pulse_doppler_shift_hz - first_pulse_atom_frame_detuning_hz - first_pulse_probe_shift_hz
    
    

    # Now we calculate the detuning of all the beams due only to gravity. The simulation will handle the probe-induced Stark shift.
    # TODO: wrap the gravity Doppler shift into the main sim

    # We define the UP beam as having k = +1, so gravity causes the up beam to
    # be BLUE-shifted
    effective_laser_detuning_hz = (
        # Recentre to the new centre freq:
        (total_laser_frequency_hz - centre_freq_hz)
        # Add the effect of the Doppler shift to bring the detunings into the
        # freely-falling frame:
        + (up_beam_doppler_hz * beam_sign  )  
    )

    sequence_timestamps = []
    sequence = []
    t_now = 0.0

    for (
        this_is_up,
        this_timestamp,
        this_duration,
        this_effective_laser_detuning_hz,
    ) in zip(
        is_up,
        timestamps,
        durations,
        effective_laser_detuning_hz,
    ):
        if this_timestamp < t_now:
            raise ValueError(
                f"Pulse timestamps must be non-decreasing. Got {this_timestamp} < {t_now}."
            )
        if this_timestamp > t_now:
            sequence_timestamps.append(t_now)
            sequence.append(Freefall(duration=this_timestamp - t_now))
            t_now = this_timestamp

        if this_duration > pi_pulse_threshold_s:
            rabi_freq_hz = 1 / (2 * this_duration)
        else:
            rabi_freq_hz = 1 / (4 * this_duration)

        sequence_timestamps.append(this_timestamp)
        sequence.append(
            # Note that we simply report the laser frequency to the simulation
            # and rely on it to sort out the probe shift etc. This method needs
            # only think about it when using the initial pulse to determing the
            # resonance frequency
            Pulse(
                k=+1 if this_is_up else -1,
                detuning_hz=this_effective_laser_detuning_hz,
                phi=0.0,
                label="LMT",
                rabi_frequency=rabi_freq_hz,
                duration=this_duration,
                probe_shift_coefficient=(
                    probe_induced_alpha_up if this_is_up else probe_induced_alpha_down
                ),
            )
        )
        t_now += this_duration

    return np.array(sequence_timestamps), sequence


def calibrate_probe_shift_and_velocity_from_dump(
    is_up,
    start_times_s,
    durations_s,
    opll_hz,
    switch_hz,
    delivery_hz,
    delivery_setpoint,
    pi_pulse_threshold_s=50e-6,
):
    r"""Infer ``(probe_shift_alpha, initial_velocity_z)`` from a lab pulse dump.

    .. warning::

        This is a **hacky stop-gap**. It reverse-engineers two physical
        constants by *assuming the experiment was correctly tuned* -- i.e. that
        every pulse was meant to sit on the integer recoil ladder -- and then
        fitting whatever offsets make that true. It is a self-fulfilling
        calibration, not a measurement. These numbers should instead come from
        **real, independent calibrations**: the probe (AC-Stark) coefficient
        from a dedicated light-shift-vs-intensity measurement, and the
        initial velocity from the measured launch dynamics / time-of-flight.
        A loud :class:`UserWarning` is emitted every time this runs so it can
        never quietly masquerade as a real calibration.



    All frequencies are anchored on the FIRST pulse of the dump, whichever beam
    fired it (matching :func:`build_sequence_from_lab_pulse_dump`), not on the
    first up pulse. "Anchor beam" below means the beam of that first pulse.

    * ``alpha`` (probe-shift coefficient, 1/Hz): the anchor-beam pulses share
      one beam, so after removing the probe shift ``alpha * rabi**2`` their
      effective detunings must differ only by an integer number of recoils
      (assuming the experimental sequence is correct - this is the hacky bit).
      Comparing the two anchor-beam pulses of most-different Rabi frequency
      lets us deduce a value for ``alpha``. The anchor beam must be used
      because its built detunings are independent of ``v0`` (the first-pulse
      anchor absorbs the Doppler shift for its own beam).

    * ``v0`` (initial z-velocity, m/s): the two beams counter-propagate, so a
      nonzero ``v0`` shifts the opposite beam's detunings by ``2 * v0 /
      lambda`` relative to the anchored centre. We pin ``v0`` by **assuming
      the first opposite-beam pulse is resonant** on the transition that
      addresses the freshly velocity-selected atom (4 recoils below the
      anchor pulse in effective-detuning space, for either beam order).

    Parameters match :func:`build_sequence_from_lab_pulse_dump`.

    Returns
    -------
    (float, float)
        ``(probe_shift_alpha, initial_velocity_z)`` ready to feed straight into
        :func:`build_sequence_from_lab_pulse_dump` as
        ``probe_induced_alpha_up = probe_induced_alpha_down = probe_shift_alpha``
        and ``initial_velocity_z``.
    """
    warnings.warn(
        "calibrate_probe_shift_and_velocity_from_dump is a HACKY self-consistent "
        "fit: it assumes every pulse was meant to be on the recoil ladder and "
        "backs out alpha and v0 to force that. These are NOT measurements. "
        "Replace with real light-shift and launch-velocity calibrations.",
        UserWarning,
        stacklevel=2,
    )

    # Build with the probe shift and initial-velocity Doppler switched OFF so the
    # recorded (centre-anchored) detunings are exposed directly. The AOM-sign
    # convention is whatever build_sequence_from_lab_pulse_dump uses -- this
    # calibration is consistent with it by construction.
    bare_sequence_timestamps, bare_sequence = build_sequence_from_lab_pulse_dump(
        is_up=is_up,
        start_times_s=start_times_s,
        durations_s=durations_s,
        opll_hz=opll_hz,
        switch_hz=switch_hz,
        delivery_hz=delivery_hz,
        delivery_setpoint=delivery_setpoint,
        probe_induced_alpha_up=0.0,
        probe_induced_alpha_down=0.0,
        pi_pulse_threshold_s=pi_pulse_threshold_s,
        initial_velocity_z=0.0,
    )

    pulses = [e for e in bare_sequence if isinstance(e, Pulse)]
    if len(pulses) == 0:
        raise ValueError("Lab pulse dump contains no pulses")

    # The build anchors all frequencies on the first pulse, whichever beam
    # fired it. With probe shift and v0 switched off, the built detunings obey
    #
    #   det_i = rung_i * REC + alpha * (rabi_i**2 - rabi_0**2)
    #           + (v0 / lambda) * (k_i - k_0)
    #
    # where rung_i is the integer recoil-ladder position the lab intended.
    # Anchor-beam pulses (k_i == k_0) are therefore independent of v0, and
    # opposite-beam pulses are offset by exactly 2 * k_0 * v0 / lambda.
    anchor_pulse = pulses[0]
    anchor_beam_sign = anchor_pulse.k

    # --- alpha: from two anchor-beam pulses with the largest Rabi**2 separation ---
    anchor_beam_pulses = [p for p in pulses if p.k == anchor_beam_sign]
    pulse_min_rabi = min(anchor_beam_pulses, key=lambda p: p.rabi_frequency)
    pulse_max_rabi = max(anchor_beam_pulses, key=lambda p: p.rabi_frequency)

    if pulse_min_rabi.rabi_frequency == pulse_max_rabi.rabi_frequency:
        raise ValueError(
            "All anchor-beam (first-pulse beam) pulses share the same Rabi "
            "frequency; cannot separate the probe shift from the recoil ladder."
        )

    f_pulse_difference = pulse_max_rabi.detuning_hz - pulse_min_rabi.detuning_hz

    # The free-fall Doppler shift was already removed by
    # build_sequence_from_lab_pulse_dump, and v0 drops out for anchor-beam
    # pulses, so this difference is only the probe-induced Stark shift plus an
    # integer number of recoils. This pins down the Stark contribution, on the
    # assumption that the probe-induced shift is less than half a recoil.
    ladder_separation_hz = (
        round(f_pulse_difference / sim.RECOIL_FREQUENCY_HZ)
        * sim.RECOIL_FREQUENCY_HZ
    )
    residual_probe_shift_hz = f_pulse_difference - ladder_separation_hz

    probe_shift_alpha = residual_probe_shift_hz / (
        pulse_max_rabi.rabi_frequency**2 - pulse_min_rabi.rabi_frequency**2
    )

    # --- v0: anchor on the FIRST opposite-beam pulse being resonant ---

    # The build already assumes the first pulse is a velocity-selection pi
    # pulse, resonant on m=0 -> m=k_0 (one recoil above the bare transition).
    #
    # The first pulse of the OPPOSITE beam then addresses the freshly shelved
    # atom (excited, m=k_0), whose ground-state partner for that beam is
    # m_g = 2*k_0: its resonant effective detuning is 3 recoils BELOW the bare
    # transition, for either beam order. The anchor and first-opposite pulse
    # therefore sit exactly 4 recoils apart in effective-detuning space, and
    # any residual is the 2 * k_0 * v0 / lambda Doppler offset between the
    # counter-propagating beams:
    #
    #   v0 = (k_0 * lambda / 2) * (det_0 - det_j - 4 * REC - (shift_0 - shift_j))
    #
    # Gravity needs no separate term here: the built detunings already include
    # the free-fall Doppler ramp consistently for both beams.
    first_opposite = next((p for p in pulses if p.k == -anchor_beam_sign), None)

    if first_opposite is None:
        # The initial velocity is irrelevant
        initial_velocity_z = 0.0
    else:
        anchor_probe_shift_hz = probe_shift_alpha * anchor_pulse.rabi_frequency**2
        opposite_probe_shift_hz = (
            probe_shift_alpha * first_opposite.rabi_frequency**2
        )

        initial_velocity_z = (
            anchor_beam_sign
            * 0.5
            * sim.TRANSITION_WAVELENGTH
            * (
                anchor_pulse.detuning_hz
                - first_opposite.detuning_hz
                - 4 * sim.RECOIL_FREQUENCY_HZ
                - (anchor_probe_shift_hz - opposite_probe_shift_hz)
            )
        )

    return probe_shift_alpha, initial_velocity_z


def compute_spacetime_trajectory(
    sequence, *, flip_threshold=0.75, max_branches=None, plot=False
):
    """Infer intended spacetime trajectory by simulating an ideal atom.

    Walks the sequence with a stationary, on-axis atom in the ground state and
    decides for each pulse whether each cloud flips, drifts, or splits based on
    the Rabi transition probability.

    Parameters
    ----------
    sequence : list[Pulse | Clearout | Freefall]
    flip_threshold : float
        Probability >= this → flip; <= 1-this → no-op; between → split.
    max_branches : int | None
        Maximum allowed number of live branches. ``None`` disables the limit.
    plot : bool
        If True, produce a spacetime/momentum figure.

    Returns
    -------
    tuple
        (clouds, clearout_times) where clouds is a list of Cloud objects.
    """

    @dataclass
    class Cloud:
        times: list
        z: list
        m: list
        is_ground: list
        labels: list
        alive: bool = True
        fork_index: int = 0
        color_index: int = 0

        @property
        def v(self):
            return self.m[-1] * sim.RECOIL_VELOCITY

        def _fork(self):
            return Cloud(
                times=list(self.times),
                z=list(self.z),
                m=list(self.m),
                is_ground=list(self.is_ground),
                labels=list(self.labels),
                fork_index=self.fork_index,
                color_index=self.color_index,
            )

    previous_event = None
    for event in sequence:
        if not isinstance(event, (Pulse, Clearout, Freefall)):
            raise TypeError(f"Unsupported sequence event type: {type(event)!r}")
        if (
            isinstance(event, Pulse)
            and event.simultaneous_with_previous
            and not isinstance(previous_event, Pulse)
        ):
            raise ValueError(
                "A simultaneous_with_previous pulse must directly follow "
                "another Pulse in the sequence"
            )
        previous_event = event
    if max_branches is not None and max_branches < 1:
        raise ValueError("max_branches must be positive or None")

    def enforce_max_branches():
        if max_branches is None:
            return
        live_branch_count = sum(cloud.alive for cloud in clouds)
        if live_branch_count > max_branches:
            if plot:
                _plot_spacetime(sequence, clouds, clearout_times)
            raise RuntimeError(
                "compute_spacetime_trajectory exceeded max_branches: "
                f"{live_branch_count} live branches > {max_branches}"
            )

    t = 0.0
    clouds = [Cloud(times=[0.0], z=[0.0], m=[0], is_ground=[True], labels=[""])]
    clearout_times = []
    next_color_index = 1

    for event in sequence:
        # A simultaneous pulse fires at the same time as its predecessor: the
        # sequence clock (and hence cloud positions) must not advance again.
        dt = (
            0.0
            if isinstance(event, Pulse) and event.simultaneous_with_previous
            else event.duration
        )

        if isinstance(event, (Freefall, Clearout)):
            t += dt
            for cloud in clouds:
                if cloud.alive:
                    cloud.times.append(t)
                    cloud.z.append(cloud.z[-1] + cloud.v * dt)
                    cloud.m.append(cloud.m[-1])
                    cloud.is_ground.append(cloud.is_ground[-1])
                    cloud.labels.append(event.label)
            if isinstance(event, Clearout):
                clearout_times.append(t)
                for cloud in clouds:
                    if cloud.alive and cloud.is_ground[-1]:
                        cloud.alive = False
            continue

        # Pulse
        t += dt
        new_clouds = []
        for cloud in clouds:
            if not cloud.alive:
                new_clouds.append(cloud)
                continue
            p = _transition_probability(cloud.m[-1], cloud.is_ground[-1], event)
            if p >= flip_threshold:
                dm = event.k if cloud.is_ground[-1] else -event.k
                new_m = cloud.m[-1] + dm
                new_z = cloud.z[-1] + new_m * sim.RECOIL_VELOCITY * dt
                cloud.times.append(t)
                cloud.z.append(new_z)
                cloud.m.append(new_m)
                cloud.is_ground.append(not cloud.is_ground[-1])
                cloud.labels.append(event.label)
                new_clouds.append(cloud)
            elif p <= 1.0 - flip_threshold:
                cloud.times.append(t)
                cloud.z.append(cloud.z[-1] + cloud.v * dt)
                cloud.m.append(cloud.m[-1])
                cloud.is_ground.append(cloud.is_ground[-1])
                cloud.labels.append(event.label)
                new_clouds.append(cloud)
            else:
                drifter = cloud._fork()
                flipper = cloud._fork()
                flipper.fork_index = len(cloud.times)
                flipper.color_index = next_color_index
                next_color_index += 1
                drifter.times.append(t)
                drifter.z.append(drifter.z[-1] + drifter.v * dt)
                drifter.m.append(drifter.m[-1])
                drifter.is_ground.append(drifter.is_ground[-1])
                drifter.labels.append(event.label)
                dm = event.k if flipper.is_ground[-1] else -event.k
                new_m = flipper.m[-1] + dm
                new_z = flipper.z[-1] + new_m * sim.RECOIL_VELOCITY * dt
                flipper.times.append(t)
                flipper.z.append(new_z)
                flipper.m.append(new_m)
                flipper.is_ground.append(not flipper.is_ground[-1])
                flipper.labels.append(event.label)
                new_clouds.extend([drifter, flipper])
            clouds = new_clouds
            enforce_max_branches()

    if plot:
        _plot_spacetime(sequence, clouds, clearout_times)

    return clouds, np.asarray(clearout_times)


def _plot_spacetime(sequence, clouds, clearout_times):
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10.colors
    addressed_bar_padding = 0.05
    addressed_bar_alpha = 0.24
    fig, (ax_z, ax_m) = plt.subplots(
        2, 1, figsize=(13, 15), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    def build_plot_trace(cloud):
        # Build the complete midpoint-convention trace from t=0 so that all
        # positions are self-consistent regardless of where the cloud forked.
        # Then slice to only show from one event before the fork.
        times = [cloud.times[0]]
        positions = [cloud.z[0]]
        momentum_times = [cloud.times[0]]
        momentum = [cloud.m[0]]
        ground = [cloud.is_ground[0]]
        m_ground = [cloud.is_ground[0]]

        current_time = cloud.times[0]
        current_position = cloud.z[0]
        current_m = cloud.m[0]
        current_ground = cloud.is_ground[0]

        for i in range(len(cloud.times) - 1):
            event = sequence[i]
            # Simultaneous pulses do not advance the clock (see the main loop)
            dt = (
                0.0
                if isinstance(event, Pulse) and event.simultaneous_with_previous
                else event.duration
            )
            event_end_time = current_time + dt

            if isinstance(event, Pulse):
                mid_time = current_time + dt / 2
                mid_position = (
                    current_position + current_m * sim.RECOIL_VELOCITY * dt / 2
                )
                next_m = cloud.m[i + 1]
                next_ground = cloud.is_ground[i + 1]
                end_position = mid_position + next_m * sim.RECOIL_VELOCITY * dt / 2

                times.extend([mid_time, event_end_time])
                positions.extend([mid_position, end_position])
                momentum_times.extend([mid_time, mid_time, event_end_time])
                momentum.extend([current_m, next_m, next_m])
                ground.extend([current_ground, next_ground])
                m_ground.extend([current_ground, next_ground, next_ground])
            else:
                next_m = cloud.m[i + 1]
                next_ground = cloud.is_ground[i + 1]
                end_position = current_position + current_m * sim.RECOIL_VELOCITY * dt

                times.append(event_end_time)
                positions.append(end_position)
                momentum_times.append(event_end_time)
                momentum.append(next_m)
                ground.append(next_ground)
                m_ground.append(next_ground)

            current_time = event_end_time
            current_position = positions[-1]
            current_m = momentum[-1]
            current_ground = ground[-1]

        # Slice to start one event before the fork so forked clouds are only
        # plotted from their branch point. Each event contributes 2 entries to
        # the z/ground trace (Pulse) or 1 (everything else), and 3 or 1 to the
        # momentum trace.
        fi = max(0, cloud.fork_index - 1)
        z_start = sum(2 if isinstance(sequence[i], Pulse) else 1 for i in range(fi))
        m_start = sum(3 if isinstance(sequence[i], Pulse) else 1 for i in range(fi))

        return (
            np.asarray(times[z_start:]),
            np.asarray(positions[z_start:]),
            np.asarray(momentum_times[m_start:]),
            np.asarray(momentum[m_start:]),
            np.asarray(ground[z_start:]),
            np.asarray(m_ground[m_start:]),
        )

    for cloud in clouds:
        color = colors[cloud.color_index % len(colors)]
        times_us, z_mm, m_times_us, m_arr, is_ground, m_is_ground = build_plot_trace(
            cloud
        )
        label_added = False
        for j in range(len(times_us) - 1):
            ls = "-" if is_ground[j + 1] else ":"
            lbl = f"cloud {cloud.color_index}" if not label_added else None
            ax_z.plot(
                times_us[j : j + 2] * 1e6,
                z_mm[j : j + 2] * 1e3,
                ls,
                color=color,
                lw=1.5,
                label=lbl,
            )
            label_added = True
        m_label_added = False
        for j in range(len(m_times_us) - 1):
            ls = "-" if m_is_ground[j + 1] else ":"
            lbl = f"cloud {cloud.color_index}" if not m_label_added else None
            ax_m.plot(
                m_times_us[j : j + 2] * 1e6,
                m_arr[j : j + 2],
                ls,
                color=color,
                lw=1.5,
                label=lbl
            )
            m_label_added = True
        if cloud.alive:
            pass
            # ax_z.plot(times_us * 1e6, z_mm * 1e3, "o", color=color, ms=3)
            # ax_m.plot(m_times_us * 1e6, m_arr, "o", color=color, ms=3)
        else:
            ax_z.plot(times_us[:-1] * 1e6, z_mm[:-1] * 1e3, "o", color=color, ms=3)
            ax_z.plot(
                times_us[-1:] * 1e6, z_mm[-1:] * 1e3, "x", color=color, ms=5, mew=1.5
            )
            ax_m.plot(m_times_us[:-1] * 1e6, m_arr[:-1], "o", color=color, ms=3)
            ax_m.plot(
                m_times_us[-1:] * 1e6, m_arr[-1:], "x", color=color, ms=5, mew=1.5
            )

    for t_co in clearout_times:
        ax_z.axvline(t_co * 1e6, color="tab:green", lw=4, alpha=0.3, linestyle="-")
    if len(clearout_times) > 0:
        ax_z.plot(
            [],
            [],
            color="tab:green",
            linestyle="-",
            alpha=0.3,
            label=f"clearout ({len(clearout_times)} positions)",
        )

    # Pulse shading
    pulse_fill_added = {+1: False, -1: False}
    pulse_colors = {+1: "tab:blue", -1: "tab:red"}
    pulse_labels = {+1: "k=+1 pulse", -1: "k=−1 pulse"}
    pulse_edge_alpha = 0.45
    pulse_edge_lw = 0.6
    t_event = 0.0
    last_pulse_start = 0.0
    for event in sequence:
        if isinstance(event, Pulse):
            # A simultaneous pulse shares its predecessor's time span
            if event.simultaneous_with_previous:
                t_start = last_pulse_start
            else:
                t_start = t_event
                last_pulse_start = t_event
            t_start_us = t_start * 1e6
            t_width_us = event.duration * 1e6
            t_end_us = (t_start + event.duration) * 1e6
            m_ground, m_excited = _addressed_momentum_classes(event)
            m_low = min(m_ground, m_excited)
            m_high = max(m_ground, m_excited)
            lbl = pulse_labels[event.k] if not pulse_fill_added[event.k] else None
            for ax in (ax_z, ax_m):
                ax.axvspan(
                    t_start_us,
                    t_end_us,
                    color=pulse_colors[event.k],
                    alpha=0.12,
                    lw=0,
                    label=lbl,
                )
                ax.axvline(
                    t_start_us,
                    color=pulse_colors[event.k],
                    lw=pulse_edge_lw,
                    alpha=pulse_edge_alpha,
                )
                ax.axvline(
                    t_end_us,
                    color=pulse_colors[event.k],
                    lw=pulse_edge_lw,
                    alpha=pulse_edge_alpha,
                )
            ax_m.broken_barh(
                [(t_start_us, t_width_us)],
                (
                    m_low - addressed_bar_padding,
                    (m_high - m_low) + 2 * addressed_bar_padding,
                ),
                facecolors=pulse_colors[event.k],
                edgecolors=pulse_colors[event.k],
                linewidth=pulse_edge_lw,
                alpha=addressed_bar_alpha,
                zorder=1.5,
            )
            lbl = None  # only add to one axis
            pulse_fill_added[event.k] = True
        t_event += (
            0.0
            if isinstance(event, Pulse) and event.simultaneous_with_previous
            else event.duration
        )

    ax_z.plot([], [], "-", color="gray", lw=1.5, label="|g> (solid)")
    ax_z.plot([], [], ":", color="gray", lw=1.5, label="|e> (dotted)")
    ax_m.plot([], [], "-", color="gray", lw=1.5, label="|g> (solid)")
    ax_m.plot([], [], ":", color="gray", lw=1.5, label="|e> (dotted)")
    ax_z.set_ylabel("z position (mm)")
    ax_z.set_title("LMT spacetime diagram")
    ax_z.legend(loc="upper left")
    ax_z.grid(True, alpha=0.3)

    all_m = [m for cloud in clouds for m in cloud.m]
    ax_m.axhline(0, color="k", lw=0.3, alpha=0.3)
    ax_m.set_xlabel("time (us)")
    ax_m.set_ylabel(r"$v_z$ ($v_\mathrm{recoil}$)")
    ax_m.set_title(
        f"v_recoil = {sim.RECOIL_VELOCITY * 1e3:.2f} mm/s; "
        + f"|m|_max = {int(np.abs(all_m).max()) if all_m else 0}"
    )
    ax_m.grid(True, alpha=0.3)


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
