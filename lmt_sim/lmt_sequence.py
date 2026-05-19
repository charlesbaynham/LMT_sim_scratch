from dataclasses import dataclass
import logging

import numpy as np
import lmt_sim.lmt_simulation as sim
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


@dataclass
class Cloud:
    """Trajectory of one cloud branch through a pulse sequence."""

    times: list
    z: list
    m: list
    is_ground: list
    labels: list
    alive: bool = True

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
        )


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
        logger.warning("No pulses in sequence, defaulting to zero detuning")
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


def run_pulse_sequence_in_borde_representation(
    state,
    pulse_sequence,
    initial_velocity_z=0.0,
    rng=None,
):
    """Run a pulse sequence while staying in the Borde representation."""

    for event in pulse_sequence:
        if not isinstance(event, (Pulse, Clearout, Freefall)):
            raise TypeError(f"Unsupported sequence event type: {type(event)!r}")

    detunings_hz = [
        pulse.detuning_hz for pulse in pulse_sequence if isinstance(pulse, Pulse)
    ]
    current_detuning_hz = detunings_hz[0] if len(detunings_hz) > 0 else 0.0
    current_time = 0.0

    # Process the sequence event by event
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
            state = sim.pulse_interaction_in_borde_representation(
                state,
                pulse_detuning=event.detuning_hz,
                t_pulse=event.duration,
                pulse_rabi_freq=event.rabi_frequency,
                pulse_phase=event.phi,
                k_sign=event.k,
                k_wavevector=sim.K_WAVEVECTOR,
                vz=initial_velocity_z,
            )

        elif isinstance(event, Clearout):
            # If it's a clearout, do the projection and abort if the atom is
            # cleared out. N.B. This does not do balliastic propegation - we
            # must do it later
            result = sim.do_clearout(state, rng=rng)
            if result is None:
                return None
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

    return state, current_detuning_hz, current_time


def calculate_excited_fraction_for_pulse_sequence(
    pulse_sequence,
    initial_velocity_z=0.0,
):
    """Calculate final excited-state fraction for a sequence without clearout."""
    import lmt_sim.lmt_simulation as sim

    if any(isinstance(event, Clearout) for event in pulse_sequence):
        raise ValueError(
            "calculate_excited_fraction_for_pulse_sequence does not support Clearout events"
        )

    state = sim.make_atom_states(initial_velocity_z=initial_velocity_z)

    result = run_pulse_sequence_in_lab_frame(
        state,
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
    )

    if result is None:
        return None

    state, *_ = result
    ground_prob, excited_prob = sim.calculate_ground_and_excited_probabilities(state)
    return excited_prob / (ground_prob + excited_prob)


def _transition_probability(m, is_ground, pulse):
    """Rabi transition probability for a stationary on-axis atom at momentum class m."""
    k = pulse.k
    m_ground = m if is_ground else m - k
    omega_ab = np.pi * pulse.rabi_frequency
    # delta_rec = hbar*K²/(2M) = K*RECOIL_VELOCITY/2
    delta_rec = sim.K_WAVEVECTOR * sim.RECOIL_VELOCITY / 2
    Omega_3 = 2 * np.pi * pulse.detuning_hz - (2 * m_ground + k) * delta_rec
    Omega = np.sqrt(Omega_3**2 + 4 * omega_ab**2)
    return float((2 * omega_ab / Omega) ** 2 * np.sin(Omega * pulse.duration / 2) ** 2)


def compute_spacetime_trajectory(sequence, *, flip_threshold=0.75, plot=False):
    """Infer intended spacetime trajectory by simulating an ideal atom.

    Walks the sequence with a stationary, on-axis atom in the ground state and
    decides for each pulse whether each cloud flips, drifts, or splits based on
    the Rabi transition probability.

    Parameters
    ----------
    sequence : list[Pulse | Clearout | Freefall]
    flip_threshold : float
        Probability >= this → flip; <= 1-this → no-op; between → split.
    plot : bool
        If True, produce a spacetime/momentum figure.

    Returns
    -------
    tuple
        (clouds, clearout_times) where clouds is a list of Cloud objects.
    """
    for event in sequence:
        if not isinstance(event, (Pulse, Clearout, Freefall)):
            raise TypeError(f"Unsupported sequence event type: {type(event)!r}")

    t = 0.0
    clouds = [Cloud(times=[0.0], z=[0.0], m=[0], is_ground=[True], labels=[""])]
    clearout_times = []

    for event in sequence:
        dt = event.duration

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
                new_z = cloud.z[-1] + 0.5 * (cloud.v + new_m * sim.RECOIL_VELOCITY) * dt
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
                drifter.times.append(t)
                drifter.z.append(drifter.z[-1] + drifter.v * dt)
                drifter.m.append(drifter.m[-1])
                drifter.is_ground.append(drifter.is_ground[-1])
                drifter.labels.append(event.label)
                dm = event.k if flipper.is_ground[-1] else -event.k
                new_m = flipper.m[-1] + dm
                new_z = flipper.z[-1] + 0.5 * (flipper.v + new_m * sim.RECOIL_VELOCITY) * dt
                flipper.times.append(t)
                flipper.z.append(new_z)
                flipper.m.append(new_m)
                flipper.is_ground.append(not flipper.is_ground[-1])
                flipper.labels.append(event.label)
                new_clouds.extend([drifter, flipper])
        clouds = new_clouds

    if plot:
        _plot_spacetime(clouds, clearout_times)

    return clouds, np.asarray(clearout_times)


def _plot_spacetime(clouds, clearout_times):
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10.colors
    fig, (ax_z, ax_m) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for i, cloud in enumerate(clouds):
        color = colors[i % len(colors)]
        times_us = np.asarray(cloud.times) * 1e6
        z_mm = np.asarray(cloud.z) * 1e3
        m_arr = np.asarray(cloud.m)
        label_added = False
        for j in range(len(times_us) - 1):
            ls = "-" if cloud.is_ground[j] else ":"
            lbl = f"cloud {i}" if not label_added else None
            ax_z.plot(times_us[j : j + 2], z_mm[j : j + 2], ls, color=color, lw=1.5, label=lbl)
            label_added = True
        ax_z.plot(times_us, z_mm, "o", color=color, ms=3)
        ax_m.plot(times_us, m_arr, "-o", color=color, ms=3, label=f"cloud {i}")

    for t_co in clearout_times:
        ax_z.axvline(t_co * 1e6, color="tab:green", lw=0.6, alpha=0.6, linestyle="--")
    if len(clearout_times) > 0:
        ax_z.plot([], [], color="tab:green", linestyle="--", alpha=0.6,
                  label=f"clearout ({len(clearout_times)} positions)")

    ax_z.plot([], [], "-", color="gray", lw=1.5, label="|g> (solid)")
    ax_z.plot([], [], ":", color="gray", lw=1.5, label="|e> (dotted)")
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
