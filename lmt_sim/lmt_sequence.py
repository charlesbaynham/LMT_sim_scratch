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


def compute_spacetime_trajectory(sequence, plot=False):
    """Compute deterministic TOP/BOTTOM cloud trajectory for a sequence.

    Pulse labels must include "-TOP" or "-BOT" or "-BOTH" to indicate which cloud they are
    intended to address.

    Parameters
    ----------
    sequence : list[Pulse | Clearout | Freefall]
        Sequence of events to track.
    plot : bool
        If True, produce a spacetime/momentum figure.

    Returns
    -------
    tuple
        (times, z_top, z_bot, v_top, v_bot, m_top, m_bot,
         s_top, s_bot, labels, clearout_times)
    """

    for event in sequence:
        if not isinstance(event, (Pulse, Clearout, Freefall)):
            raise TypeError(f"Unsupported sequence event type: {type(event)!r}")

    v_recoil = sim.RECOIL_VELOCITY
    top = {"z": 0.0, "v": 0.0, "m": 0, "state": "g"}
    bot = dict(top)
    bottom_exists = False
    clearout_times = []

    def _flip(cloud, k_sign, t_pulse):
        if cloud["state"] == "g":
            dm = +k_sign
            new_state = "e"
        else:
            dm = -k_sign
            new_state = "g"
        m_new = cloud["m"] + dm
        v_new = m_new * v_recoil
        cloud["z"] = cloud["z"] + 0.5 * (cloud["v"] + v_new) * t_pulse
        cloud["v"] = v_new
        cloud["m"] = m_new
        cloud["state"] = new_state

    def _drift(cloud, t_pulse):
        cloud["z"] = cloud["z"] + cloud["v"] * t_pulse

    times = [0.0]
    z_top_list, z_bot_list = [top["z"]], [bot["z"]]
    v_top_list, v_bot_list = [top["v"]], [bot["v"]]
    m_top_list, m_bot_list = [top["m"]], [bot["m"]]
    s_top_list, s_bot_list = [top["state"]], [bot["state"]]
    labels = [""]

    t = 0.0

    for event in sequence:
        if isinstance(event, (Clearout, Freefall)):
            if isinstance(event, Clearout):
                clearout_times.append(t)
            if event.duration > 0.0:
                _drift(top, event.duration)
                if bottom_exists:
                    _drift(bot, event.duration)
            t += event.duration
            times.append(t)
            z_top_list.append(top["z"])
            z_bot_list.append(bot["z"])
            v_top_list.append(top["v"])
            v_bot_list.append(bot["v"])
            m_top_list.append(top["m"])
            m_bot_list.append(bot["m"])
            s_top_list.append(top["state"])
            s_bot_list.append(bot["state"])
            labels.append(event.label)
            continue

        k_sign = event.k
        t_pulse = event.duration
        label = event.label
        addresses_top = "-TOP" in label
        addresses_bot = "-BOT" in label

        if label == "vel sel (UP-TOP)":
            _flip(top, +1, t_pulse)
            bot.update({k: top[k] for k in ("z", "v", "m", "state")})
        elif (
            (not bottom_exists)
            and addresses_top
            and ("BS1" in label)
            and not addresses_bot
        ):
            _drift(top, t_pulse)
            _flip(bot, k_sign, t_pulse)
            bottom_exists = True
        else:
            if addresses_top and addresses_bot:
                # Both clouds addressed (e.g., "-BOTH" label)
                _flip(top, k_sign, t_pulse)
                if bottom_exists:
                    _flip(bot, k_sign, t_pulse)
            elif addresses_top ^ addresses_bot:
                # Exactly one cloud addressed
                target = top if addresses_top else bot
                other = bot if addresses_top else top
                _flip(target, k_sign, t_pulse)
                if bottom_exists:
                    _drift(other, t_pulse)
            else:
                raise ValueError(
                    "Pulse label must address exactly one cloud with '-TOP', '-BOT', or both with '-BOTH': "
                    + label
                )

        t += t_pulse
        times.append(t)
        z_top_list.append(top["z"])
        z_bot_list.append(bot["z"])
        v_top_list.append(top["v"])
        v_bot_list.append(bot["v"])
        m_top_list.append(top["m"])
        m_bot_list.append(bot["m"])
        s_top_list.append(top["state"])
        s_bot_list.append(bot["state"])
        labels.append(label)

    outputs = (
        np.asarray(times),
        np.asarray(z_top_list),
        np.asarray(z_bot_list),
        np.asarray(v_top_list),
        np.asarray(v_bot_list),
        np.asarray(m_top_list),
        np.asarray(m_bot_list),
        s_top_list,
        s_bot_list,
        labels,
        np.asarray(clearout_times),
    )

    if plot:
        import matplotlib.pyplot as plt

        (
            times_a,
            z_top_a,
            z_bot_a,
            _v_top_a,
            _v_bot_a,
            m_top_a,
            m_bot_a,
            s_top_a,
            s_bot_a,
            _labels_a,
            clearout_times_a,
        ) = outputs

        def _plot_state_trajectory(ax, times_us, z_mm, states, color, legend_label):
            label_added = False
            for i in range(len(times_us) - 1):
                ls = ":" if states[i] == "e" else "-"
                lbl = legend_label if not label_added else None
                ax.plot(
                    times_us[i : i + 2],
                    z_mm[i : i + 2],
                    ls,
                    color=color,
                    lw=1.5,
                    label=lbl,
                )
                label_added = True
            ax.plot(times_us, z_mm, "o", color=color, ms=3)

        fig, (ax_z, ax_v) = plt.subplots(
            2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        _plot_state_trajectory(
            ax_z,
            times_a * 1e6,
            z_top_a * 1e3,
            s_top_a,
            "tab:red",
            "TOP",
        )
        _plot_state_trajectory(
            ax_z,
            times_a * 1e6,
            z_bot_a * 1e3,
            s_bot_a,
            "tab:blue",
            "BOTTOM",
        )

        for t_clearout in clearout_times_a:
            ax_z.axvline(
                t_clearout * 1e6,
                color="tab:green",
                lw=0.6,
                alpha=0.6,
                linestyle="--",
            )
        if len(clearout_times_a) > 0:
            ax_z.plot(
                [],
                [],
                color="tab:green",
                linestyle="--",
                alpha=0.6,
                label=f"clearout ({len(clearout_times_a)} positions)",
            )

        ax_z.plot([], [], "-", color="gray", lw=1.5, label="|g> (solid)")
        ax_z.plot([], [], ":", color="gray", lw=1.5, label="|e> (dotted)")
        ax_z.set_ylabel("z position (mm)")
        ax_z.set_title("LMT spacetime diagram")
        ax_z.legend(loc="upper left")
        ax_z.grid(True, alpha=0.3)

        ax_v.plot(times_a * 1e6, m_top_a, "-o", color="tab:red", ms=3, label="TOP")
        ax_v.plot(times_a * 1e6, m_bot_a, "-o", color="tab:blue", ms=3, label="BOTTOM")
        ax_v.axhline(0, color="k", lw=0.3, alpha=0.3)
        ax_v.set_xlabel("time (us)")
        ax_v.set_ylabel(r"$v_z$ ($v_\mathrm{recoil}$)")
        ax_v.set_title(
            "v_recoil = "
            + f"{v_recoil * 1e3:.2f} mm/s; "
            + f"|m_TOP|_max = {int(np.abs(m_top_a).max())}, "
            + f"|m_BOTTOM|_max = {int(np.abs(m_bot_a).max())}"
        )
        ax_v.grid(True, alpha=0.3)

    return outputs


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
