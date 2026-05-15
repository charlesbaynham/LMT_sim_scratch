"""LMT-interferometer pulse sequence as run on the real experiment.

Mirrors ``do_clock_interferometry`` from ``LMTInterferometryMixin`` in
``icl_experiments/repository/lib/experiment_templates/mixins/LMT_launch_mixins.py``
plus the preceding velocity-selection pulse from
``ClockShelvingAndClearoutBase.clock_shelving``.

Conventions
-----------
The atom enters the sequence in ``|g, 0⟩`` (the simulator's initial state).  In
the simulator's frame we set ``v_0 = 0`` and track only the m relative to the
zero-momentum class.  After velocity selection the population is in
``|e, +1⟩``; the rest of the sequence walks the two arms across momentum
classes through ``m ≈ +8``.

The launch (``LMTLaunchMixin``) is *not* modelled here.  In the real experiment
a launch already accelerated the atoms by ``N_launch`` recoils before the
sequence below begins; the OPLL offsets in the experiment are programmed
relative to that baseline (this is why ``N_launch = 16`` is hardcoded in
``do_clock_interferometry``).  Since the simulator is in the atom's rest frame,
the on-resonance condition for each pulse is just
``Δ = (4*m_g + k) * δ_recoil`` independent of any launch offset.

Clearouts are represented by a separate :class:`Clearout` marker mixed into the
pulse list.  A consumer can iterate the list and dispatch to
``pulse_interaction_in_borde_representation`` or ``do_clearout`` accordingly.
"""

from dataclasses import dataclass

import numpy as np

from lmt_simulation import Pulse, RECOIL_FREQUENCY_HZ


@dataclass(frozen=True)
class Clearout:
    """A fluorescence clearout: discards ground-state amplitude, renormalises."""

    time: float
    duration: float
    label: str


# Experiment defaults, copied verbatim from icl_experiments/repository/lib/constants.py
CLOCK_SHELVING_PULSE_TIME = 380e-6
SHELVING_PULSE_CLEAROUT_DURATION = 2200e-6
LMT_PULSE_CLEAROUT_DURATION = 50e-6
DOWN_CLOCK_BEAM_PI_TIME = 68e-6
CLOCK_PI_TIME = 55e-6
LMT_SELECTIVE_PI_TIME = 95e-6
DELAY_BETWEEN_INTERFEROMETRY_PULSES = 50e-6

# Timing structure inside the ARTIQ kernel functions.  These reproduce the
# implicit gaps from `fire_lmt_pulse`, `lmt_series`, `do_selective_lmt_pulse`,
# and `do_clock_interferometry`.
LMT_PRE_DELAY = (
    50e-6  # `t_start = now_mu() + 50us` at the top of each lmt_series iteration
)
LMT_POST_DELAY = 10e-6  # `delay(10e-6)` at the end of fire_lmt_pulse
SELECTIVE_PRE_DELAY = 50e-6  # `t_pulse = now_mu() + 50us` inside do_selective_lmt_pulse
POST_BS1_DELAY = 100e-6  # `delay(100e-6)` after BS1
POST_MIRROR_DELAY = 1e-6  # `delay(1e-6)` after the mirror pulse
PRE_BS2_DELAY = 100e-6  # `t_start_last_pulse_mu = now_mu() + 100us` before BS2

# Gap between the end of velocity-selection clearout and BS1.  In the real
# experiment this is a long chain of post-dipole-trap hooks; left as a
# parameter so consumers can tune it for their model of the atom's free fall.
DEFAULT_VS_TO_BS1_GAP = 1e-3


def build_lmt_real_sequence(
    N: int = 7,
    phase_step: float = 0.0,
    delay_between_interferometry_pulses: float = DELAY_BETWEEN_INTERFEROMETRY_PULSES,
    vs_to_bs1_gap: float = DEFAULT_VS_TO_BS1_GAP,
):
    """Build the LMT pulse sequence as run on the real experiment.

    Parameters
    ----------
    N : int
        Total number of LMT pulses per side, matching ``lmt_pulses_number``.
        Must be >= 3 to exercise the full LMT structure (N=7 is the
        experiment default).
    phase_step : float
        Phase applied to the mirror pulse (and 4× to BS2), in radians.  All
        other pulses use phase 0.  Default 0.
    delay_between_interferometry_pulses : float
        Dark time between forward and reverse LMT on each arm.  Default 50us.
    vs_to_bs1_gap : float
        Free-evolution time between the end of the velocity-selection
        clearout and BS1.  Not modelled precisely; defaults to 1ms.

    Returns
    -------
    list[Pulse | Clearout]
        Time-ordered list of laser pulses and clearouts that, applied in
        order, reproduces the experimental sequence.
    """

    if N < 3:
        raise ValueError("N must be >= 3 to run the full LMT sequence")

    sequence: list = []
    rabi_vs = 1.0 / (2 * CLOCK_SHELVING_PULSE_TIME)
    rabi_down = 1.0 / (2 * DOWN_CLOCK_BEAM_PI_TIME)
    rabi_up_high = 1.0 / (2 * CLOCK_PI_TIME)
    rabi_selective = 1.0 / (2 * LMT_SELECTIVE_PI_TIME)

    # --- VELOCITY SELECTION ---
    t = 0.0
    sequence.append(
        Pulse(
            time=t,
            k=+1,
            detuning_hz=(4 * 0 + 1) * RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="velocity_selection",
            rabi_frequency=rabi_vs,
            pulse_area=np.pi,
        )
    )
    t += sequence[-1].duration
    sequence.append(
        Clearout(
            time=t,
            duration=SHELVING_PULSE_CLEAROUT_DURATION,
            label="velocity_selection_clearout",
        )
    )
    t += SHELVING_PULSE_CLEAROUT_DURATION
    t += vs_to_bs1_gap

    # --- BS1: DOWN π/2 ---
    sequence.append(
        Pulse(
            time=t,
            k=-1,
            detuning_hz=(4 * 2 - 1) * RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="BS1",
            rabi_frequency=rabi_down,
            pulse_area=np.pi / 2,
        )
    )
    t += sequence[-1].duration
    t += POST_BS1_DELAY

    # --- First selective UP π (upper arm) ---
    t += SELECTIVE_PRE_DELAY
    sequence.append(
        Pulse(
            time=t,
            k=+1,
            detuning_hz=(4 * 2 + 1) * RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="first_selective_upper",
            rabi_frequency=rabi_selective,
            pulse_area=np.pi,
        )
    )
    t += sequence[-1].duration
    sequence.append(
        Clearout(
            time=t,
            duration=LMT_PULSE_CLEAROUT_DURATION,
            label="clearout_after_first_selective_upper",
        )
    )
    t += LMT_PULSE_CLEAROUT_DURATION

    # --- Forward LMT on upper arm: N-2 pulses ---
    m = 3
    is_excited = True
    for i in range(N - 2):
        t += LMT_PRE_DELAY
        if i % 2 == 0:
            # DOWN: |e, m⟩ → |g, m+1⟩
            k = -1
            rabi = rabi_down
            m_g = m + 1
            phi = 0.0
            label = f"upper_fw_down_{i}"
            is_excited = False
            m += 1
        else:
            # UP: |g, m⟩ → |e, m+1⟩
            k = +1
            rabi = rabi_up_high
            m_g = m
            phi = 0.0
            label = f"upper_fw_up_{i}"
            is_excited = True
            m += 1
        sequence.append(
            Pulse(
                time=t,
                k=k,
                detuning_hz=(4 * m_g + k) * RECOIL_FREQUENCY_HZ,
                phi=phi,
                label=label,
                rabi_frequency=rabi,
                pulse_area=np.pi,
            )
        )
        t += sequence[-1].duration
        t += LMT_POST_DELAY
    m_after_upper_fw = m

    # --- Dark time upper arm ---
    t += delay_between_interferometry_pulses

    # --- Reverse LMT on upper arm: N-2 pulses ---
    m = m_after_upper_fw
    is_excited = False
    for i in range(N - 2):
        t += LMT_PRE_DELAY
        if i % 2 == 0:
            # DOWN: |g, m⟩ → |e, m-1⟩
            k = -1
            rabi = rabi_down
            m_g = m
            phi = 0.0
            label = f"upper_rv_down_{i}"
            is_excited = True
            m -= 1
        else:
            # UP: |e, m⟩ → |g, m-1⟩
            k = +1
            rabi = rabi_up_high
            m_g = m - 1
            phi = 0.0
            label = f"upper_rv_up_{i}"
            is_excited = False
            m -= 1
        sequence.append(
            Pulse(
                time=t,
                k=k,
                detuning_hz=(4 * m_g + k) * RECOIL_FREQUENCY_HZ,
                phi=phi,
                label=label,
                rabi_frequency=rabi,
                pulse_area=np.pi,
            )
        )
        t += sequence[-1].duration
        t += LMT_POST_DELAY

    # --- Clearout, then last upper selective UP π ---
    sequence.append(
        Clearout(
            time=t,
            duration=LMT_PULSE_CLEAROUT_DURATION,
            label="clearout_before_last_selective_upper",
        )
    )
    t += LMT_PULSE_CLEAROUT_DURATION
    t += SELECTIVE_PRE_DELAY
    sequence.append(
        Pulse(
            time=t,
            k=+1,
            detuning_hz=(4 * 2 + 1) * RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="last_selective_upper",
            rabi_frequency=rabi_selective,
            pulse_area=np.pi,
        )
    )
    t += sequence[-1].duration

    # --- MIRROR DOWN π ---
    t += POST_MIRROR_DELAY
    sequence.append(
        Pulse(
            time=t,
            k=-1,
            detuning_hz=(4 * 2 - 1) * RECOIL_FREQUENCY_HZ,
            phi=phase_step,
            label="mirror",
            rabi_frequency=rabi_down,
            pulse_area=np.pi,
        )
    )
    t += sequence[-1].duration
    t += POST_MIRROR_DELAY

    # --- First lower selective UP π ---
    t += SELECTIVE_PRE_DELAY
    sequence.append(
        Pulse(
            time=t,
            k=+1,
            detuning_hz=(4 * 2 + 1) * RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="first_selective_lower",
            rabi_frequency=rabi_selective,
            pulse_area=np.pi,
        )
    )
    t += sequence[-1].duration
    sequence.append(
        Clearout(
            time=t,
            duration=LMT_PULSE_CLEAROUT_DURATION,
            label="clearout_after_first_selective_lower",
        )
    )
    t += LMT_PULSE_CLEAROUT_DURATION

    # --- Forward LMT on lower arm: N-2 pulses ---
    m = 3
    is_excited = True
    for i in range(N - 2):
        t += LMT_PRE_DELAY
        if i % 2 == 0:
            # DOWN: |e, m⟩ → |g, m+1⟩, phase=phase_step
            k = -1
            rabi = rabi_down
            m_g = m + 1
            phi = phase_step
            label = f"lower_fw_down_{i}"
            is_excited = False
            m += 1
        else:
            # UP: |g, m⟩ → |e, m+1⟩
            k = +1
            rabi = rabi_up_high
            m_g = m
            phi = 0.0
            label = f"lower_fw_up_{i}"
            is_excited = True
            m += 1
        sequence.append(
            Pulse(
                time=t,
                k=k,
                detuning_hz=(4 * m_g + k) * RECOIL_FREQUENCY_HZ,
                phi=phi,
                label=label,
                rabi_frequency=rabi,
                pulse_area=np.pi,
            )
        )
        t += sequence[-1].duration
        t += LMT_POST_DELAY
    m_after_lower_fw = m

    # --- Dark time lower arm ---
    t += delay_between_interferometry_pulses

    # --- Reverse LMT on lower arm: N-2 pulses ---
    m = m_after_lower_fw
    is_excited = False
    for i in range(N - 2):
        t += LMT_PRE_DELAY
        if i % 2 == 0:
            # DOWN: |g, m⟩ → |e, m-1⟩, phase=phase_step
            k = -1
            rabi = rabi_down
            m_g = m
            phi = phase_step
            label = f"lower_rv_down_{i}"
            is_excited = True
            m -= 1
        else:
            # UP: |e, m⟩ → |g, m-1⟩
            k = +1
            rabi = rabi_up_high
            m_g = m - 1
            phi = 0.0
            label = f"lower_rv_up_{i}"
            is_excited = False
            m -= 1
        sequence.append(
            Pulse(
                time=t,
                k=k,
                detuning_hz=(4 * m_g + k) * RECOIL_FREQUENCY_HZ,
                phi=phi,
                label=label,
                rabi_frequency=rabi,
                pulse_area=np.pi,
            )
        )
        t += sequence[-1].duration
        t += LMT_POST_DELAY

    # --- Clearout, then last lower selective UP π ---
    sequence.append(
        Clearout(
            time=t,
            duration=LMT_PULSE_CLEAROUT_DURATION,
            label="clearout_before_last_selective_lower",
        )
    )
    t += LMT_PULSE_CLEAROUT_DURATION
    t += SELECTIVE_PRE_DELAY
    sequence.append(
        Pulse(
            time=t,
            k=+1,
            detuning_hz=(4 * 2 + 1) * RECOIL_FREQUENCY_HZ,
            phi=0.0,
            label="last_selective_lower",
            rabi_frequency=rabi_selective,
            pulse_area=np.pi,
        )
    )
    t += sequence[-1].duration

    # --- BS2 DOWN π/2 ---
    t += PRE_BS2_DELAY
    sequence.append(
        Pulse(
            time=t,
            k=-1,
            detuning_hz=(4 * 2 - 1) * RECOIL_FREQUENCY_HZ,
            phi=4 * phase_step,
            label="BS2",
            rabi_frequency=rabi_down,
            pulse_area=np.pi / 2,
        )
    )
    t += sequence[-1].duration

    return sequence


if __name__ == "__main__":
    seq = build_lmt_real_sequence(N=7)
    print(f"Built sequence with {len(seq)} events:")
    for ev in seq:
        if isinstance(ev, Pulse):
            print(
                f"  t={ev.time*1e6:8.2f}us  PULSE   k={ev.k:+d}  "
                f"Δ={ev.detuning_hz/1e3:+7.2f}kHz  "
                f"area={ev.pulse_area/np.pi:.2f}π  "
                f"φ={ev.phi:+.3f}  {ev.label}"
            )
        else:
            print(
                f"  t={ev.time*1e6:8.2f}us  CLEAR   duration={ev.duration*1e6:.1f}us  "
                f"{ev.label}"
            )
