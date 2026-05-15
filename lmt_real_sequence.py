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
LMT_PRE_DELAY = 50e-6  # `t_start = now_mu() + 50us` at the top of each lmt_series iteration
LMT_POST_DELAY = 10e-6  # `delay(10e-6)` at the end of fire_lmt_pulse
SELECTIVE_PRE_DELAY = 50e-6  # `t_pulse = now_mu() + 50us` inside do_selective_lmt_pulse
POST_BS1_DELAY = 100e-6  # `delay(100e-6)` after BS1
POST_MIRROR_DELAY = 1e-6  # `delay(1e-6)` after the mirror pulse
PRE_BS2_DELAY = 100e-6  # `t_start_last_pulse_mu = now_mu() + 100us` before BS2

# Gap between the end of velocity-selection clearout and BS1.  In the real
# experiment this is a long chain of post-dipole-trap hooks; left as a
# parameter so consumers can tune it for their model of the atom's free fall.
DEFAULT_VS_TO_BS1_GAP = 1e-3


def _res_detuning(m_ground: int, k: int) -> float:
    """On-resonance laser detuning for ``|g, m_ground⟩ ↔ |e, m_ground + k⟩``.

    Derived from Bordé's effective detuning ``δ_eff(m) = Δ - v(m)/λ - (2m + k) δ_rec``
    with ``v_0 = 0``: setting ``δ_eff = 0`` yields ``Δ = (4 m_g + k) δ_rec``.
    """
    return (4 * m_ground + k) * RECOIL_FREQUENCY_HZ


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

    def add_pulse(t, k, m_ground, phi, label, rabi, area):
        p = Pulse(
            time=t,
            k=k,
            detuning_hz=_res_detuning(m_ground, k),
            phi=phi,
            label=label,
            rabi_frequency=rabi,
            pulse_area=area,
        )
        sequence.append(p)
        return t + p.duration

    def add_clearout(t, label, duration):
        sequence.append(Clearout(time=t, duration=duration, label=label))
        return t + duration

    def add_lmt_series(t, pulses_spec, *, label_prefix):
        """Append an LMT series matching `lmt_series` / `lmt_series_start_down_launch_down`.

        Each spec is a tuple ``(pulse_type, m_ground, phi)`` where pulse_type
        is "down" or "up" and m_ground is the ground-state m of the
        target transition.

        Replicates the 50 us pre-delay before the first pulse and a
        50+10 us gap between consecutive pulses (10 us at the tail of
        ``fire_lmt_pulse`` plus the 50 us pre-delay of the next iteration).
        """
        for i, (pulse_type, m_g, phi) in enumerate(pulses_spec):
            t += LMT_PRE_DELAY
            if pulse_type == "down":
                k = -1
                rabi = rabi_down
            else:
                k = +1
                rabi = rabi_up_high
            t = add_pulse(
                t,
                k=k,
                m_ground=m_g,
                phi=phi,
                label=f"{label_prefix}_{pulse_type}_{i}",
                rabi=rabi,
                area=np.pi,
            )
            t += LMT_POST_DELAY
        return t

    def lmt_walk_forward(initial_m, initial_is_excited, K, phi_for_down):
        """Spec for ``K`` LMT pulses, each adding +1 to m.

        Pattern alternates DOWN (i=0,2,...) and UP (i=1,3,...), matching
        ``lmt_series`` which is "use if we start in the excited state".

        ``phi_for_down`` is the laser phase on down pulses (the down DDS
        keeps whatever phase was last set on it).  Up pulses always carry
        the up DDS's "phase_constant", i.e. effective phase 0 here.
        """
        m, is_excited = initial_m, initial_is_excited
        out = []
        for i in range(K):
            if i % 2 == 0:
                # DOWN: forward requires |e, m⟩ → |g, m+1⟩ (m_g = m + 1).
                assert is_excited
                out.append(("down", m + 1, phi_for_down))
                m += 1
                is_excited = False
            else:
                # UP: forward requires |g, m⟩ → |e, m+1⟩ (m_g = m).
                assert not is_excited
                out.append(("up", m, 0.0))
                m += 1
                is_excited = True
        return out, m, is_excited

    def lmt_walk_reverse(initial_m, initial_is_excited, K, phi_for_down):
        """Spec for ``K`` LMT pulses, each subtracting 1 from m.

        Same alternating pattern (DOWN first), but the entry condition is
        |g, m⟩ — matching ``lmt_series_start_down_launch_down`` which runs
        after the forward sweep has parked the upper (or lower) arm in
        ground state at high m.
        """
        m, is_excited = initial_m, initial_is_excited
        out = []
        for i in range(K):
            if i % 2 == 0:
                # DOWN: reverse requires |g, m⟩ → |e, m-1⟩ (m_g = m).
                assert not is_excited
                out.append(("down", m, phi_for_down))
                m -= 1
                is_excited = True
            else:
                # UP: reverse requires |e, m⟩ → |g, m-1⟩ (m_g = m - 1).
                assert is_excited
                out.append(("up", m - 1, 0.0))
                m -= 1
                is_excited = False
        return out, m, is_excited

    # ------------------------------------------------------------------
    # VELOCITY SELECTION
    # ------------------------------------------------------------------
    # `ClockShelvingAndClearoutBase.clock_shelving` fires one up-beam pi pulse
    # of duration CLOCK_SHELVING_PULSE_TIME (narrow bandwidth -> velocity
    # selective) on resonance with |g, 0⟩ → |e, +1⟩, then does a long
    # fluorescence clearout to discard residual ground-state atoms.
    t = 0.0
    t = add_pulse(
        t,
        k=+1,
        m_ground=0,
        phi=0.0,
        label="velocity_selection",
        rabi=rabi_vs,
        area=np.pi,
    )
    t = add_clearout(t, "velocity_selection_clearout", SHELVING_PULSE_CLEAROUT_DURATION)

    # Atoms are now in |e, +1⟩.  Free fall / setup happens before BS1.
    t += vs_to_bs1_gap

    # ------------------------------------------------------------------
    # BS1: DOWN π/2
    # ------------------------------------------------------------------
    # Splits |e, +1⟩ into |e, +1⟩ (lower arm) + |g, +2⟩ (upper arm).
    # Target transition: |g, +2⟩ ↔ |e, +1⟩ (m_g=+2, k=-1).
    t = add_pulse(
        t, k=-1, m_ground=2, phi=0.0, label="BS1",
        rabi=rabi_down, area=np.pi / 2,
    )
    t += POST_BS1_DELAY

    # ------------------------------------------------------------------
    # First selective UP π (addresses upper arm |g, +2⟩ → |e, +3⟩)
    # ------------------------------------------------------------------
    t += SELECTIVE_PRE_DELAY
    t = add_pulse(
        t, k=+1, m_ground=2, phi=0.0, label="first_selective_upper",
        rabi=rabi_selective, area=np.pi,
    )
    t = add_clearout(t, "clearout_after_first_selective_upper", LMT_PULSE_CLEAROUT_DURATION)

    # ------------------------------------------------------------------
    # Forward LMT on upper arm: N-2 pulses, pattern down/up/down/...
    # ------------------------------------------------------------------
    # State at entry: upper |e, +3⟩.  Each pulse advances m by +1.
    upper_forward, m_after_upper_fw, _ = lmt_walk_forward(
        initial_m=3, initial_is_excited=True, K=N - 2, phi_for_down=0.0,
    )
    t = add_lmt_series(t, upper_forward, label_prefix="upper_fw")

    # ------------------------------------------------------------------
    # Dark time between forward and reverse LMT on upper arm
    # ------------------------------------------------------------------
    # `t_start_lmt_mirror_mu = t_end_bs_mu + delay_between_interferometry_pulses`.
    # `t_end_bs_mu` ≈ end of last forward pulse + 10us (LMT_POST_DELAY already added).
    t += delay_between_interferometry_pulses

    # ------------------------------------------------------------------
    # Reverse LMT on upper arm: N-2 pulses, walks m back down
    # ------------------------------------------------------------------
    # State at entry: upper |g, m_after_upper_fw⟩.  Each pulse subtracts 1 from m.
    upper_reverse, _, _ = lmt_walk_reverse(
        initial_m=m_after_upper_fw, initial_is_excited=False,
        K=N - 2, phi_for_down=0.0,
    )
    t = add_lmt_series(t, upper_reverse, label_prefix="upper_rv")

    # ------------------------------------------------------------------
    # Clearout, then last upper selective UP π (|e, +3⟩ → |g, +2⟩)
    # ------------------------------------------------------------------
    t = add_clearout(t, "clearout_before_last_selective_upper", LMT_PULSE_CLEAROUT_DURATION)
    t += SELECTIVE_PRE_DELAY
    t = add_pulse(
        t, k=+1, m_ground=2, phi=0.0, label="last_selective_upper",
        rabi=rabi_selective, area=np.pi,
    )

    # ------------------------------------------------------------------
    # MIRROR DOWN π (swaps the two arms)
    # ------------------------------------------------------------------
    # Target |g, +2⟩ ↔ |e, +1⟩.  Upper |g, +2⟩ → |e, +1⟩; lower |e, +1⟩ → |g, +2⟩.
    t += POST_MIRROR_DELAY
    t = add_pulse(
        t, k=-1, m_ground=2, phi=phase_step, label="mirror",
        rabi=rabi_down, area=np.pi,
    )
    t += POST_MIRROR_DELAY

    # ------------------------------------------------------------------
    # First lower selective UP π (lower arm now |g, +2⟩ → |e, +3⟩)
    # ------------------------------------------------------------------
    t += SELECTIVE_PRE_DELAY
    t = add_pulse(
        t, k=+1, m_ground=2, phi=0.0, label="first_selective_lower",
        rabi=rabi_selective, area=np.pi,
    )
    t = add_clearout(t, "clearout_after_first_selective_lower", LMT_PULSE_CLEAROUT_DURATION)

    # ------------------------------------------------------------------
    # Forward LMT on lower arm: same structure as upper forward
    # ------------------------------------------------------------------
    # After the mirror the down DDS phase is `phase_constant + phase_step`,
    # so every down pulse in the lower-arm LMT carries phase ``phase_step``.
    lower_forward, m_after_lower_fw, _ = lmt_walk_forward(
        initial_m=3, initial_is_excited=True, K=N - 2, phi_for_down=phase_step,
    )
    t = add_lmt_series(t, lower_forward, label_prefix="lower_fw")

    # Dark time on lower arm.  In the experiment this is an explicit
    # `delay(self.delay_between_interferometry_pulses.get())` between the
    # forward and reverse lower LMT series.
    t += delay_between_interferometry_pulses

    # ------------------------------------------------------------------
    # Reverse LMT on lower arm
    # ------------------------------------------------------------------
    lower_reverse, _, _ = lmt_walk_reverse(
        initial_m=m_after_lower_fw, initial_is_excited=False,
        K=N - 2, phi_for_down=phase_step,
    )
    t = add_lmt_series(t, lower_reverse, label_prefix="lower_rv")

    # ------------------------------------------------------------------
    # Clearout, then last lower selective UP π (|e, +3⟩ → |g, +2⟩)
    # ------------------------------------------------------------------
    t = add_clearout(t, "clearout_before_last_selective_lower", LMT_PULSE_CLEAROUT_DURATION)
    t += SELECTIVE_PRE_DELAY
    t = add_pulse(
        t, k=+1, m_ground=2, phi=0.0, label="last_selective_lower",
        rabi=rabi_selective, area=np.pi,
    )

    # ------------------------------------------------------------------
    # BS2 DOWN π/2
    # ------------------------------------------------------------------
    t += PRE_BS2_DELAY
    t = add_pulse(
        t, k=-1, m_ground=2, phi=4 * phase_step, label="BS2",
        rabi=rabi_down, area=np.pi / 2,
    )

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
