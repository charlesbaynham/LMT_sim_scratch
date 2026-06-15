# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: lmt-sim-scratch
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Shaped double launch -- RID 74397
#
# The LMT shaped-double-launch sequence run on 2026-06-12
# (`LMTInterferometryWithShapedDoubleLaunch` in icl_experiments), decoded from
# the lab pulse record and reconstructed with `lmt_sim`. The dump is embedded
# here so the script is self-contained (this repo has no RDS access).
#
# The pulse record alone is NOT enough to infer the trajectory; three things
# the recorder cannot capture have to be reapplied (each in its own section):
#
# 1. **The velocity-selection pulse's OPLL ramp.** The 380 us shelving pulse
#    fires during the gravity DRG ramp, which the recorder cannot see: it
#    stores the static pre-ramp setpoint (exactly 80 MHz). The pulse-centre
#    value is +2.67 kHz higher; without this correction the parser's frequency
#    anchor -- and hence EVERY pulse's detuning -- is off by 0.57 recoils.
# 2. **The 200 us Jesse pulse.** A phase-shaped pulse driving both arms at
#    once; the recorder stores it as a square pulse at the carrier (a known
#    limitation, see the TODO in `first_shaped_lmt_pulse`). It is replaced by
#    two simultaneous arm-restricted stand-in pi pulses.
# 3. **Clearouts.** The lab fires ground-state clearout pulses at six points;
#    the pulse record does not store them, so they are reinserted by hand.
#
# No other free parameters are needed: with the measured probe-shift alpha and
# each square pulse's duration-implied Rabi frequency taken as its TRUE Rabi
# frequency, all ninety plain pi pulses land on odd recoil rungs to within
# 0.13 recoils (the six "selective" pulses carry known sub-recoil software
# detunings on top). The inference then closes into the intended topology:
# velocity selection, splitter, double launch, simultaneous second split,
# W-shaped LMT interferometer out to +-63 recoil rungs, and recombination into
# 8 clouds whose 4+4 outputs overlap.

# %%
import sys
import dataclasses

sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import lmt_sim.lmt_simulation as sim
import lmt_sim.lmt_sequence as seq

REC = sim.RECOIL_FREQUENCY_HZ

# %%
# Genuine new-format pulse dump from ARTIQ RID 74397
# (LMTInterferometryWithShapedDoubleLaunch): float64 SI pulse_record_flat as
# emitted by PulseDMARecording in icl_experiments.
pulse_record_flat = np.array([
    99.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 4.936000000000001e-06, 0.002690399,
    0.002736543, 0.0029450870000000003, 0.003025086, 0.003092085, 0.0031720840000000004, 0.0032390830000000002,
    0.003319082, 0.0033860810000000004, 0.0034660800000000003, 0.008033079, 0.008150094, 0.008230093,
    0.008347108, 0.008427107000000001, 0.008544122000000001, 0.008624121, 0.008741136, 0.008821135,
    0.00893815, 0.009023327000000001, 0.00925689, 0.009837921000000001, 0.011029944, 0.011085735000000001,
    0.011393591000000002, 0.01147359, 0.011540589, 0.011620588000000001, 0.011687587000000001, 0.011767586,
    0.011834585, 0.011914584, 0.011981583, 0.012061582000000001, 0.012128581000000001, 0.01220858,
    0.012275579, 0.012355578, 0.012422577, 0.012502576000000001, 0.012569575000000001, 0.012699581000000001,
    0.01277958, 0.012846579, 0.012926578000000001, 0.012993577000000001, 0.013073576, 0.013140575000000002,
    0.013220574, 0.013287573, 0.013367572000000001, 0.013434571000000001, 0.01351457, 0.013581569000000002,
    0.013661568, 0.013728567, 0.013808566000000001, 0.013875565000000001, 0.014163367000000001, 0.014281215000000002,
    0.014369742000000001, 0.014677599000000001, 0.014757598, 0.014824597, 0.014904596, 0.014971595,
    0.015051594000000001, 0.015118593000000001, 0.015198592, 0.015265591, 0.015345590000000001, 0.015412589,
    0.015492588000000002, 0.015559587000000001, 0.015639586, 0.015706585000000002, 0.015786584, 0.015853583,
    0.015983590000000002, 0.016063589, 0.016130588, 0.016210587000000002, 0.016277586, 0.016357585,
    0.016424584000000002, 0.016504583, 0.016571582, 0.016651581000000002, 0.01671858, 0.016798579,
    0.016865578000000003, 0.016945577, 0.017012576, 0.017092575000000002, 0.017159574, 0.017447367000000002,
    0.017568887000000002, 0.00038, 3.4e-05, 0.00019999900000000002, 6.8e-05, 5.4999000000000007e-05,
    6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05,
    5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05,
    5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 3.0555e-05, 6.8e-05,
    5.4999000000000007e-05, 3.4e-05, 9.9999e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05,
    5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05,
    5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05,
    5.4999000000000007e-05, 6.8e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05,
    6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05,
    6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05,
    6.8e-05, 9.9999e-05, 6.8e-05, 9.9999e-05, 6.8e-05, 5.4999000000000007e-05,
    6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05,
    6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05,
    6.8e-05, 5.4999000000000007e-05, 6.8e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05,
    5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05,
    5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05, 5.4999000000000007e-05, 6.8e-05,
    5.4999000000000007e-05, 6.8e-05, 9.9999e-05, 3.4e-05, 80000000.0, 79972625.31596854,
    80032357.66978168, 79987049.32424746, 80004873.91580191, 80003785.37293835, 79988137.86711103, 80020521.42162924,
    79971401.81842014, 80037257.47032012, 79954665.76972926, 80112789.52355428, 79894967.50672889, 80096755.73332049,
    79911001.29696268, 80080721.9430867, 79927035.08719645, 80064688.15285291, 79943068.87743025, 80048654.36261913,
    79959102.66766404, 80033338.8136347, 79973427.34255576, 80025330.72553419, 79967332.46558852, 80024883.93675801,
    79981226.61583747, 80010696.6242119, 79997962.66452836, 79993960.575521, 80014698.71321924, 79977224.52683012,
    80031434.76191013, 79960488.47813924, 80048170.81060103, 79943752.42944835, 80064906.85929191, 79927016.38075747,
    80081642.9079828, 79910280.33206658, 80098378.95667368, 79893544.2833757, 80115115.00536457, 80113089.63322417,
    79897433.6068252, 80092225.68191506, 79918297.5581343, 80071361.73060594, 79939161.50944342, 80050497.77929683,
    79960025.46075253, 80029633.82798772, 79980889.41206165, 80008769.87667862, 80001753.36337076, 79987905.9253695,
    80022617.31467988, 79967041.97406039, 80043481.26598899, 79946178.02275127, 80067895.97090386, 79921482.42248091,
    80070793.61543928, 79934916.92311554, 80056806.31693384, 79951652.97180642, 80040070.26824296, 79968389.0204973,
    80023334.21955207, 79985125.06918819, 80006598.17086118, 80001861.11787908, 79989862.1221703, 80018597.16656996,
    79973126.0734794, 80035333.21526085, 79956390.02478851, 80052069.26395173, 79939653.97609763, 80068805.31264262,
    80066979.92646156, 79943543.31358781, 80046115.97515245, 79964407.26489693, 80025252.02384333, 79985271.21620604,
    80004388.07253422, 80006135.16751514, 79983524.1212251, 80026999.11882426, 79962660.16991599, 80047863.07013337,
    79941796.21860689, 80068727.02144249, 79920932.26729777, 80089590.9727516, 79900068.31598866, 80113005.55130038,
    79875590.4771736, 200000000.0, 200005800.0, 200000000.0, 200005800.0, 200001500.0,
    200005800.0, 200001500.0, 200005800.0, 200001500.0, 200005800.0, 200001500.0,
    200001500.0, 200005800.0, 200001500.0, 200005800.0, 200001500.0, 200005800.0,
    200001500.0, 200005800.0, 200001500.0, 200005800.0, 200000500.0, 200005800.0,
    200000500.0, 200005800.0, 200000500.0, 200005800.0, 200001500.0, 200005800.0,
    200001500.0, 200005800.0, 200001500.0, 200005800.0, 200001500.0, 200005800.0,
    200001500.0, 200005800.0, 200001500.0, 200005800.0, 200001500.0, 200005800.0,
    200001500.0, 200005800.0, 200005800.0, 200001500.0, 200005800.0, 200001500.0,
    200005800.0, 200001500.0, 200005800.0, 200001500.0, 200005800.0, 200001500.0,
    200005800.0, 200001500.0, 200005800.0, 200001500.0, 200005800.0, 200001500.0,
    200005800.0, 200000500.0, 200005800.0, 200000500.0, 200005800.0, 200001500.0,
    200005800.0, 200001500.0, 200005800.0, 200001500.0, 200005800.0, 200001500.0,
    200005800.0, 200001500.0, 200005800.0, 200001500.0, 200005800.0, 200001500.0,
    200005800.0, 200001500.0, 200005800.0, 200005800.0, 200001500.0, 200005800.0,
    200001500.0, 200005800.0, 200001500.0, 200005800.0, 200001500.0, 200005800.0,
    200001500.0, 200005800.0, 200001500.0, 200005800.0, 200001500.0, 200005800.0,
    200001500.0, 200005800.0, 200000500.0, 200005800.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0, 99471700.0,
    99471700.0, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6,
])
pulse_record_offsets = np.array([0], dtype=np.int64)

dump = seq.decode_pulse_record_flat(pulse_record_flat, pulse_record_offsets)[0]
n_pulses = len(dump.is_up)
assert n_pulses == 99

# %% [markdown]
# ## Pulse classes
#
# The clock switch-AOM offset recorded per pulse cleanly partitions the dump
# into the beam/intensity classes the experiment code uses (each
# `set_clock_up_dds` / `set_clock_down_dds` call in `LMT_launch_mixins.py`
# selects one):
#
# * `+0 Hz`: the 380 us velocity-selection pulse and the 200 us Jesse pulse
#   (RAM playback at the base carrier).
# * `+1500 Hz` (`up_switch_detuning_higher_intensity`): the 41 full-power
#   55 us up-beam LMT pi pulses.
# * `+500 Hz` (`up_switch_detuning_lower_intensity`): the 6 "selective"
#   up-beam pulses fired through `do_selective_lmt_pulse` (the 100 us
#   single-arm pulses, the 30.6 us launch pi/2 and one 55 us ladder pulse).
# * `+5800 Hz` (`down_switch_detuning`): all 50 down-beam pulses
#   (68 us pi and 34 us pi/2).

# %%
switch_offset = np.round(dump.switch_hz - 200e6).astype(int)
classes = {
    off: np.where(switch_offset == off)[0] for off in np.unique(switch_offset)
}
for off, idx in classes.items():
    durs = np.unique(np.round(dump.durations_s[idx] * 1e6, 1))
    print(
        f"switch +{off:4d} Hz: {len(idx):2d} pulses, "
        f"is_up={np.unique(dump.is_up[idx])}, durations {durs} us"
    )
assert set(classes) == {0, 500, 1500, 5800}
assert np.all(dump.is_up[classes[0]]) and np.all(dump.is_up[classes[500]])
assert np.all(dump.is_up[classes[1500]]) and not np.any(dump.is_up[classes[5800]])
SELECTIVE_PULSES = list(classes[500])
assert SELECTIVE_PULSES == [21, 23, 25, 60, 62, 97]

# %% [markdown]
# ## Correct the VS pulse's OPLL to its pulse-centre value
#
# The 380 us velocity-selection pulse fires while the clock OPLL is running
# the gravity DRG ramp (`clock_shelving` starts the ramp immediately before
# it). The recorder reports ramp-aware pulse-centre OPLL values only for
# pulses whose ramp it starts itself (`do_selective_lmt_pulse` etc.); for the
# shelving pulse it stores the static pre-ramp setpoint -- visible in the dump
# as EXACTLY 80 MHz, while every other pulse's OPLL is a non-round value.
#
# The parser anchors the entire frequency scale on this pulse, so the missing
# ramp_rate * duration / 2 = +2.67 kHz (0.57 recoil) shifts every detuning in
# the sequence. This is the same correction the RID 74108 analysis applied.
# Without it, the residuals look like an extra ~2.8 kHz light shift on the
# up-beam pulses -- a wrong attribution: the shift is the same for the
# full-power and 10.5 dB-attenuated classes, so it cannot be intensity-
# dependent physics.

# %%
opll_hz = dump.opll_hz.copy()
# Guard against double-correcting if the recorder is ever fixed to report the
# ramp centre itself: the static value is exactly the 80 MHz base.
assert opll_hz[0] == 80e6, (
    "VS pulse OPLL is not the static pre-ramp value -- the recorder may now "
    "report pulse-centre values, in which case this correction must be removed"
)
vs_ramp_correction_hz = sim.GRAVITY_DOPPLER_PER_SEC_HZ * dump.durations_s[0] / 2
opll_hz[0] += vs_ramp_correction_hz
print(f"VS OPLL pulse-centre correction: {vs_ramp_correction_hz:+.1f} Hz")

# %% [markdown]
# ## Stark shift and initial velocity
#
# The probe-shift coefficient is the measured value from
# "2026-06-09 Clock shift gap-filling even Omega2 grid" (defined there per
# rad/s, converted to 1/Hz). All pulses are plain square pulses (the Jesse
# pulse is replaced below), so each pulse's duration-implied Rabi frequency IS
# its true Rabi frequency and the default `alpha * rabi**2` light-shift model
# applies with no `stark_rabi_frequency` markings. The residual check below
# validates this: every plain pi pulse must land on an odd recoil rung.
#
# `initial_velocity_z` is the one genuinely unknown parameter: the atom's
# residual velocity at the anchor time. It is inferred by requiring pulse 3
# (first down-beam ladder pulse, e m=3 -> g m=4) to be resonant on rung -7.
# Because the down pulses are square pi pulses, their duration-implied Rabi
# frequencies are exact and the measured alpha fixes their Stark shift
# outright -- nothing on the down beam is fitted -- so under the assumption
# that pulse 3 was on resonance this IS a velocity inference, not a catch-all
# parameter. An independent time-of-flight measurement would be a worthwhile
# cross-check of the inferred value.

# %%
PROBE_SHIFT_ALPHA = -2.04e-6 * (2 * np.pi)  # 1/Hz, measured 2026-06-09

build_kwargs = dict(
    **{**dataclasses.asdict(dump), "opll_hz": opll_hz},
    probe_induced_alpha_up=PROBE_SHIFT_ALPHA,
    probe_induced_alpha_down=PROBE_SHIFT_ALPHA,
)


def built_pulses(initial_velocity_z):
    _, s = seq.build_sequence_from_lab_pulse_dump(
        **build_kwargs, initial_velocity_z=initial_velocity_z
    )
    return s, [e for e in s if isinstance(e, seq.Pulse)]


def effective_rung(pulse):
    return (
        sim._effective_detuning_hz(
            pulse.detuning_hz,
            pulse.probe_shift_coefficient,
            pulse.effective_stark_rabi_frequency,
        )
        / REC
    )


# Pin v0 from pulse 3's EFFECTIVE detuning = rung -7. The dependence of the
# down-beam detunings on v0 is linear (2 v0 / lambda); fit the slope
# empirically so this stays correct if the parser's frame handling changes.
_, p_a = built_pulses(0.0)
_, p_b = built_pulses(1e-4)
slope = (p_b[3].detuning_hz - p_a[3].detuning_hz) / 1e-4
initial_velocity_z = (-7 - effective_rung(p_a[3])) * REC / slope
print(f"inferred initial_velocity_z = {initial_velocity_z * 1e3:+.4f} mm/s")
assert abs(initial_velocity_z) < 5e-3

sequence, pulses = built_pulses(initial_velocity_z)

# Validation: with measured alpha, duration-implied Rabi frequencies and the
# VS ramp correction -- no fitted Stark parameters -- every plain pi pulse
# must sit on an odd recoil rung. The selective pulses carry known sub-recoil
# software detunings (first_lmt_freq etc., 0.2-1 kHz) so they get a slightly
# looser bound.
residuals_plain, residuals_selective = [], []
for i, p in enumerate(pulses):
    if i in (0, 2):
        continue
    rung = effective_rung(p)
    resid = rung - (2 * round((rung - 1) / 2) + 1)
    (residuals_selective if i in SELECTIVE_PULSES else residuals_plain).append(resid)
print(
    f"plain pi-pulse odd-rung residuals: rms "
    f"{np.sqrt(np.mean(np.square(residuals_plain))):.3f}, "
    f"max |.| {np.max(np.abs(residuals_plain)):.3f} recoils\n"
    f"selective-pulse residuals: max |.| "
    f"{np.max(np.abs(residuals_selective)):.3f} recoils"
)
assert np.max(np.abs(residuals_plain)) < 0.15
assert np.max(np.abs(residuals_selective)) < 0.25

# %% [markdown]
# ## Replace the 200 us Jesse pulse with two arm-restricted stand-ins
#
# Pulse 2 is `JessePulseLMT` fired by `launch_hook_double_cloud` (N_kicks=1,
# detuning=2 kHz): a phase-shaped pulse addressing BOTH arms left by the pi/2
# splitter, kicking them in opposite directions:
#
# * upper arm: ground m=2 -> excited m=3 (absorption) -- rung **5**
# * lower arm: excited m=1 -> ground m=0 (stimulated emission) -- rung **1**
#
# The recorder stores it as a square pulse at the carrier (see the TODO in
# `first_shaped_lmt_pulse`), which sits BETWEEN the two driven transitions, so
# fed to the simulator unmodified it does nothing at all -- this single pulse
# was why the raw inference exploded: the launch never happened and every
# later pulse mis-targeted.
#
# As in the RID 74108 analysis, it becomes two simultaneous pi pulses, each
# restricted IN CODE to its own arm (`restrict_to_m_ground` stands in for the
# pulse shaping). Timing is preserved by padding the remainder of the 200 us
# slot with freefall.

# %%
jesse = pulses[2]
assert np.isclose(jesse.duration, 200e-6, rtol=0.05)
# Sanity: the recorded carrier must lie between the two driven rungs
assert 1 * REC < jesse.detuning_hz < 5 * REC, (
    f"Jesse carrier {jesse.detuning_hz:.0f} Hz is not between rungs 1 and 5 -- "
    "re-derive the stand-in rungs before trusting this surgery"
)

t_pi_full = 1 / (2 * pulses[4].rabi_frequency)
common = dict(
    k=+1,
    phi=jesse.phi,
    rabi_frequency=pulses[4].rabi_frequency,
    duration=t_pi_full,
    probe_shift_coefficient=0.0,
)
jesse_upper = seq.Pulse(
    detuning_hz=5 * REC, label="jesse_upper", restrict_to_m_ground=2, **common
)
jesse_lower = seq.Pulse(
    detuning_hz=1 * REC,
    label="jesse_lower",
    restrict_to_m_ground=0,
    simultaneous_with_previous=True,
    **common,
)
jesse_padding = seq.Freefall(
    duration=jesse.duration - t_pi_full, label="jesse_padding"
)

jesse_position = sequence.index(jesse)
sequence = (
    sequence[:jesse_position]
    + [jesse_upper, jesse_lower, jesse_padding]
    + sequence[jesse_position + 1 :]
)
print(
    f"replaced the {jesse.duration * 1e6:.0f} us Jesse pulse with two "
    f"simultaneous {t_pi_full * 1e6:.0f} us stand-ins on rungs 5 and 1"
)

# %% [markdown]
# ## Reinsert the clearouts
#
# The lab fires ground-state clearout pulses that the pulse record does not
# store. Without them the inference keeps ground-state branches the experiment
# physically removed. Positions from the icl_experiments source
# (`launch_hook_double_cloud` and `LMTInterferometryMixin.do_clock_interferometry`),
# given as "after pulse N" in the dump numbering:
#
# * after 21: launch-closing pi/2, ground state thrown away
# * after 23: end of the launch ladder-clearing block
# * after 25: after the first selective interferometry pulse
# * after 59: before the upper-arm mirror selective pulse
# * after 62: after the lower-arm mirror selective pulse
# * after 96: before the last selective pulse
#
# They are inserted with zero duration: the recorded inter-pulse gaps already
# appear as freefall events, so the clearouts must not add time.

# %%
CLEAROUT_AFTER = [21, 23, 25, 59, 62, 96]

with_clearouts = []
pulse_index = -1
for event in sequence:
    with_clearouts.append(event)
    if isinstance(event, seq.Pulse) and event.label == "LMT":
        # Stand-ins are labelled jesse_*; the dump numbering skips them but
        # counts the replaced pulse, hence the bump when passing the splice.
        pulse_index += 1
    elif event is jesse_padding:
        pulse_index += 1  # the slot of original pulse 2
    if pulse_index in CLEAROUT_AFTER and isinstance(event, seq.Pulse):
        with_clearouts.append(seq.Clearout(duration=0.0))
sequence = with_clearouts

# %% [markdown]
# ## Recoil ladder
#
# Each pulse's effective (Stark-corrected) detuning in recoil units, i.e. the
# rung the simulator sees after the corrections above. Plain pi pulses sit on
# odd rungs; the four 100 us selective pulses all address rung 29.

# %%
plot_pulses = [e for e in sequence if isinstance(e, seq.Pulse)]
rungs = [effective_rung(p) for p in plot_pulses]
colours = ["tab:orange" if p.k == 1 else "tab:red" for p in plot_pulses]

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(plot_pulses)), rungs, color=colours, width=0.9)
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xlabel("Pulse index (after surgery)")
ax.set_ylabel("Effective detuning (recoil units)")
ax.set_title("RID 74397: shaped double-launch pulse ladder")
ax.legend(
    handles=[
        Patch(color="tab:orange", label="up (k=+1)"),
        Patch(color="tab:red", label="down (k=-1)"),
    ],
    frameon=False,
)

# %% [markdown]
# ## Space-time trajectory
#
# Velocity selection, splitter, double launch (two clouds at the same momentum
# but different heights), the simultaneous second split into four, the
# W-shaped LMT sequence out to rung +-63, and recombination: the final pi/2
# leaves 8 clouds, four excited (m=13) overlapping four ground (m=14).

# %%
clouds, clearout_times = seq.compute_spacetime_trajectory(
    sequence, plot=True, max_branches=16, include_gravity=False
)


# %%
clouds, clearout_times = seq.compute_spacetime_trajectory(
    sequence, plot=True, max_branches=16, include_gravity=True
)
assert len(clearout_times) == len(CLEAROUT_AFTER)

live = [c for c in clouds if c.alive]
final = sorted((c.m[-1], c.is_ground[-1]) for c in live)
print(f"{len(live)} live clouds at the end: {final}")
assert len(live) == 8
assert final == [(13, False)] * 4 + [(14, True)] * 4
assert max(abs(m) for c in clouds for m in c.m) == 32
