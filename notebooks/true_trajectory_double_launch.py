# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: lmt-sim-scratch
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trajectory Inference Tests
#
# Test the new simulation-based trajectory inference for various pulse sequences.

# %%
# Add auto reload:
# %load_ext autoreload
# %autoreload 2

# %%
import sys
from pathlib import Path

sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
import lmt_sim.lmt_simulation as sim
import lmt_sim.lmt_sequence as seq
from scipy import constants as scipy_constants
import version_info as vs

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator

# %
# Genuine new-format pulse dump from ARTIQ run RID 74397
# (LMTInterferometryWithShapedDoubleLaunch, 2026-06-12) -- the shaped double
# launch being run on the experiment. Captured live from the master's
# `pulse_record` dataset, which the lab recorder (PulseDMARecording in
# icl_experiments) now emits in the float64 SI `pulse_record_flat` /
# `pulse_record_offsets` layout: SI seconds for the times, Hz for the
# frequencies and full-precision volts for `delivery_setpoint` (2.6 V here, not
# the 2.0 V the old int64 recorder truncated everything to). It is decoded with
# the shared decoder so this notebook runs the exact same path as real archived
# data.
pulse_record_flat = np.array(
    [
        99.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        4.936000000000001e-06,
        0.002690399,
        0.002736543,
        0.0029450870000000003,
        0.003025086,
        0.003092085,
        0.0031720840000000004,
        0.0032390830000000002,
        0.003319082,
        0.0033860810000000004,
        0.0034660800000000003,
        0.008033079,
        0.008150094,
        0.008230093,
        0.008347108,
        0.008427107000000001,
        0.008544122000000001,
        0.008624121,
        0.008741136,
        0.008821135,
        0.00893815,
        0.009023327000000001,
        0.00925689,
        0.009837921000000001,
        0.011029944,
        0.011085735000000001,
        0.011393591000000002,
        0.01147359,
        0.011540589,
        0.011620588000000001,
        0.011687587000000001,
        0.011767586,
        0.011834585,
        0.011914584,
        0.011981583,
        0.012061582000000001,
        0.012128581000000001,
        0.01220858,
        0.012275579,
        0.012355578,
        0.012422577,
        0.012502576000000001,
        0.012569575000000001,
        0.012699581000000001,
        0.01277958,
        0.012846579,
        0.012926578000000001,
        0.012993577000000001,
        0.013073576,
        0.013140575000000002,
        0.013220574,
        0.013287573,
        0.013367572000000001,
        0.013434571000000001,
        0.01351457,
        0.013581569000000002,
        0.013661568,
        0.013728567,
        0.013808566000000001,
        0.013875565000000001,
        0.014163367000000001,
        0.014281215000000002,
        0.014369742000000001,
        0.014677599000000001,
        0.014757598,
        0.014824597,
        0.014904596,
        0.014971595,
        0.015051594000000001,
        0.015118593000000001,
        0.015198592,
        0.015265591,
        0.015345590000000001,
        0.015412589,
        0.015492588000000002,
        0.015559587000000001,
        0.015639586,
        0.015706585000000002,
        0.015786584,
        0.015853583,
        0.015983590000000002,
        0.016063589,
        0.016130588,
        0.016210587000000002,
        0.016277586,
        0.016357585,
        0.016424584000000002,
        0.016504583,
        0.016571582,
        0.016651581000000002,
        0.01671858,
        0.016798579,
        0.016865578000000003,
        0.016945577,
        0.017012576,
        0.017092575000000002,
        0.017159574,
        0.017447367000000002,
        0.017568887000000002,
        0.00038,
        3.4e-05,
        0.00019999900000000002,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        3.0555e-05,
        6.8e-05,
        5.4999000000000007e-05,
        3.4e-05,
        9.9999e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        9.9999e-05,
        6.8e-05,
        9.9999e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        5.4999000000000007e-05,
        6.8e-05,
        9.9999e-05,
        3.4e-05,
        80000000.0,
        79972625.31596854,
        80032357.66978168,
        79987049.32424746,
        80004873.91580191,
        80003785.37293835,
        79988137.86711103,
        80020521.42162924,
        79971401.81842014,
        80037257.47032012,
        79954665.76972926,
        80112789.52355428,
        79894967.50672889,
        80096755.73332049,
        79911001.29696268,
        80080721.9430867,
        79927035.08719645,
        80064688.15285291,
        79943068.87743025,
        80048654.36261913,
        79959102.66766404,
        80033338.8136347,
        79973427.34255576,
        80025330.72553419,
        79967332.46558852,
        80024883.93675801,
        79981226.61583747,
        80010696.6242119,
        79997962.66452836,
        79993960.575521,
        80014698.71321924,
        79977224.52683012,
        80031434.76191013,
        79960488.47813924,
        80048170.81060103,
        79943752.42944835,
        80064906.85929191,
        79927016.38075747,
        80081642.9079828,
        79910280.33206658,
        80098378.95667368,
        79893544.2833757,
        80115115.00536457,
        80113089.63322417,
        79897433.6068252,
        80092225.68191506,
        79918297.5581343,
        80071361.73060594,
        79939161.50944342,
        80050497.77929683,
        79960025.46075253,
        80029633.82798772,
        79980889.41206165,
        80008769.87667862,
        80001753.36337076,
        79987905.9253695,
        80022617.31467988,
        79967041.97406039,
        80043481.26598899,
        79946178.02275127,
        80067895.97090386,
        79921482.42248091,
        80070793.61543928,
        79934916.92311554,
        80056806.31693384,
        79951652.97180642,
        80040070.26824296,
        79968389.0204973,
        80023334.21955207,
        79985125.06918819,
        80006598.17086118,
        80001861.11787908,
        79989862.1221703,
        80018597.16656996,
        79973126.0734794,
        80035333.21526085,
        79956390.02478851,
        80052069.26395173,
        79939653.97609763,
        80068805.31264262,
        80066979.92646156,
        79943543.31358781,
        80046115.97515245,
        79964407.26489693,
        80025252.02384333,
        79985271.21620604,
        80004388.07253422,
        80006135.16751514,
        79983524.1212251,
        80026999.11882426,
        79962660.16991599,
        80047863.07013337,
        79941796.21860689,
        80068727.02144249,
        79920932.26729777,
        80089590.9727516,
        79900068.31598866,
        80113005.55130038,
        79875590.4771736,
        200000000.0,
        200005800.0,
        200000000.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200000500.0,
        200005800.0,
        200000500.0,
        200005800.0,
        200000500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200000500.0,
        200005800.0,
        200000500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200001500.0,
        200005800.0,
        200000500.0,
        200005800.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        99471700.0,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
        2.6,
    ]
)
pulse_record_offsets = np.array([0], dtype=np.int64)

dump = seq.decode_pulse_record_flat(pulse_record_flat, pulse_record_offsets)[0]
is_up = dump.is_up
start_times_s = dump.start_times_s
durations_s = dump.durations_s
opll_hz = dump.opll_hz
switch_hz = dump.switch_hz
delivery_hz = dump.delivery_hz
delivery_setpoint = dump.delivery_setpoint

# No shelving-pulse OPLL correction is applied: this is genuine new-format data,
# and the lab recorder (PulseDMARecording.register_pulse in icl_experiments) now
# reports the pulse-CENTRE OPLL (averaged over the pulse). The parked legacy
# dumps recorded the START-of-pulse value and had to be patched by hand for the
# 380 us velocity-selection pulse that fires during the gravity OPLL ramp; with
# centre-of-pulse reporting that correction is already baked in.

# %%
# Manually specify which durations are pi pulses and which are pi/2
is_pi_pulse = lambda d: d > 50000e-9

# %%
# Cut after the launch

t_launch_finished = 5e-3

is_before_launch = start_times_s < t_launch_finished

is_up = is_up[is_before_launch]
start_times_s = start_times_s[is_before_launch]
durations_s = durations_s[is_before_launch]
opll_hz = opll_hz[is_before_launch]
switch_hz = switch_hz[is_before_launch]
delivery_hz = delivery_hz[is_before_launch]
delivery_setpoint = delivery_setpoint[is_before_launch]

# %%
# --- HACKY AUTO-CALIBRATION (a warning is emitted at runtime) ---
# alpha (probe-shift coefficient) and v0 (initial atom velocity) used to be
# hand-tuned magic numbers. Instead we now back them out of the lab pulse dump:
# alpha from how the up-beam pulses sit relative to each other vs their Rabi
# frequency, and v0 from the residual up/down detuning once alpha is applied.
# This is a self-consistent fit, NOT a measurement -- replace with real
# light-shift and launch-velocity calibrations.
# probe_shift_alpha, initial_velocity_z = (
#     seq.calibrate_probe_shift_and_velocity_from_dump(
#         is_up=is_up,
#         start_times_s=start_times_s,
#         durations_s=durations_s,
#         opll_hz=opll_hz,
#         switch_hz=switch_hz,
#         delivery_hz=delivery_hz,
#         delivery_setpoint=delivery_setpoint,
#     )
# )


t_pi_ref = 55e-6


# From measurement - see "2026-06-09 Clock shift gap-filling even Omega2 grid"
# The measurement defined alpha in terms of 1/ (rad/s), so convert to 1/Hz
probe_shift_alpha = -2.04e-6 * (2 * np.pi)
print(f"calibrated probe-shift alpha = {probe_shift_alpha:.4e} Hz^-1")

rabi_freq_ref = 1 / (2 * t_pi_ref)
probe_shift_ref = probe_shift_alpha * rabi_freq_ref**2
print(
    f"corresponding probe shift at Rabi frequency {rabi_freq_ref:.1f} Hz is {probe_shift_ref:.1f} Hz"
)


# %%
# Pin initial_velocity_z so the first DOWN pulse is resonant on its intended
# rung. The parser anchors everything on the first (up) pulse, so the only
# leftover beam-asymmetric term is the 2*v0/lambda Doppler split between the
# counter-propagating beams. Same logic as
# seq.calibrate_probe_shift_and_velocity_from_dump, but holding alpha at the
# measured value above instead of fitting it. (Replaces an old hand-tuned
# FIXME value.)

# N.B. the lab bakes small software detunings into some pulses (e.g.
# double_trap_launch_bs_detuning, lmt_launch_offset_detuning, the up/down
# switch offsets in icl_experiments). These compensate REAL effects in the
# lab -- the recorded frequencies are what was actually resonant -- but the
# compensated effects (e.g. intensity-dependent light shifts beyond the
# alpha*rabi**2 model used here) are not all in the simulation, so this
# pinning absorbs any model mismatch into initial_velocity_z
# (1 kHz <-> 0.35 mm/s).
_, _bare = seq.build_sequence_from_lab_pulse_dump(
    is_up=is_up,
    start_times_s=start_times_s,
    durations_s=durations_s,
    opll_hz=opll_hz,
    switch_hz=switch_hz,
    delivery_hz=delivery_hz,
    delivery_setpoint=delivery_setpoint,
    probe_induced_alpha_up=0.0,
    probe_induced_alpha_down=0.0,
    initial_velocity_z=0.0,
)
_pulses = [e for e in _bare if isinstance(e, seq.Pulse)]
_anchor = _pulses[0]
_first_opposite = next(p for p in _pulses if p.k == -_anchor.k)
_anchor_rabi = _anchor.rabi_frequency
initial_velocity_z = (
    _anchor.k
    * 0.5
    * sim.TRANSITION_WAVELENGTH
    * (
        _anchor.detuning_hz
        - _first_opposite.detuning_hz
        - 4 * sim.RECOIL_FREQUENCY_HZ
        - probe_shift_alpha * (_anchor_rabi**2 - _first_opposite.rabi_frequency**2)
    )
)
print(f"initial_velocity_z = {initial_velocity_z * 1e3:+.3f} mm/s")

# %%
_, sequence = seq.build_sequence_from_lab_pulse_dump(
    is_up=is_up,
    start_times_s=start_times_s,
    durations_s=durations_s,
    opll_hz=opll_hz,
    switch_hz=switch_hz,
    delivery_hz=delivery_hz,
    delivery_setpoint=delivery_setpoint,
    initial_velocity_z=initial_velocity_z,
    probe_induced_alpha_up=probe_shift_alpha,
    probe_induced_alpha_down=probe_shift_alpha,
)


# %%
def plot_lab_pulse_sequence(
    plot_sequence, plot_corrected=True, plot_by_timestamp=False
):
    raw_detunings_recoil = []
    corrected_detunings_recoil = []
    pulse_colours = []
    pulse_hatches = []
    clearout_links = []

    last_pulse_index = None
    clearout_since_last_pulse = False

    PULSE_STYLES = {
        #              colour          hatch
        (True, True): ("tab:orange", ""),  # up,   pi
        (True, False): ("tab:blue", "///"),  # up,   pi/2
        (False, True): ("tab:red", "\\\\"),  # down, pi
        (False, False): ("tab:purple", "xxx"),  # down, pi/2
    }

    timestamps = []
    pulse_durations = []
    now = 0.0
    for event in plot_sequence:
        if isinstance(event, seq.Clearout):
            clearout_since_last_pulse = True
            now += event.duration
            continue

        if not isinstance(event, seq.Pulse):
            now += event.duration
            continue

        timestamps.append(now + event.duration / 2)
        pulse_durations.append(event.duration)
        now += event.duration

        pulse_index = len(raw_detunings_recoil)
        raw_detunings_recoil.append(event.detuning_hz / sim.RECOIL_FREQUENCY_HZ)
        probe_shift_hz = (
            event.probe_shift_coefficient * event.effective_stark_rabi_frequency**2
        )
        corrected_detunings_recoil.append(
            (event.detuning_hz - probe_shift_hz) / sim.RECOIL_FREQUENCY_HZ
        )

        colour, hatch = PULSE_STYLES[(event.k == 1, is_pi_pulse(event.duration))]
        pulse_colours.append(colour)
        pulse_hatches.append(hatch)

        if last_pulse_index is not None and clearout_since_last_pulse:
            clearout_links.append((last_pulse_index, pulse_index))

        last_pulse_index = pulse_index
        clearout_since_last_pulse = False

    pulse_indices = np.arange(len(raw_detunings_recoil))

    fig, ax = plt.subplots(figsize=(18, 10))
    plot_detunings = (
        corrected_detunings_recoil if plot_corrected else raw_detunings_recoil
    )

    x_axis = timestamps if plot_by_timestamp else pulse_indices

    bars = ax.bar(
        x_axis,
        plot_detunings,
        color=pulse_colours,
        width=pulse_durations if plot_by_timestamp else 0.9,
    )
    for bar, hatch in zip(bars, pulse_hatches):
        bar.set_hatch(hatch)

    for left_index, right_index in clearout_links:
        ax.plot(
            [x_axis[left_index], x_axis[right_index]],
            [
                plot_detunings[left_index],
                plot_detunings[right_index],
            ],
            color="tab:green",
            linestyle=":",
            linewidth=2,
            zorder=3,
        )

    if not plot_by_timestamp:
        tick_step = max(1, len(pulse_indices) // 20)
        ax.set_xticks(pulse_indices[::tick_step])
        ax.set_xlabel("Pulse index")
    else:
        ax.set_xlabel("Time (s)")
    ax.set_ylabel("Detuning (recoil frequencies)")

    # Show where the uncorrected (raw) detuning sat, for comparison.
    if plot_corrected:
        ax.scatter(
            x_axis,
            raw_detunings_recoil,
            facecolor="none",
            edgecolor="0.2",
            marker="o",
            s=40,
            zorder=4,
        )

    # Horizontal gridlines on the odd integers only.
    y_min, y_max = ax.get_ylim()
    odd_integers = np.arange(np.ceil(y_min), np.floor(y_max) + 1)
    odd_integers = odd_integers[odd_integers % 2 != 0]
    for y in odd_integers:
        ax.axhline(y, color="0.5", linestyle=":", alpha=0.8, zorder=0)
    ax.axhline(0.0, color="0.5", linestyle="-", alpha=0.8, zorder=0)

    legend_handles = [
        Patch(facecolor="tab:orange", hatch="", label="up, pi"),
        Patch(facecolor="tab:blue", hatch="///", label="up, pi/2"),
        Patch(facecolor="tab:red", hatch="\\\\", label="down, pi"),
        Patch(facecolor="tab:purple", hatch="xxx", label="down, pi/2"),
    ]
    if clearout_links:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="tab:green",
                linestyle=":",
                linewidth=2,
                label="clearout between pulses",
            )
        )

    ax.legend(handles=legend_handles, loc="upper left", frameon=False)


# %%
plot_lab_pulse_sequence(sequence, plot_corrected=True)
plt.title("Debug output")
vs.tag_plot(small=True)
# %%
plot_lab_pulse_sequence(sequence, plot_corrected=True, plot_by_timestamp=True)
plt.title("Corrected pulse frequencies")
vs.tag_plot(small=True)
# %%
seq.compute_spacetime_trajectory(sequence, plot=True, max_branches=20)

# %%
# Apply a hacky correction - round each pulse to the nearest corrected integer number of recoils
import dataclasses

bodged_sequence = []
for event in sequence:
    if not isinstance(event, seq.Pulse):
        bodged_sequence.append(event)
        continue

    probe_shift_hz = (
        event.probe_shift_coefficient * event.effective_stark_rabi_frequency**2
    )

    closest_integer = int(
        round((event.detuning_hz - probe_shift_hz) / sim.RECOIL_FREQUENCY_HZ)
    )

    # Re-add the probe shift so that the EFFECTIVE detuning (the sim subtracts
    # the shift again during the pulse) lands exactly on the recoil ladder.
    new_pulse = dataclasses.replace(
        event,
        detuning_hz=closest_integer * sim.RECOIL_FREQUENCY_HZ + probe_shift_hz,
    )
    bodged_sequence.append(new_pulse)


# %%
seq.compute_spacetime_trajectory(bodged_sequence, plot=True, max_branches=20)
