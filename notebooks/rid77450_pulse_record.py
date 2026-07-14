"""RID 77450 pulse record -- shared data module, not an analysis notebook.

The genuine float64 SI ``pulse_record_flat`` emitted by ``PulseDMARecording``
for ARTIQ RID 77450: the RID 76695 symmetric Mach-Zehnder re-run with
identical parameters (``n_launch = 0``, ``n_recoils = 2``, up-pi 56 us,
down-pi 67 us, slice 380 us, phi = 0) on the fixed ``declarative-lmt`` branch
(icl_experiments PR #86), lasers off / no atoms -- captured purely to record
the corrected pulse program. Its two down-beam beamsplitters (pulses 1 and 19)
carry a genuine down-pi/2 = 33.5 us duration, so the sequence is a real
Mach-Zehnder.

Kept in one place so every notebook analysing this record shares a single
copy. Import from a notebook (they run with ``notebooks/`` as the working
directory) as::

    import rid77450_pulse_record as rid77450

    dump = rid77450.load_dump()
    alpha, v0, timestamps, sequence = rid77450.calibrate_and_build()

See ``population_leakage_by_m_rid77450.py`` for the full story of why this
re-run exists and what the record contains.
"""

import dataclasses
import sys
import warnings

sys.path.insert(0, "..")

import numpy as np

import lmt_sim.lmt_sequence as seq

# The 8th row (per-pulse interferometry phase) is all zeros: this is the
# interferometer_phase = 0 point of the scan.
PULSE_RECORD_FLAT = np.array(
    [
        20.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0011392000000000002,
        0.002142592,
        0.002197812,
        0.002475548,
        0.002603268,
        0.002691988,
        0.0028307080000000004,
        0.0031194440000000003,
        0.003258164,
        0.0033358840000000003,
        0.003416128,
        0.003504848,
        0.0037825840000000003,
        0.003910304,
        0.003999024,
        0.0041377440000000005,
        0.00442648,
        0.0045652,
        0.00464292,
        0.004723168000000001,
        0.000379999,
        3.35e-05,
        5.5999000000000004e-05,
        5.5999000000000004e-05,
        6.7e-05,
        6.7e-05,
        6.7e-05,
        6.7e-05,
        5.5999000000000004e-05,
        5.5999000000000004e-05,
        6.7e-05,
        5.5999000000000004e-05,
        5.5999000000000004e-05,
        6.7e-05,
        6.7e-05,
        6.7e-05,
        6.7e-05,
        5.5999000000000004e-05,
        5.5999000000000004e-05,
        3.35e-05,
        80014038.33602092,
        79984677.23407157,
        80009139.72161916,
        80031843.41483626,
        79996777.9422243,
        79957924.06945309,
        79993584.53084627,
        79951922.29819915,
        80024027.78063032,
        80043923.11397147,
        79966560.74623615,
        80027491.3907797,
        80050195.08399677,
        79978426.27306378,
        79939572.40029258,
        79975232.86168575,
        79933570.6290386,
        80042379.44979084,
        80062274.78313199,
        79948444.20223802,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        200000000.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        99426200.0,
        0.012,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)
PULSE_RECORD_OFFSETS = np.array([0], dtype=np.int64)


def load_dump():
    """Decode the record into a single :class:`lmt_sim.lmt_sequence.LabPulseDump`."""
    return seq.decode_pulse_record_flat(PULSE_RECORD_FLAT, PULSE_RECORD_OFFSETS)[0]


def calibrate_and_build(dump=None):
    """Calibrate and build the simulation sequence for this record.

    Runs the (self-consistent, loudly-warning -- suppressed here for notebook
    readability) ``calibrate_probe_shift_and_velocity_from_dump`` fit and
    feeds the result into ``build_sequence_from_lab_pulse_dump``, which folds
    the free-fall Doppler ramp into the recorded detunings so the quantum run
    is done in the freely-falling frame.

    Returns
    -------
    (alpha, v0, timestamps, sequence)
        Probe-shift coefficient (1/Hz), fitted initial z-velocity (m/s), the
        per-event start times, and the event sequence.
    """
    if dump is None:
        dump = load_dump()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        alpha, v0 = seq.calibrate_probe_shift_and_velocity_from_dump(
            **dataclasses.asdict(dump)
        )
    timestamps, sequence = seq.build_sequence_from_lab_pulse_dump(
        **dataclasses.asdict(dump),
        probe_induced_alpha_up=alpha,
        probe_induced_alpha_down=alpha,
        initial_velocity_z=v0,
    )
    return alpha, v0, timestamps, sequence


def intended_arm_trajectories(dump, force_bs_pi2, flip_threshold=0.75):
    """Intended MZ arms: list of dicts {pulse_index: (m, is_ground)}.

    Walks an ideal on-axis atom through the sequence, splitting at π/2 and
    flipping at π (the `compute_spacetime_trajectory` rule). If ``force_bs_pi2``
    the two beamsplitters (pulses 1 and n-1) are set to a down-π/2 duration
    first, so the intended split is shown even for a record in which they fired
    as full π (RID 76695); for a record that already holds π/2 beamsplitters
    (RID 77450) pass ``force_bs_pi2=False``.

    Shared here (rather than living in one notebook) because both the leakage
    map (intended-path overlay) and the realistic-readout notebook (clearout
    window placement) need the same walk.
    """
    d = dump
    n = len(d.is_up)
    if force_bs_pi2:
        dur = np.array(d.durations_s)
        for b in (1, n - 1):
            dur[b] = 33.5e-6  # down-π/2 = half the 67 µs down-π
        d = dataclasses.replace(d, durations_s=dur)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        alpha, v0 = seq.calibrate_probe_shift_and_velocity_from_dump(
            **dataclasses.asdict(d)
        )
    _t, sequence = seq.build_sequence_from_lab_pulse_dump(
        **dataclasses.asdict(d),
        probe_induced_alpha_up=alpha,
        probe_induced_alpha_down=alpha,
        initial_velocity_z=v0,
    )
    pulses = [e for e in sequence if isinstance(e, seq.Pulse)]
    arms = [{"m": 0, "g": True, "hist": {}, "cid": 0}]
    first_split_done = False
    for p, pl in enumerate(pulses):
        nxt = []
        for a in arms:
            pr = seq._transition_probability(a["m"], a["g"], pl)
            if pr >= flip_threshold:
                a["m"] += pl.k if a["g"] else -pl.k
                a["g"] = not a["g"]
                a["hist"][p] = (a["m"], a["g"])
                nxt.append(a)
            elif pr <= 1 - flip_threshold:
                a["hist"][p] = (a["m"], a["g"])
                nxt.append(a)
            else:  # π/2 → split into a drifter and a flipper
                # The input splitter (bs1) creates the two interferometer arms
                # (distinct colours); the recombiner's forks inherit their arm.
                b_cid = 1 if not first_split_done else a["cid"]
                first_split_done = True
                b = {"m": a["m"], "g": a["g"], "hist": dict(a["hist"]), "cid": b_cid}
                a["hist"][p] = (a["m"], a["g"])
                b["m"] += pl.k if b["g"] else -pl.k
                b["g"] = not b["g"]
                b["hist"][p] = (b["m"], b["g"])
                nxt.extend([a, b])
        arms = nxt
    return [(a["hist"], a["cid"]) for a in arms]


def both_arms_excited_after_pulses(dump, force_bs_pi2=False):
    """Pulse indices after which EVERY intended arm sits in the excited state.

    These are the only windows where a perfect 461 nm ground clearout can fire
    without destroying interferometer population: everything the sequence means
    to keep is shelved in the excited state, so removing the ground manifold
    only removes junk (unselected cloud, leaked population). Before the first
    beamsplitter there is a single arm, so this includes the classic
    post-velocity-selection clearout window after pulse 0.
    """
    arms = intended_arm_trajectories(dump, force_bs_pi2=force_bs_pi2)
    n_pulses = len(dump.is_up)
    windows = []
    for p in range(n_pulses):
        # Distinct arm states after pulse p (pre-fork copies share history).
        states = {hist[p] for hist, _cid in arms if p in hist}
        if states and all(not is_ground for _m, is_ground in states):
            windows.append(p)
    return windows
