r"""Velocity-resolved excited-state readout: a swept clock "imaging" pulse.

The lab cannot project an atom directly onto the ``(m, internal)`` basis the
way ``calculate_ground_and_excited_probabilities`` does. What it CAN do is the
two-frame clock-shelving detection:

1. **Ground blow-away.** A 461 nm imaging flash counts -- and removes -- every
   ground-state atom. It is broadband on the recoil scale, so it carries NO
   momentum resolution: the entire ground manifold disappears as one number.
2. **Clock imaging pulse.** A narrow 698 nm pulse at laser detuning
   ``delta`` transfers the one excited-state velocity class that is resonant,
   ``|e, m>  ->  |g, m - k>``, back to the ground state. Choosing the pulse as
   long as the original velocity-selection ("slicing") pulse makes its Rabi
   linewidth much smaller than one photon recoil.
3. **Second imaging flash.** Counts the freshly transferred ground atoms:
   the number of *excited* atoms in the addressed velocity class.

Each experimental shot therefore yields the excited-state population of a
single velocity class; sweeping ``delta`` shot-to-shot builds up the
velocity-resolved excited-state spectrum. The ground-state momentum
distribution is invisible to this scheme.

This module models exactly that observable for a simulated ``AtomState``:

- :func:`remove_ground_rows` is step 1 -- an (ensemble-averaged) projective
  removal of the ground manifold. Deliberately NOT renormalised: the readout
  measures absolute populations, and blowing away ground atoms does not make
  the surviving excited atoms any brighter.
- :func:`simulate_excited_state_readout` is steps 2-3, swept over a detuning
  grid. The imaging pulse is applied with the canonical
  :func:`~lmt_sim.lmt_simulation.pulse_interaction_in_borde_representation`
  (single source of truth for pulse physics), and the signal is the population
  arriving in each ground momentum class.

Two exactness notes:

- **Row merging.** For a spatially uniform readout beam the 2x2 propagator of
  the imaging pulse depends on a row only through ``(m, internal)``, so all
  rows sharing a class can be merged coherently (amplitudes summed) before the
  sweep with NO approximation -- the transferred amplitude
  ``sum_rows C * c_row = C * sum_rows c_row`` is identical. This is what makes
  sweeping hundreds of detunings over multi-thousand-row states affordable.
- **Delay independence.** Free evolution between the last sequence pulse and
  the readout multiplies all rows of one ``(m, internal)`` class by the same
  phase. With the ground manifold removed, each output ground class ``m - k``
  receives amplitude from exactly ONE excited class ``m``, so those per-class
  phases never interfere and the signal does not depend on when the readout
  fires. (Positions do change, but a uniform beam does not see them.) The
  free-fall Doppler ramp during that delay is the lab's problem, handled by
  quoting detunings Doppler-adjusted; see the notes in
  ``build_sequence_from_lab_pulse_dump``.

Frame safety: the signal is computed from ``|amplitude|**2`` per ``(m,
internal)`` class, which is invariant under the Bordé<->lab transform (all
rows of a class share the same transform phase), and a laser-frequency step
does not touch Bordé amplitudes (see
``change_laser_frequency_in_borde_representation``). The readout can therefore
be applied directly to a final Bordé-frame state at any sweep detuning.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import lmt_sim.lmt_simulation as sim


def remove_ground_rows(state: sim.AtomState) -> sim.AtomState:
    """Blow away the ground manifold (the first imaging flash).

    Keeps only excited-state rows. This is an incoherent, ensemble-averaged
    projective removal: any ground/excited coherence is destroyed by the
    photon scattering that images the ground atoms.

    The result is deliberately NOT renormalised -- the readout measures
    absolute atom numbers, so the surviving excited population must keep its
    weight ``P_e < 1``. (Contrast :func:`~lmt_sim.lmt_simulation.do_clearout`,
    which Monte-Carlo samples one atom's fate and renormalises the survivor.)
    """
    keep = ~state.internal_is_ground
    return sim.AtomState(
        m_values=state.m_values[keep],
        positions=state.positions[keep],
        velocities=state.velocities[keep],
        amplitudes=state.amplitudes[keep],
        internal_is_ground=state.internal_is_ground[keep],
        # Removing rows does not change the laser frame.
        t_ref=state.t_ref,
        detuning_ref_hz=state.detuning_ref_hz,
        accumulated_detuning_cycles=state.accumulated_detuning_cycles,
    )


def merge_rows_by_momentum_class(state: sim.AtomState) -> sim.AtomState:
    """Coherently merge all rows sharing the same ``(m, internal)`` class.

    Amplitudes within a class are summed (coherently -- this is the same sum
    ``calculate_ground_and_excited_probabilities`` squares). Velocities are
    identical within a class by the ``v = v_0 + m * v_recoil`` invariant and
    are carried over; positions are collapsed to the ``|amplitude|**2``
    -weighted mean and become bookkeeping only.

    This is EXACT for any subsequent spatially uniform interaction (the 2x2
    pulse propagator depends on a row only through ``(m, internal)``), and
    wrong for anything position-dependent (Gaussian-beam pulses, imaging) --
    do not feed a merged state into those.
    """
    if len(state.m_values) == 0:
        return state

    # Stable class key: (m, is_ground). np.unique on the stacked key gives one
    # row per class plus the inverse map used to accumulate the sums.
    key = np.column_stack((state.m_values, state.internal_is_ground.astype(int)))
    unique_keys, inverse = np.unique(key, axis=0, return_inverse=True)
    n_classes = len(unique_keys)

    amplitudes = np.zeros(n_classes, dtype=state.amplitudes.dtype)
    np.add.at(amplitudes, inverse, state.amplitudes)

    weights = np.abs(state.amplitudes) ** 2
    class_weight = np.zeros(n_classes)
    np.add.at(class_weight, inverse, weights)

    positions = np.zeros((n_classes, 3))
    np.add.at(positions, inverse, state.positions * weights[:, None])
    # A class whose every row has zero amplitude gets the plain mean instead
    # of 0/0.
    counts = np.zeros(n_classes)
    np.add.at(counts, inverse, 1.0)
    plain_mean = np.zeros((n_classes, 3))
    np.add.at(plain_mean, inverse, state.positions)
    plain_mean /= counts[:, None]
    safe = class_weight > 0
    positions[safe] /= class_weight[safe, None]
    positions[~safe] = plain_mean[~safe]

    # Velocities are identical within a class; take each class's first row.
    _, first_row = np.unique(inverse, return_index=True)
    velocities = state.velocities[first_row]

    return sim.AtomState(
        m_values=unique_keys[:, 0].astype(state.m_values.dtype),
        positions=positions,
        velocities=velocities,
        amplitudes=amplitudes,
        internal_is_ground=unique_keys[:, 1].astype(bool),
        t_ref=state.t_ref,
        detuning_ref_hz=state.detuning_ref_hz,
        accumulated_detuning_cycles=state.accumulated_detuning_cycles,
    )


def run_sequence_with_ground_clearouts(
    pulse_sequence,
    clearout_after_pulses,
    *,
    initial_velocity_z=0.0,
    discard_threshold=1e-9,
    c0=1.0,
    c1=0.0,
    weight_floor=1e-12,
):
    """Run a pulse sequence with perfect, instantaneous ground clearouts.

    ``clearout_after_pulses`` lists pulse indices (0-based, counting only
    ``Pulse``/``CompositePulse`` events); immediately after each listed pulse
    the entire ground manifold is removed, modelling an ideal 461 nm clearout
    fired in a window where everything the sequence means to keep is shelved
    in the excited state.

    This is the deterministic, ensemble-averaged counterpart of
    :func:`~lmt_sim.lmt_simulation.do_clearout` (which Monte-Carlo samples one
    atom's fate and renormalises the survivor): here the ground rows are
    dropped with their weight, the surviving amplitudes are renormalised so
    the canonical sequence loop can continue, and the removed weight is
    accumulated into a single ``weight`` factor. Quadratic quantities computed
    from the returned state (readout signals, per-class populations) multiplied
    by ``weight`` are exactly those of the projected state evolved through the
    full sequence.

    Exactness note: the sequence is run in segments through the canonical
    ``run_pulse_sequence_in_borde_representation``, which restarts the Bordé
    frame bookkeeping (and evaluates any free fall at the start of a segment
    at that segment's first-pulse detuning). Both are harmless here: frame
    bookkeeping never touches amplitudes, and because the post-clearout state
    is entirely excited, the detuning-dependent part of the free-fall phase is
    a global phase. Populations therefore match an unsegmented run exactly.

    Parameters
    ----------
    pulse_sequence : sequence of events
        As for ``run_pulse_sequence_in_borde_representation``. Must not
        contain ``Clearout`` events (this function IS the clearout model).
    clearout_after_pulses : iterable of int
        Pulse indices after which to fire the clearout.
    initial_velocity_z, discard_threshold, c0, c1 :
        As for the canonical sequence runners.
    weight_floor : float, optional
        If the surviving weight falls below this, the atom is treated as fully
        removed and ``(None, 0.0)`` is returned.

    Returns
    -------
    (AtomState or None, float)
        The final state (unit row norm, ready for
        :func:`simulate_excited_state_readout`) and the surviving weight, i.e.
        the fraction of the original atom number still present at detection.
        ``(None, 0.0)`` if every clearout survivor fell below ``weight_floor``.
    """
    # Late import: lmt_sequence imports this package's sibling module chain via
    # lmt_sim, so import locally to keep module import order flexible.
    from lmt_sim import lmt_sequence as seq

    clearout_after_pulses = set(clearout_after_pulses)

    # Split into (segment, clearout_after_segment) pairs: a segment ends
    # immediately after each clearout pulse, so the clearout fires between it
    # and the following event (before any free fall).
    segments = [([], False)]
    pulse_index = -1
    for event in pulse_sequence:
        if isinstance(event, seq.Clearout):
            raise ValueError(
                "pulse_sequence must not contain Clearout events; pass the "
                "clearout positions via clearout_after_pulses instead"
            )
        segments[-1][0].append(event)
        if isinstance(event, (seq.Pulse, seq.CompositePulse)):
            pulse_index += 1
            if pulse_index in clearout_after_pulses:
                segments[-1] = (segments[-1][0], True)
                segments.append(([], False))
    missing = {p for p in clearout_after_pulses if p > pulse_index}
    if missing:
        raise ValueError(
            f"clearout_after_pulses {sorted(missing)} exceed the last pulse "
            f"index {pulse_index}"
        )
    if not segments[-1][0]:
        segments.pop()

    state = sim.make_atom_states(c0=c0, c1=c1, initial_velocity_z=initial_velocity_z)
    weight = 1.0
    for segment, clearout_after in segments:
        state, _detuning, _t = seq.run_pulse_sequence_in_borde_representation(
            state,
            segment,
            initial_velocity_z=initial_velocity_z,
            discard_threshold=discard_threshold,
        )
        if not clearout_after:
            continue
        state = remove_ground_rows(state)
        survived = float(np.sum(np.abs(state.amplitudes) ** 2))
        weight *= survived
        if weight <= weight_floor:
            return None, 0.0
        state = replace(state, amplitudes=state.amplitudes / np.sqrt(survived))
    return state, weight


def readout_resonance_detuning_hz(m_excited, k_sign=+1, v0=0.0):
    """Laser detuning (Hz) at which the imaging pulse resonantly drives
    ``|e, m_excited> -> |g, m_excited - k_sign>``.

    From the Bordé resonance condition (``Omega_3 = 0`` in
    ``_borde_frame_constants`` with ``m_ground = m_excited - k_sign``):

    ``delta = k_sign * v0 / lambda + (2 * m_excited * k_sign - 1) * delta_rec``

    i.e. for the up beam (+k) the excited classes sit at ODD multiples of the
    recoil frequency, ``(2 m - 1) * delta_rec``, spaced by two recoils
    (~9.4 kHz for Sr-87 698 nm), plus the Doppler shift of the atom's base
    velocity ``v0``; the down beam mirrors the ladder,
    ``(-2 m - 1) * delta_rec``. Detunings are in the same Doppler-adjusted
    convention as the rest of the simulation (free-fall ramp folded out). The
    probe (light) shift of the imaging pulse itself is not included -- add
    ``probe_shift_coefficient * rabi**2`` if it matters.

    ``m_excited`` may be an array.
    """
    m_excited = np.asarray(m_excited)
    return (
        k_sign * v0 / sim.TRANSITION_WAVELENGTH
        + (2 * m_excited * k_sign - 1) * sim.RECOIL_FREQUENCY_HZ
    )


def simulate_excited_state_readout(
    state: sim.AtomState,
    detunings_hz,
    *,
    pulse_duration,
    pulse_rabi_frequency,
    k_sign=+1,
    vz=0.0,
    probe_shift_coefficient=0.0,
):
    """Sweep the clock imaging pulse over one atom's final state.

    For each sweep detuning the two-frame readout described in the module
    docstring is simulated: ground manifold removed, imaging pulse applied
    (uniform beam), and the population arriving in each ground momentum class
    counted. The returned signal is per EXCITED source class ``m``: with the
    ground manifold gone, ground class ``m - k_sign`` is fed only by excited
    class ``m``, so the decomposition is exact, and summing over classes gives
    the total number the second imaging flash would see.

    Parameters
    ----------
    state : AtomState
        Final (Bordé-frame) state of one atom after the pulse sequence, e.g.
        from ``run_pulse_sequence_in_borde_representation``. May contain
        ground rows; they are removed here (they are invisible to this
        readout).
    detunings_hz : array-like, shape (n_det,)
        Sweep grid of imaging-pulse laser detunings, in the simulation's
        Doppler-adjusted convention (Hz).
    pulse_duration : float
        Imaging pulse duration in seconds (typically the slicing-pulse
        duration, for matched sub-recoil resolution).
    pulse_rabi_frequency : float
        Imaging pulse Rabi frequency in Hz (``1 / (2 * duration)`` for a pi
        pulse).
    k_sign : int, optional
        Imaging beam direction (+1 up, -1 down), by default +1.
    vz : float, optional
        The atom's base z-velocity ``v_0`` (the same reference passed to the
        sequence run), by default 0.0.
    probe_shift_coefficient : float, optional
        Probe (light) shift coefficient of the imaging pulse in 1/Hz, by
        default 0.0.

    Returns
    -------
    m_classes : ndarray of int, shape (n_m,)
        The excited momentum classes present in ``state`` (sorted).
    signal : ndarray, shape (n_det, n_m)
        ``signal[i, j]`` is the probability of finding the atom in ground
        class ``m_classes[j] - k_sign`` after an imaging pulse at
        ``detunings_hz[i]``, i.e. the readout signal contributed by excited
        class ``m_classes[j]``. ``signal.sum(axis=1)`` is the total detected
        signal per shot (in units of probability per atom).
    """
    detunings_hz = np.asarray(detunings_hz, dtype=float)

    excited = merge_rows_by_momentum_class(remove_ground_rows(state))
    # Merged classes are unique and sorted (np.unique), so ground output class
    # m - k_sign identifies its excited source class m unambiguously.
    m_classes = excited.m_values.copy()
    signal = np.zeros((len(detunings_hz), len(m_classes)))
    if len(m_classes) == 0:
        return m_classes, signal

    for i, delta in enumerate(detunings_hz):
        after = sim.pulse_interaction_in_borde_representation(
            excited,
            pulse_detuning=float(delta),
            t_pulse=pulse_duration,
            pulse_rabi_freq=pulse_rabi_frequency,
            k_sign=k_sign,
            vz=vz,
            probe_shift_coefficient=probe_shift_coefficient,
        )
        is_ground = after.internal_is_ground
        ground_m = after.m_values[is_ground]
        ground_pop = np.abs(after.amplitudes[is_ground]) ** 2
        source_m = ground_m + k_sign
        idx = np.searchsorted(m_classes, source_m)
        signal[i, idx] = ground_pop

    return m_classes, signal
