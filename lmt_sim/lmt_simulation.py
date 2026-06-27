######

import logging
from dataclasses import dataclass, replace

import numpy as np
from scipy import constants

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


######

MASS_ATOM = constants.atomic_mass * 87
TRANSITION_WAVELENGTH = 698e-9
RABI_FREQ = 1 / (2 * 45e-6)
T_PI = 1 / (2 * RABI_FREQ)

TRANSITION_FREQUENCY = constants.c / TRANSITION_WAVELENGTH
K_WAVEVECTOR = 2 * np.pi / TRANSITION_WAVELENGTH
RECOIL_VELOCITY = constants.hbar * K_WAVEVECTOR / MASS_ATOM

# Recoil frequency: delta_rec = hbar * k^2 / (2 * M), expressed in Hz
# This is Bordé's "delta" = hbar*k^2 / (2M).
# In our convention where all frequencies are in Hz (not angular),
# we need this in Hz: delta_rec_hz = hbar * k^2 / (2 * M) / (2*pi)
# But since we use the factor-of-pi convention (H/hbar = pi * ...),
# we keep it in the same units.  The recoil shift enters the detuning
# as (2m ± 1) * RECOIL_FREQUENCY_HZ.
RECOIL_FREQUENCY_HZ = constants.hbar * K_WAVEVECTOR**2 / (2 * MASS_ATOM) / (2 * np.pi)

GRAVITY_G = constants.g
GRAVITY_DOPPLER_PER_SEC_HZ = TRANSITION_FREQUENCY * constants.g / constants.c


@dataclass(frozen=True)
class AtomState:
    m_values: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    amplitudes: np.ndarray
    internal_is_ground: np.ndarray
    # Reference for the laser-phase part of the Bordé (Eq. 4) transform.
    #
    # The Bordé frame co-rotates with the laser, so the lab<->Bordé transform
    # depends on the integral of the laser detuning, ``Phi(t) = integral_0^t
    # delta(t') dt'`` (Hz*s = cycles). We store that piecewise-linear integral as
    # its current segment: ``detuning_ref_hz`` (Hz) is the laser detuning since ``t_ref``
    # (s), and ``accumulated_detuning_cycles`` is ``Phi(t_ref)`` -- the integral
    # over all *closed* (earlier) segments. So ``Phi(t) = accumulated_detuning_cycles
    # + detuning_ref_hz * (t - t_ref)`` for any ``t`` in the current segment.
    #
    # IMPORTANT: the amplitudes are kept in the *instantaneous* Bordé frame and are
    # NOT touched when the laser frequency steps (the laser phase is continuous, so
    # the instantaneous frames coincide at the step -- see
    # docs/arp_frame_change_finding.md). The frame-change phase is held entirely in
    # ``accumulated_detuning_cycles`` and applied only at the lab boundary by
    # ``transform_state_vector``. Folding it into the amplitudes instead would
    # corrupt any subsequent pulse (a non-diagonal op acting on a rotated g/e phase)
    # -- the very bug this design removes.
    t_ref: float = 0.0
    detuning_ref_hz: float = 0.0
    accumulated_detuning_cycles: float = 0.0


def make_atom_states(
    position_x=0.0,
    position_y=0.0,
    position_z=0.0,
    velocity_x=0.0,
    velocity_y=0.0,
    initial_velocity_z=0.0,
    c0=1.0,
    c1=0.0,
):
    """Make the initial state for an atom with the given parameters.

    Parameters
    ----------
    position_x : float
        Initial x-position of the atom in metres.
    position_y : float
        Initial y-position of the atom in metres.
    position_z : float
        Initial z-position of the atom in metres.
    velocity_x : float
        Transverse x-velocity in m/s. Constant -- never changed by pulses.
    velocity_y : float
        Transverse y-velocity in m/s. Constant -- never changed by pulses.
    initial_velocity_z : float
        Initial z-velocity of the atom in m/s.  This is the "lab frame"
        velocity v_0; the total z-velocity of branch m is v_0 + m * v_recoil.
        vz is updated by recoil kicks during pulses.
    c0 : complex
        Initial ground-state amplitude.
    c1 : complex
        Initial excited-state amplitude.

    Returns
    -------
    AtomState
        Initial atom state.
    """
    m_values = np.array([0, 0], dtype=int)
    positions = np.array(
        [
            [position_x, position_y, position_z],
            [position_x, position_y, position_z],
        ],
        dtype=np.float64,
    )
    velocities = np.array(
        [
            [velocity_x, velocity_y, initial_velocity_z],
            [velocity_x, velocity_y, initial_velocity_z],
        ],
        dtype=np.float64,
    )

    internal_amplitude = np.array([c0, c1], dtype=np.complex128)
    internal_is_ground = np.array([True, False], dtype=bool)

    return AtomState(
        m_values=m_values,
        positions=positions,
        velocities=velocities,
        amplitudes=internal_amplitude,
        internal_is_ground=internal_is_ground,
        # Fresh lab state at t=0; the laser detuning of the Bordé frame is
        # established at the lab->Bordé boundary in run_pulse_sequence_in_lab_frame.
        t_ref=0.0,
        detuning_ref_hz=0.0,
        accumulated_detuning_cycles=0.0,
    )


def transform_state_vector(
    state: AtomState,
    detuning_hz,
    t,
    z,
    vz,
    t_ref=0.0,
    accumulated_detuning_cycles=0.0,
    k=K_WAVEVECTOR,
    omega_0=2 * np.pi * TRANSITION_FREQUENCY,
    inverse=False,
):
    """
    Transform a state vector to/from the position & time independent frame.

    Eq. 4 of Bordé's paper.

    The Bordé frame co-rotates with the laser, so this transformation depends on
    the integral of the laser detuning,
    ``Phi(t) = accumulated_detuning_cycles + detuning_hz * (t - t_ref)`` (in
    cycles), where ``accumulated_detuning_cycles`` is the integral over earlier
    (closed) constant-detuning segments and ``detuning_hz`` is the detuning held
    since ``t_ref``. The laser-dependent part of the transform is then
    ``exp(-/+ i * pi * Phi(t))`` (ground -, excited +). The transition-frequency
    (``omega_0``) global/internal phase and the spatial ``m * k * (z + vz * t)``
    term carry NO laser-frequency dependence and stay on **absolute** ``t``.

    Writing the laser angular frequency as ``omega_laser = omega_0 + 2*pi*delta``,
    the laser-dependent angle ``omega_laser * t`` of the old (fixed-frequency)
    formulation becomes ``omega_0 * t + 2*pi * Phi(t)``: the ``omega_0`` piece is
    kept on absolute ``t`` (it is the fixed transition frequency) and the detuning
    piece is the genuine integral ``Phi(t)``. With
    ``accumulated_detuning_cycles = 0`` and ``t_ref = 0`` this reduces exactly to
    the old ``omega_laser * t`` behaviour. Using the integral (rather than
    ``delta * t``) is what lets the laser frequency step between pulses without the
    "assume current frequency since t=0" approximation; the closed-segment integral
    is accumulated by ``change_laser_frequency_in_borde_representation`` (which does
    NOT touch the amplitudes). See docs/arp_frame_change_finding.md.
    """
    global_phase = np.exp(1j * omega_0 / 2 * t)

    # Laser-dependent angle: omega_0 on absolute t, detuning as the genuine
    # integral Phi(t) (closed segments + current segment).
    phi_cycles = accumulated_detuning_cycles + detuning_hz * (t - t_ref)
    laser_angle = omega_0 * t + 2 * np.pi * phi_cycles

    m_dependent_phase_gnd = np.exp(
        1j / 2 * (-laser_angle - 2 * state.m_values * k * (z + vz * t))
    )
    m_dependent_phase_excited = np.exp(
        1j / 2 * (laser_angle - 2 * state.m_values * k * (z + vz * t))
    )
    m_dependent_phase = np.where(
        state.internal_is_ground, m_dependent_phase_gnd, m_dependent_phase_excited
    )

    transform = global_phase * m_dependent_phase

    if inverse:
        transform = np.conj(transform)

    return replace(state, amplitudes=transform * state.amplitudes)


def _effective_detuning_hz(detuning_hz, probe_shift_coefficient, rabi_freq):
    """Laser detuning corrected for the intensity-dependent probe (light) shift.

    The shift scales with intensity, i.e. ``rabi_freq**2``. It is subtracted
    because the recorded detuning sits above the bare resonance by this amount
    (the lab tunes the laser up to compensate the light shift).
    """
    return detuning_hz - probe_shift_coefficient * rabi_freq**2


def _borde_frame_constants(
    detuning_hz,
    k_sign=+1,
    k=K_WAVEVECTOR,
    vz=0.0,
    m_ground=0,
):
    """Single source of truth for the Bordé recoil-shifted detunings (Eqs. 7-8).

    Shared by free-fall propagation and pulse interaction so the recoil-shift
    formula lives in exactly one place.
    """
    Delta = 2 * np.pi * detuning_hz
    delta_recoil = constants.hbar * k**2 / (2 * MASS_ATOM)

    # Equation 7: Omega_3 = Delta - k_sign*k*vz - [(m+k_sign)^2 - m^2]*delta_recoil
    Omega_3 = (
        Delta
        - k_sign * k * vz
        - ((m_ground + k_sign) ** 2 - m_ground**2) * delta_recoil
    )

    # Equation 8: Omega_0_val = -[(m+k_sign)^2 + m^2]*delta_recoil - (2*m + k_sign)*k*vz
    Omega_0_val = (
        -((m_ground + k_sign) ** 2 + m_ground**2) * delta_recoil
        - (2 * m_ground + k_sign) * k * vz
    )

    return Delta, delta_recoil, Omega_0_val, Omega_3


def _calculate_interaction_constants(
    detuning_hz,
    t_pulse,
    omega_ab,
    k_sign=+1,
    k=K_WAVEVECTOR,
    vz=0.0,
    m_ground=0,
):
    _, _, _, Omega_3 = _borde_frame_constants(
        detuning_hz, k_sign=k_sign, k=k, vz=vz, m_ground=m_ground
    )

    # Eq 12: Generalized Rabi frequency
    Omega = np.sqrt(Omega_3**2 + 4 * omega_ab**2)

    # Equation 13: Matrix elements
    A = np.cos(Omega * t_pulse / 2) + 1j * Omega_3 / Omega * np.sin(Omega * t_pulse / 2)
    B = 2j * omega_ab / Omega * np.sin(Omega * t_pulse / 2)
    C = B
    D = np.cos(Omega * t_pulse / 2) - 1j * Omega_3 / Omega * np.sin(Omega * t_pulse / 2)

    return A, B, C, D


def _single_pulse_propagator_2x2(
    detuning_hz,
    t_pulse,
    rabi_freq_hz,
    pulse_phase=0.0,
    k_sign=+1,
    k=K_WAVEVECTOR,
    vz=0.0,
    m_ground=0,
):
    r"""Single-branch 2x2 Bordé pulse propagator, state order ``[excited, ground]``.

    This is the single source of truth for the per-pulse propagator matrix
    ``[[A, B e^{-i\phi}], [C e^{+i\phi}, D]]``. It is shared by
    ``pulse_interaction_in_borde_representation`` (which applies it per row) and
    the staircase ARP composer in :mod:`lmt_sim.arp`, so the two never encode
    different pulse physics.

    ``rabi_freq_hz`` is the dynamics Rabi frequency in Hz; the Bordé angular
    value is ``omega_ab = pi * rabi_freq_hz`` (the pi-not-2pi convention).
    """
    omega_ab = np.pi * rabi_freq_hz
    A, B, C, D = _calculate_interaction_constants(
        detuning_hz,
        t_pulse,
        omega_ab,
        k=k,
        vz=vz,
        k_sign=k_sign,
        m_ground=m_ground,
    )
    return np.array(
        [[A, B * np.exp(-1j * pulse_phase)], [C * np.exp(1j * pulse_phase), D]]
    )


def propagate_states_in_borde_representation(
    state: AtomState,
    time_of_propegation: float,
    detuning_hz: float,
    vz: float,
    k_wavevector=K_WAVEVECTOR,
):
    """
    Accumulate phase during free fall

    Following equation 5 of Borde's paper, which is broken down nicely in eq 14.

    We can ignore the velocity and spatial dependence - this was already taken
    care of by the unitary transformation to the Borde frame.

    Parameters
    ----------
    state : AtomState
        Atom state in the Borde representation.
    time_of_propegation : float
        Time to propagate
    detuning_hz : float
        Laser detuning in Hz
    vz : float
        Reference z-velocity (v_0) used for Borde phase calculations
    k_wavevector : float, optional
        Wavevector magnitude, by default K_WAVEVECTOR

    Returns
    -------
    AtomState
        Propagated atom state.
    """

    amplitudes_out = np.empty_like(state.amplitudes)
    positions_out = np.empty_like(state.positions)

    k_sign = 1  # FIXME: I think I am free to choose to consider the m <-> m+1 pairs like this, but I should read the paper again and make sure
    # FIXME I can test by e.g. running an interferometer from excited to ground and making sure it works

    for idx in range(len(state.m_values)):
        is_ground = state.internal_is_ground[idx]
        if is_ground:
            m_ground = state.m_values[idx]
        else:
            m_ground = state.m_values[idx] - k_sign

        Delta, delta_recoil, Omega_0_val, Omega_3 = _borde_frame_constants(
            detuning_hz,
            k=k_wavevector,
            vz=vz,
            k_sign=k_sign,
            m_ground=m_ground,
        )

        phase = np.exp(1j * Omega_3 * time_of_propegation / 2)

        if is_ground:
            amplitudes_out[idx] = state.amplitudes[idx] * phase
        else:
            amplitudes_out[idx] = state.amplitudes[idx] * np.conj(phase)

        # Update position ballistically for all three dimensions.
        # velocities[idx, 2] already encodes v_0 + m * v_recoil accumulated from
        # recoil kicks during previous pulses, so no separate m-term is needed here.
        positions_out[idx] = (
            state.positions[idx] + state.velocities[idx] * time_of_propegation
        )

    return AtomState(
        m_values=state.m_values,
        positions=positions_out,
        velocities=state.velocities,
        amplitudes=amplitudes_out,
        internal_is_ground=state.internal_is_ground,
        # Free fall does not change the laser frame; carry the reference forward.
        t_ref=state.t_ref,
        detuning_ref_hz=state.detuning_ref_hz,
        accumulated_detuning_cycles=state.accumulated_detuning_cycles,
    )


def _branch_row_with_propagator(
    idx,
    state: AtomState,
    prop_matrix,
    m_ground,
    m_excited,
    vz,
    *,
    t_total,
    kick_fraction,
    new_amplitudes,
    new_m_values,
    new_positions,
    new_velocities,
    ind_excited,
):
    """Apply one row's 2x2 propagator and write its two output branches.

    ``prop_matrix`` acts on ``[c_excited, c_ground]``. The input row ``idx`` is
    split into a ground-output branch (written to ``idx``) and an excited-output
    branch (written to ``ind_excited + idx``). Shared by the square-pulse
    interaction (``pulse_interaction_in_borde_representation``) and the composite
    interaction (``composite_pulse_interaction_in_borde_representation``) so the
    branching / velocity / position bookkeeping lives in exactly one place.

    ``t_total`` is the full event duration used for ballistic propagation;
    ``kick_fraction`` in ``[0, 1]`` is the fraction of it spent at the OLD
    velocity before the discrete ``m -> m+-k`` recoil is imparted (0.5 =
    midpoint). A no-change branch (ground->ground or excited->excited) keeps its
    velocity for the whole duration.

    TODO (stationary-atom approximation): each branch moves at a single (old or
    new) velocity across the whole pulse and the position/beam is read once. For
    long composite/ARP pulses the atom moves appreciably DURING the pulse, so
    this is a poor approximation; the proper fix is per-sub-slice position
    re-evaluation at each slice's midpoint. See section 6.2 of
    docs/LMT_milestones_and_implementation_plan.md.
    """
    # Borde notation: state = [excited_amp, ground_amp] (b, a)
    if state.internal_is_ground[idx]:
        amplitude_vector_in = np.array([0, state.amplitudes[idx]], dtype=complex)
    else:
        amplitude_vector_in = np.array([state.amplitudes[idx], 0], dtype=complex)

    amplitude_vector_out = prop_matrix @ amplitude_vector_in

    # Build 3D velocity vectors for the two output branches. vz for each branch is
    # the reference velocity plus m * RECOIL_VELOCITY, consistent with the
    # invariant velocities[idx, 2] == vz + m_values[idx] * RECOIL_VELOCITY. vx and
    # vy are taken from the input velocities (unchanged).
    vz_ground = vz + m_ground * RECOIL_VELOCITY
    vz_excited = vz + m_excited * RECOIL_VELOCITY
    vel_ground_3d = np.array(
        [state.velocities[idx, 0], state.velocities[idx, 1], vz_ground]
    )
    vel_excited_3d = np.array(
        [state.velocities[idx, 0], state.velocities[idx, 1], vz_excited]
    )

    # Ground-output branch: m = m_ground
    new_amplitudes[idx] = amplitude_vector_out[1]
    new_m_values[idx] = m_ground
    new_velocities[idx] = vel_ground_3d
    # Excited-output branch: m = m_excited
    new_amplitudes[ind_excited + idx] = amplitude_vector_out[0]
    new_m_values[ind_excited + idx] = m_excited
    new_velocities[ind_excited + idx] = vel_excited_3d

    # Positions: a transition branch spends kick_fraction of the pulse at the OLD
    # velocity and the remainder at the NEW velocity (the recoil is imparted at
    # that instant). kick_fraction=0.5 reproduces the square-pulse midpoint rule.
    t_before = kick_fraction * t_total
    t_after = t_total - t_before
    if state.internal_is_ground[idx]:  # start in ground
        # ground->ground: stays at ground velocity
        new_positions[idx] = state.positions[idx] + vel_ground_3d * t_total
        # ground->excited: old = ground, new = excited
        new_positions[ind_excited + idx] = (
            state.positions[idx] + vel_ground_3d * t_before + vel_excited_3d * t_after
        )
    else:  # start in excited
        # excited->excited: stays at excited velocity
        new_positions[idx] = state.positions[idx] + vel_excited_3d * t_total
        # excited->ground: old = excited, new = ground
        new_positions[ind_excited + idx] = (
            state.positions[idx] + vel_excited_3d * t_before + vel_ground_3d * t_after
        )


def pulse_interaction_in_borde_representation(
    state: AtomState,
    pulse_detuning: float,
    t_pulse: float,
    pulse_rabi_freq,
    pulse_phase=0.0,
    k_sign=+1,
    k_wavevector=K_WAVEVECTOR,
    vz: float = 0.0,
    probe_shift_coefficient: float = 0.0,
    stark_rabi_freq=None,
):
    r"""
    Apply a laser pulse in the Borde representation

    Each pulse couples |a, m> (ground, momentum m) to |b, m+1> (excited,
    momentum m+1) for a +k laser, or |b, m-1> for a -k laser.

    Since we track each state independently (so that we can consider spatial
    overlap) this doubles the size of the state vector each pulse.

    The paramaterisation of the probe_shift_coefficient $\alpha$ is
    $$
    \delta_\text{probe} = \alpha_\mathrm{beam} \Omega^2
    $$
    where $\delta_\text{probe}$ is the probe (light) shift in Hz and $\Omega$ is the
    Rabi frequency in Hz. $\alpha$ is therefore in units of inverse frequency (1/Hz).
    There is no factor of $2\pi$: everything here is in Hz. The shift is subtracted
    from the laser detuning (the lab tunes the laser above the bare resonance to
    compensate the light shift, so removing it recovers the bare recoil ladder) and
    applies only during the pulse, not during free evolution.

    Parameters
    ----------
    state : AtomState
        Atom state in the Borde representation.
    pulse_detuning : float
        Laser detuning from resonance in Hz
    t_pulse : float
        Pulse duration in seconds
    pulse_rabi_freq : float or array-like, shape (N,)
        Rabi frequency in Hz. Scalar is broadcast to all rows; an (N,) array
        gives a per-row Rabi frequency.
    pulse_phase : float, optional
        Pulse phase in radians, by default 0.0
    k_sign : int, optional
        Direction of laser (+1 for +k, -1 for -k), by default +1
    k_wavevector : float, optional
        Wavevector magnitude, by default K_WAVEVECTOR
    vz : float, optional
        Reference z-velocity (v_0) used for Borde phase calculations, by default 0.0
    probe_shift_coefficient : float, optional
        Probe (light) shift coefficient in 1/Hz, by default 0.0. The effective
        detuning is reduced by ``probe_shift_coefficient * rabi_freq**2`` Hz.
    stark_rabi_freq : float or array-like, shape (N,), optional
        Rabi frequency in Hz used for the probe (light) shift only, for pulses
        whose intensity does not match their dynamics (e.g. shaped /
        optimal-control pulses modelled as a plain pi pulse: ``pulse_rabi_freq``
        is the fictitious value implied by the duration, while the light shift
        scales with the true intensity). ``None`` (default) uses
        ``pulse_rabi_freq``, i.e. an ordinary square pulse.
    Returns
    -------
    AtomState
        Atom state after the pulse.
    """

    # Implement equation 13 / 14 / 15

    # Prepare output arrays -- each row branches into two
    N = state.amplitudes.shape[0]
    new_num_rows = N * 2

    # Broadcast pulse_rabi_freq to a per-row array
    rabi_arr = np.broadcast_to(np.asarray(pulse_rabi_freq, dtype=float), (N,)).copy()

    # The light shift follows the true intensity, which for a shaped pulse is
    # decoupled from the (fictitious) dynamics Rabi frequency above
    if stark_rabi_freq is None:
        stark_rabi_arr = rabi_arr
    else:
        stark_rabi_arr = np.broadcast_to(
            np.asarray(stark_rabi_freq, dtype=float), (N,)
        ).copy()

    new_amplitudes = np.empty(new_num_rows, dtype=state.amplitudes.dtype)
    new_m_values = np.empty(new_num_rows, dtype=state.m_values.dtype)
    new_positions = np.empty((new_num_rows, 3), dtype=state.positions.dtype)
    new_velocities = np.empty((new_num_rows, 3), dtype=state.velocities.dtype)
    new_is_ground = np.empty(new_num_rows, dtype=state.internal_is_ground.dtype)

    # Ground-output rows first, excited-output rows second
    ind_excited = N
    new_is_ground[:ind_excited] = True
    new_is_ground[ind_excited:] = False

    for idx in range(N):
        # Borde notation: state = [excited_amp, ground_amp] (b, a)
        if state.internal_is_ground[idx]:
            # The ground state has m = m_a, so the relevant excited state for
            # the pulse is m = m_a +- 1
            m_ground = state.m_values[idx]
            m_excited = state.m_values[idx] + k_sign
        else:
            # This excited state has m = m_b, so the relevant ground state for
            # the pulse is m = m_b -+ 1
            m_ground = state.m_values[idx] - k_sign
            m_excited = state.m_values[idx]

        effective_detuning_hz = _effective_detuning_hz(
            pulse_detuning, probe_shift_coefficient, stark_rabi_arr[idx]
        )

        prop_matrix = _single_pulse_propagator_2x2(
            effective_detuning_hz,
            t_pulse,
            rabi_arr[idx],
            pulse_phase=pulse_phase,
            k_sign=k_sign,
            k=k_wavevector,
            vz=vz,
            m_ground=m_ground,
        )

        # Square pulse: recoil imparted at the midpoint (kick_fraction=0.5).
        _branch_row_with_propagator(
            idx,
            state,
            prop_matrix,
            m_ground,
            m_excited,
            vz,
            t_total=t_pulse,
            kick_fraction=0.5,
            new_amplitudes=new_amplitudes,
            new_m_values=new_m_values,
            new_positions=new_positions,
            new_velocities=new_velocities,
            ind_excited=ind_excited,
        )

    return AtomState(
        m_values=new_m_values,
        positions=new_positions,
        velocities=new_velocities,
        amplitudes=new_amplitudes,
        internal_is_ground=new_is_ground,
        # The pulse doubles the rows but does not change the laser frame.
        t_ref=state.t_ref,
        detuning_ref_hz=state.detuning_ref_hz,
        accumulated_detuning_cycles=state.accumulated_detuning_cycles,
    )


def composite_pulse_interaction_in_borde_representation(
    state: AtomState,
    subpulses,
    *,
    k_sign=+1,
    pulse_phase=0.0,
    momentum_kick_fraction=0.5,
    k_wavevector=K_WAVEVECTOR,
    vz: float = 0.0,
    probe_shift_coefficient: float = 0.0,
):
    r"""Apply a composite (multi-block) pulse as a SINGLE branching event.

    Like :func:`pulse_interaction_in_borde_representation` this couples
    ``|a, m>`` to ``|b, m+-k>`` and doubles the row count, but each row's 2x2
    mixing matrix is the composed product of the staircase of fixed-parameter
    sub-blocks (:func:`lmt_sim.arp.compose_arp_2x2`) instead of one square-pulse
    propagator. This is how an adiabatic-rapid-passage (ARP) frequency sweep --
    hundreds of tiny sub-pulses -- enters the sequence WITHOUT the ``2**N`` row
    explosion that branching every sub-pulse would cause: the whole sweep is one
    branching event.

    The frame fields ``(t_ref, detuning_ref_hz, accumulated_detuning_cycles)``
    are carried forward UNCHANGED, exactly like the square interaction: the
    sweep's laser-phase integral and end-frame are folded by the sequence loop
    via :func:`advance_composite_frame_integral`.

    Parameters
    ----------
    state : AtomState
    subpulses : sequence
        The staircase; each item exposes ``detuning_hz`` / ``rabi_freq_hz`` /
        ``duration`` (e.g. :class:`lmt_sim.arp.ARPSubPulse`).
    k_sign : int, optional
        Laser direction (+1 / -1), by default +1.
    pulse_phase : float, optional
        Constant optical phase across the staircase (rad), by default 0.0.
    momentum_kick_fraction : float, optional
        Fraction of the total pulse duration before the discrete ``m -> m+-k``
        recoil is imparted, for ballistic position propagation only (0.5 =
        midpoint), by default 0.5.
    k_wavevector : float, optional
        Wavevector magnitude, by default K_WAVEVECTOR.
    vz : float, optional
        Reference z-velocity (v_0); the per-row recoil shift is added internally
        by :func:`_borde_frame_constants`, by default 0.0.
    probe_shift_coefficient : float, optional
        Probe (light) shift coefficient (1/Hz), by default 0.0. Applied per
        sub-block against that block's Rabi frequency to the DYNAMICS only; the
        laser-phase integral and the end detuning the sequence loop folds use the
        NOMINAL sub-block detunings (the probe shift is an atom-frame energy
        shift, not a change to the laser program).

    Returns
    -------
    AtomState
        Atom state after the pulse, with ``N*2`` rows and the frame fields
        unchanged.
    """
    # Late import: arp imports this module, so importing it at module load time
    # would be circular.
    from lmt_sim import arp

    N = state.amplitudes.shape[0]
    new_num_rows = N * 2

    new_amplitudes = np.empty(new_num_rows, dtype=state.amplitudes.dtype)
    new_m_values = np.empty(new_num_rows, dtype=state.m_values.dtype)
    new_positions = np.empty((new_num_rows, 3), dtype=state.positions.dtype)
    new_velocities = np.empty((new_num_rows, 3), dtype=state.velocities.dtype)
    new_is_ground = np.empty(new_num_rows, dtype=state.internal_is_ground.dtype)

    # Ground-output rows first, excited-output rows second
    ind_excited = N
    new_is_ground[:ind_excited] = True
    new_is_ground[ind_excited:] = False

    # The DYNAMICS see the probe (light) shift per sub-block; the laser-frame
    # bookkeeping (done by the sequence loop) uses the nominal detunings.
    if probe_shift_coefficient != 0.0:
        dynamics_subpulses = [
            replace(
                sub,
                detuning_hz=_effective_detuning_hz(
                    sub.detuning_hz, probe_shift_coefficient, sub.rabi_freq_hz
                ),
            )
            for sub in subpulses
        ]
    else:
        dynamics_subpulses = subpulses

    t_total = sum(sub.duration for sub in subpulses)

    for idx in range(N):
        if state.internal_is_ground[idx]:
            m_ground = state.m_values[idx]
            m_excited = state.m_values[idx] + k_sign
        else:
            m_ground = state.m_values[idx] - k_sign
            m_excited = state.m_values[idx]

        # Compose the staircase for THIS row's two-level pair. Pass the BARE
        # reference vz and the row's m_ground -- _borde_frame_constants adds the
        # m_ground recoil/Doppler terms internally, so do NOT pass
        # vz + m*RECOIL_VELOCITY here (that would double-count the recoil shift).
        prop_matrix = arp.compose_arp_2x2(
            dynamics_subpulses,
            k_sign=k_sign,
            vz=vz,
            m_ground=m_ground,
            pulse_phase=pulse_phase,
        )

        _branch_row_with_propagator(
            idx,
            state,
            prop_matrix,
            m_ground,
            m_excited,
            vz,
            t_total=t_total,
            kick_fraction=momentum_kick_fraction,
            new_amplitudes=new_amplitudes,
            new_m_values=new_m_values,
            new_positions=new_positions,
            new_velocities=new_velocities,
            ind_excited=ind_excited,
        )

    return AtomState(
        m_values=new_m_values,
        positions=new_positions,
        velocities=new_velocities,
        amplitudes=new_amplitudes,
        internal_is_ground=new_is_ground,
        # The interaction doubles the rows but does not change the laser frame;
        # the sequence loop folds the staircase integral / end-frame afterwards
        # (advance_composite_frame_integral).
        t_ref=state.t_ref,
        detuning_ref_hz=state.detuning_ref_hz,
        accumulated_detuning_cycles=state.accumulated_detuning_cycles,
    )


def change_laser_frequency_in_borde_representation(
    state: AtomState,
    new_detuning_hz: float,
    time: float,
):
    r"""Record a laser-frequency change at ``time`` (no change to the amplitudes).

    The Bordé frame co-rotates with the laser, so the lab<->Bordé transform
    depends on the integral of the laser detuning ``Phi(t) = integral_0^t delta``.
    We carry that piecewise-linear integral on the state as
    ``(t_ref, detuning_ref_hz, accumulated_detuning_cycles)``: ``detuning_ref_hz`` is the detuning
    since ``t_ref`` and ``accumulated_detuning_cycles`` is ``Phi(t_ref)`` (closed
    segments).

    At a frequency step at ``time`` we **close the open segment** and start a new
    one:

    1. ``accumulated_detuning_cycles += detuning_ref_hz * (time - t_ref)`` -- fold the just
       -ended segment ``[t_ref, time]`` (at the OLD detuning ``detuning_ref_hz``) into the
       closed-segment integral.
    2. ``t_ref = time``, ``detuning_ref_hz = new_detuning_hz``.

    Crucially the **amplitudes are NOT changed**. The laser phase is continuous
    across the step, so the instantaneous Bordé frames before and after coincide at
    that instant and the correct boundary phase is zero. The old
    ``exp(+/- i pi Df t)`` "frame change" double-counted the chirp (the detuning is
    already carried in each pulse's ``Omega_3`` diagonal and in the free-evolution
    phase); applying any nonzero diagonal phase here would corrupt the next pulse.
    See docs/arp_frame_change_finding.md.

    Because closing a segment is just an exact additive split of the same integral,
    a "change" to the SAME frequency (``new_detuning_hz == detuning_ref_hz``) leaves both the
    amplitudes and ``Phi`` untouched -- **no effect on any physical prediction**.

    Parameters
    ----------
    state : AtomState
        Atom state in the Bordé representation. Its ``t_ref``/``detuning_ref_hz`` /
        ``accumulated_detuning_cycles`` give the current frame integral.
    new_detuning_hz : float
        Laser detuning (Hz) from this moment on.
    time : float
        **Global** simulation time at which the frame change happens, in seconds.

    Returns
    -------
    AtomState
        Atom state with the frame integral advanced to ``time`` and the new
        detuning recorded; ``amplitudes`` are unchanged.
    """
    new_accumulated = state.accumulated_detuning_cycles + state.detuning_ref_hz * (
        time - state.t_ref
    )
    return replace(
        state,
        t_ref=time,
        detuning_ref_hz=new_detuning_hz,
        accumulated_detuning_cycles=new_accumulated,
    )


def advance_composite_frame_integral(
    state: AtomState,
    phi_cycles: float,
    end_detuning_hz: float,
    start_time: float,
    end_time: float,
):
    r"""Fold a composite/ARP pulse's laser-phase integral and move to its end frame.

    A composite pulse sweeps the laser detuning WITHIN the pulse, so the genuine
    integral of the laser detuning over ``[start_time, end_time]`` is the
    staircase sum ``phi_cycles = sum_k detuning_k * dt_k`` (cycles), **not**
    ``rate * duration``. This is the one place a closed segment's integral is not
    ``rate * duration``, so it cannot go through
    :func:`change_laser_frequency_in_borde_representation`.

    Like that function the **amplitudes are NOT touched**: the laser phase is
    continuous, so the instantaneous Bordé frames coincide at each boundary. The
    output amplitudes of ``compose_arp_2x2`` are already in the instantaneous
    frame at the sweep's end detuning, which becomes the new reference.

    This advances the piecewise-linear integral by

    1. closing the OPEN segment ``[t_ref, start_time]`` at the OLD rate
       ``detuning_ref_hz`` (the laser frequency just before the sweep), and
    2. adding the swept-segment integral ``phi_cycles`` over
       ``[start_time, end_time]``,

    then sets ``t_ref = end_time`` and ``detuning_ref_hz = end_detuning_hz``. It
    therefore subsumes the usual frequency-step rebase (step 1) AND the swept
    fold (step 2) in one call, so the caller does not separately invoke
    ``change_laser_frequency_in_borde_representation`` for a composite pulse.

    See docs/arp_frame_change_finding.md.

    Parameters
    ----------
    state : AtomState
        State just after the composite interaction; its frame fields still hold
        the pre-pulse reference.
    phi_cycles : float
        Genuine laser-phase integral over the sweep, ``sum_k detuning_k * dt_k``
        (cycles), using the NOMINAL sub-block detunings.
    end_detuning_hz : float
        Laser detuning at the end of the sweep (the new frame reference).
    start_time, end_time : float
        Global simulation times (s) at the start and end of the sweep.

    Returns
    -------
    AtomState
        State with the frame integral advanced past the sweep; ``amplitudes`` are
        unchanged.
    """
    if end_time < start_time or start_time < state.t_ref:
        raise ValueError(
            "advance_composite_frame_integral expects t_ref <= start_time <= "
            f"end_time (got t_ref={state.t_ref}, start_time={start_time}, "
            f"end_time={end_time})"
        )
    new_accumulated = (
        state.accumulated_detuning_cycles
        + state.detuning_ref_hz * (start_time - state.t_ref)
        + phi_cycles
    )
    return replace(
        state,
        t_ref=end_time,
        detuning_ref_hz=end_detuning_hz,
        accumulated_detuning_cycles=new_accumulated,
    )


def gaussian_rabi(
    positions, on_axis_rabi, beam_waist, wavelength=TRANSITION_WAVELENGTH
):
    """Per-row Rabi frequency from a TEM00 Gaussian beam profile.

    Includes both the transverse intensity variation and the z-dependent beam
    expansion due to the Rayleigh range.  The beam waist is assumed to be at
    z = 0 (the atom's initial position).

    Rayleigh range:   z_R = pi * w0^2 / lambda
    Beam radius:      w(z) = w0 * sqrt(1 + (z / z_R)^2)
    Rabi frequency:   Omega(x, y, z) = Omega_0 * (w0/w(z)) * exp(-(x^2+y^2) / w(z)^2)

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        [x, y, z] positions of each state row.
    on_axis_rabi : float
        On-axis Rabi frequency at the beam waist in Hz.
    beam_waist : float
        Beam waist radius w0 (1/e field radius) in metres.
    wavelength : float, optional
        Laser wavelength in metres, by default TRANSITION_WAVELENGTH.
        Used to compute the Rayleigh range.

    Returns
    -------
    np.ndarray, shape (N,)
        Per-row Rabi frequency in Hz.
    """
    z_R = np.pi * beam_waist**2 / wavelength
    w_z = beam_waist * np.sqrt(1 + (positions[:, 2] / z_R) ** 2)
    r2 = positions[:, 0] ** 2 + positions[:, 1] ** 2
    return on_axis_rabi * (beam_waist / w_z) * np.exp(-r2 / w_z**2)


def do_gaussian_pulse(
    state: AtomState,
    pulse_detuning,
    t_pulse,
    on_axis_rabi_freq,
    beam_waist,
    pulse_phase=0.0,
    k_sign=+1,
    k_wavevector=K_WAVEVECTOR,
    vz=0.0,
    wavelength=TRANSITION_WAVELENGTH,
    probe_shift_coefficient=0.0,
    on_axis_stark_rabi_freq=None,
):
    """Apply a laser pulse with a full 3-D Gaussian (TEM00) intensity profile.

    Computes the per-row Rabi frequency at the pulse midpoint using the
    TEM00 Gaussian profile — including the Rayleigh-range z-dependence — then
    calls pulse_interaction_in_borde_representation with those per-row
    frequencies.  The beam waist is assumed to be located at z = 0 (the
    atom's initial position).

    Parameters
    ----------
    state : AtomState
        Atom state in the Borde representation.
    pulse_detuning : float
        Laser detuning from resonance in Hz.
    t_pulse : float
        Pulse duration in seconds.
    on_axis_rabi_freq : float
        On-axis (peak) Rabi frequency at the beam waist in Hz.
    beam_waist : float
        Beam waist w0 (1/e field radius) in metres. Required -- no default.
    pulse_phase : float, optional
        Pulse phase in radians, by default 0.0.
    k_sign : int, optional
        Laser direction (+1 or -1), by default +1.
    k_wavevector : float, optional
        Wavevector magnitude, by default K_WAVEVECTOR.
    vz : float, optional
        Reference z-velocity for Borde phase calculations, by default 0.0.
    wavelength : float, optional
        Laser wavelength in metres, by default TRANSITION_WAVELENGTH.
        Used to compute the Rayleigh range z_R = pi * w0^2 / wavelength.
    probe_shift_coefficient : float, optional
        Probe (light) shift coefficient in 1/Hz, by default 0.0. Reduces the
        effective detuning by ``probe_shift_coefficient * rabi_freq**2`` Hz per
        row (Rabi-squared, i.e. intensity, scaling). Because the Rabi frequency
        is per-row for a Gaussian beam, the probe shift is naturally per-row too.
    on_axis_stark_rabi_freq : float, optional
        On-axis Rabi frequency in Hz used for the probe (light) shift only, for
        shaped pulses whose true intensity does not match the fictitious
        ``on_axis_rabi_freq`` implied by their duration. It is scaled by the
        same Gaussian envelope as ``on_axis_rabi_freq``, so the per-row shift
        still tracks the local intensity. ``None`` (default) uses
        ``on_axis_rabi_freq``.

    Returns
    -------
    AtomState
        Atom state after the Gaussian pulse.
    """
    # Compute 3-D position at pulse midpoint for Gaussian Rabi calculation
    positions_mid = state.positions + state.velocities * (t_pulse / 2)
    rabi_per_row = gaussian_rabi(
        positions_mid, on_axis_rabi_freq, beam_waist, wavelength
    )
    stark_rabi_per_row = (
        None
        if on_axis_stark_rabi_freq is None
        else gaussian_rabi(
            positions_mid, on_axis_stark_rabi_freq, beam_waist, wavelength
        )
    )
    return pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=pulse_detuning,
        t_pulse=t_pulse,
        pulse_rabi_freq=rabi_per_row,
        pulse_phase=pulse_phase,
        k_sign=k_sign,
        k_wavevector=k_wavevector,
        vz=vz,
        probe_shift_coefficient=probe_shift_coefficient,
        stark_rabi_freq=stark_rabi_per_row,
    )


def calculate_ground_and_excited_probabilities(state: AtomState):
    unique_m = np.unique(state.m_values)
    ground_prob = 0.0
    excited_prob = 0.0

    for m in unique_m:
        k_this_m = state.m_values == m
        total_gnd_amp = np.sum(state.amplitudes[k_this_m & state.internal_is_ground])
        total_exc_amp = np.sum(state.amplitudes[k_this_m & ~state.internal_is_ground])

        ground_prob += np.abs(total_gnd_amp) ** 2
        excited_prob += np.abs(total_exc_amp) ** 2

    return ground_prob, excited_prob


def do_clearout(state: AtomState, rng=None):
    """Projective measurement in the {ground, excited} basis.

    Per-atom Monte Carlo: samples one outcome from the current
    P(ground):P(excited) ratio.

    The projection is performed directly in the Bordé frame.  The per-row
    phase from :func:`transform_state_vector` is the same for all rows
    sharing the same ``(m, is_ground)``, so projecting in the Bordé frame
    is identical to projecting in the lab frame.  No frame transform is
    needed inside this function.

    Parameters
    ----------
    state : AtomState
        Atom state in the Bordé or lab representation.
    rng : np.random.Generator, optional
        Random-number generator.  If ``None``, ``np.random.default_rng()``
        is used (avoids the legacy global random state).

    Returns
    -------
    None
        If the atom is projected to ground (discarded).
    AtomState
        Surviving excited-state rows, renormalised to unit norm.
    """
    if rng is None:
        rng = np.random.default_rng()

    p_g, p_e = calculate_ground_and_excited_probabilities(state)

    total_prob = p_g + p_e
    if np.isclose(total_prob, 0.0):
        # Empty state from a prior clearout -- treat as already discarded
        return None

    u = rng.uniform()
    if u < p_g / total_prob:
        # Projected to ground -- discard
        return None

    # Survived -- keep only excited rows and renormalise
    keep = ~state.internal_is_ground
    return AtomState(
        m_values=state.m_values[keep],
        positions=state.positions[keep],
        velocities=state.velocities[keep],
        amplitudes=state.amplitudes[keep] * (1.0 / np.sqrt(p_e)),
        internal_is_ground=state.internal_is_ground[keep],
        # Projection does not change the laser frame.
        t_ref=state.t_ref,
        detuning_ref_hz=state.detuning_ref_hz,
        accumulated_detuning_cycles=state.accumulated_detuning_cycles,
    )


def discard_and_renormalise_state_vector(state: AtomState, discard_threshold: float):
    """
    Discard states with amplitude^2 <= discard_threshold and renormalise.
    """

    mod_squared_amplitude = np.abs(state.amplitudes) ** 2
    keep_mask = mod_squared_amplitude > discard_threshold
    if not np.any(keep_mask):
        raise ValueError("All states discarded, increase discard_threshold")

    new_amplitudes = state.amplitudes[keep_mask] * (
        1.0 / np.sqrt(np.sum(np.abs(state.amplitudes[keep_mask]) ** 2))
    )
    new_state = AtomState(
        m_values=state.m_values[keep_mask],
        positions=state.positions[keep_mask],
        velocities=state.velocities[keep_mask],
        amplitudes=new_amplitudes,
        internal_is_ground=state.internal_is_ground[keep_mask],
        # Discarding rows does not change the laser frame.
        t_ref=state.t_ref,
        detuning_ref_hz=state.detuning_ref_hz,
        accumulated_detuning_cycles=state.accumulated_detuning_cycles,
    )
    return new_state
