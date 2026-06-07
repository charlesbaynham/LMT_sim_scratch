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

GRAVITY_DOPPLER_PER_SEC_HZ = TRANSITION_FREQUENCY * constants.g / constants.c


@dataclass(frozen=True)
class AtomState:
    m_values: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    amplitudes: np.ndarray
    internal_is_ground: np.ndarray


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
    )


def transform_state_vector(
    state: AtomState,
    omega_laser,
    t,
    z,
    vz,
    k=K_WAVEVECTOR,
    omega_0=2 * np.pi * TRANSITION_FREQUENCY,
    inverse=False,
):
    """
    Transform a state vector to/from the position & time independent frame.

    Eq. 4 of Bordé's paper

    Note that this transformation depends on the laser frequency.

    TODO: Convert to the integral of laser phase
    """
    global_phase = np.exp(1j * omega_0 / 2 * t)

    m_dependent_phase_gnd = np.exp(
        1j / 2 * (-omega_laser * t - 2 * state.m_values * k * (z + vz * t))
    )
    m_dependent_phase_excited = np.exp(
        1j / 2 * (omega_laser * t - 2 * state.m_values * k * (z + vz * t))
    )
    m_dependent_phase = np.where(
        state.internal_is_ground, m_dependent_phase_gnd, m_dependent_phase_excited
    )

    transform = global_phase * m_dependent_phase

    if inverse:
        transform = np.conj(transform)

    return replace(state, amplitudes=transform * state.amplitudes)


def _calculate_propagation_constants(
    detuning_hz,
    k_sign=+1,
    k=K_WAVEVECTOR,
    vz=0.0,
    m_ground=0,
):
    omega_0 = 2 * np.pi * TRANSITION_FREQUENCY
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
    omega_laser,
    t_pulse,
    omega_ab,
    k_sign=+1,
    k=K_WAVEVECTOR,
    vz=0.0,
    m_ground=0,
):
    omega_0 = 2 * np.pi * TRANSITION_FREQUENCY
    Delta = omega_laser - omega_0
    delta_recoil = constants.hbar * k**2 / (2 * MASS_ATOM)

    # Equation 7: Omega_3 = Delta - k_sign*k*vz - [(m+k_sign)^2 - m^2]*delta_recoil
    Omega_3 = (
        Delta
        - k_sign * k * vz
        - ((m_ground + k_sign) ** 2 - m_ground**2) * delta_recoil
    )

    # Eq 12: Generalized Rabi frequency
    Omega = np.sqrt(Omega_3**2 + 4 * omega_ab**2)

    # Equation 13: Matrix elements
    A = np.cos(Omega * t_pulse / 2) + 1j * Omega_3 / Omega * np.sin(Omega * t_pulse / 2)
    B = 2j * omega_ab / Omega * np.sin(Omega * t_pulse / 2)
    C = B
    D = np.cos(Omega * t_pulse / 2) - 1j * Omega_3 / Omega * np.sin(Omega * t_pulse / 2)

    return A, B, C, D


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

        Delta, delta_recoil, Omega_0_val, Omega_3 = _calculate_propagation_constants(
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
        # Build input state vector for propagate_pulse
        # Borde notation: state = [excited_amp, ground_amp] (b, a)
        if state.internal_is_ground[idx]:
            # The ground state has m = m_a, so the relevant excited state for
            # the pulse is m = m_a +- 1
            m_ground = state.m_values[idx]
            m_excited = state.m_values[idx] + k_sign
            amplitude_vector_in = np.array([0, state.amplitudes[idx]], dtype=complex)
        else:
            # This excited state has m = m_b, so the relevant ground state for
            # the pulse is m = m_b -+ 1
            m_ground = state.m_values[idx] - k_sign
            m_excited = state.m_values[idx]
            amplitude_vector_in = np.array([state.amplitudes[idx], 0], dtype=complex)

        # Borde uses omega_ab = pi * RABI_FREQ, angular frequencies in rad/s
        omega_ab = np.pi * rabi_arr[idx]
        # Probe (light) shift: scales with intensity, i.e. Rabi**2. Subtracted
        # because the recorded detuning sits above the bare resonance by this
        # amount (the lab tunes the laser up to compensate the light shift).
        effective_detuning_hz = (
            pulse_detuning - probe_shift_coefficient * rabi_arr[idx] ** 2
        )
        omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + effective_detuning_hz)

        A, B, C, D = _calculate_interaction_constants(
            omega_laser,
            t_pulse,
            omega_ab,
            k=k_wavevector,
            vz=vz,
            k_sign=k_sign,
            m_ground=m_ground,
        )

        prop_matrix = np.array(
            [[A, B * np.exp(-1j * pulse_phase)], [C * np.exp(1j * pulse_phase), D]]
        )

        amplitude_vector_out = prop_matrix @ amplitude_vector_in

        # Build 3D velocity vectors for the two output branches.
        # vz for each branch is computed from the reference velocity plus
        # m * RECOIL_VELOCITY, consistent with the invariant
        # velocities[idx, 2] == vz + m_values[idx] * RECOIL_VELOCITY.
        # vx and vy are taken from the input velocities (unchanged).
        vz_ground = vz + m_ground * RECOIL_VELOCITY
        vz_excited = vz + m_excited * RECOIL_VELOCITY

        vel_ground_3d = np.array(
            [state.velocities[idx, 0], state.velocities[idx, 1], vz_ground]
        )
        vel_excited_3d = np.array(
            [state.velocities[idx, 0], state.velocities[idx, 1], vz_excited]
        )

        # Ground-output branch: m = m_ground, velocity = vel_ground_3d
        new_amplitudes[idx] = amplitude_vector_out[1]
        new_m_values[idx] = m_ground
        new_velocities[idx] = vel_ground_3d

        # Excited-output branch: m = m_excited, velocity = vel_excited_3d
        new_amplitudes[ind_excited + idx] = amplitude_vector_out[0]
        new_m_values[ind_excited + idx] = m_excited
        new_velocities[ind_excited + idx] = vel_excited_3d

        # Update the positions. If this state didn't change (i.e. ground->ground
        # or excited->excited) we can just use the input velocity for the whole
        # pulse duration. If it did change, we use the midpoint approximation:
        # half the pulse with the old velocity, half the pulse with the new
        # velocity.
        if state.internal_is_ground[idx]:  # start in ground
            # ground->ground
            new_positions[idx] = state.positions[idx] + vel_ground_3d * t_pulse
            # ground->excited
            new_positions[ind_excited + idx] = (
                state.positions[idx]
                + vel_ground_3d * (t_pulse / 2)
                + vel_excited_3d * (t_pulse / 2)
            )
        else:  # start in excited
            # excited->excited
            new_positions[idx] = state.positions[idx] + vel_excited_3d * t_pulse
            # excited->ground
            new_positions[ind_excited + idx] = (
                state.positions[idx]
                + vel_ground_3d * (t_pulse / 2)
                + vel_excited_3d * (t_pulse / 2)
            )

    return AtomState(
        m_values=new_m_values,
        positions=new_positions,
        velocities=new_velocities,
        amplitudes=new_amplitudes,
        internal_is_ground=new_is_ground,
    )


def change_laser_frequency_in_borde_representation(
    state: AtomState,
    old_detuning_hz: float,
    new_detuning_hz: float,
    time: float,
):
    """Re-express Bordé-frame amplitudes for a new laser frequency.

    The Borde frame moves with the laser's frequency. So, to avoid having to
    track the total laser phase, I transform amplitudes from one frame to
    another whenever the laser frequency changes.

    Parameters
    ----------
    state : AtomState
        Atom state in the Bordé representation.
    old_detuning_hz : float
        Detuning (Hz) of the Bordé frame the amplitudes are currently in.
    new_detuning_hz : float
        Detuning (Hz) of the Bordé frame to express the amplitudes in.
    time : float
        **Global** simulation time at which the frame change happens, in seconds.

    Returns
    -------
    AtomState
        Atom state expressed in the new Bordé frame.
    """
    delta_f = new_detuning_hz - old_detuning_hz
    phase_gnd = np.exp(-1j * np.pi * delta_f * time)
    phase_exc = np.exp(+1j * np.pi * delta_f * time)
    phase = np.where(state.internal_is_ground, phase_gnd, phase_exc)
    return replace(state, amplitudes=state.amplitudes * phase)


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
    )
    return new_state
