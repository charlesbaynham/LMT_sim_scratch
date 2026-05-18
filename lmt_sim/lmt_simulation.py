######

import logging

# import logging

import lmt_sim.version_info as vs

import numpy as np
from scipy import constants
from scipy.linalg import expm
import matplotlib.pyplot as plt
from lmt_sim.lmt_sequence import (
    Clearout,
    Freefall,
    Pulse,
    build_mach_zehnder_pulse_sequence,
    calculate_excited_fraction_for_pulse_sequence,
    run_pulse_sequence_in_borde_representation,
)

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
# structlog.configure(
#     wrapper_class=logging.make_filtering_bound_logger(logging.INFO),
# )


######

N_PULSES = 1
N_ROWS = 2 ** (N_PULSES * 2)
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

np.random.seed(42)


######


def make_atom_states(
    position_x=0.0,
    position_y=0.0,
    position_z=0.0,
    velocity_x=0.0,
    velocity_y=0.0,
    initial_velocity_z=0.0,
    c0=1,
    c1=0,
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
    m_values : ndarray, shape (2,), int
        Momentum quantum numbers.  m=0 for both initial rows.
    positions : ndarray, shape (2, 3), float
        [x, y, z] positions of each state row.
    velocities : ndarray, shape (2, 3), float
        [vx, vy, vz] velocities of each state row.
    internal_amplitude : ndarray, shape (2,), complex128
        Complex amplitudes for each state row.
    internal_is_ground : ndarray, shape (2,), bool
        True for ground-state rows.
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

    return m_values, positions, velocities, internal_amplitude, internal_is_ground


def _are_states_normalized(
    m_values: np.ndarray,
    internal_amplitude: np.ndarray,
    internal_is_ground: np.ndarray,
):
    # For each unique value of m, sum the amplitudes of all rows with that m, then square to get probability
    unique_m = np.unique(m_values)

    states = []

    for m in unique_m:
        k_this_m = m_values == m
        total_gnd_amp = np.sum(internal_amplitude[k_this_m & internal_is_ground])
        total_exc_amp = np.sum(internal_amplitude[k_this_m & ~internal_is_ground])

        states.append([m, total_gnd_amp, total_exc_amp])

    states = np.array(states)

    print(states)

    prob_excited = np.sum(np.abs(states[:, 2]) ** 2)
    prob_ground = np.sum(np.abs(states[:, 1]) ** 2)

    print(
        f"Total probability: {prob_ground + prob_excited:.6f} (ground: {prob_ground:.6f}, excited: {prob_excited:.6f})"
    )

    return np.isclose(prob_excited + prob_ground, 1.0, rtol=1e-6)


def transform_state_vector(
    m_values,
    internal_amplitude,
    internal_is_ground,
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
        1j / 2 * (-omega_laser * t - 2 * m_values * k * (z + vz * t))
    )
    m_dependent_phase_excited = np.exp(
        1j / 2 * (omega_laser * t - 2 * m_values * k * (z + vz * t))
    )
    m_dependent_phase = np.where(
        internal_is_ground, m_dependent_phase_gnd, m_dependent_phase_excited
    )

    transform = global_phase * m_dependent_phase

    if inverse:
        transform = np.conj(transform)

    return transform * internal_amplitude


def _calculate_propagation_constants(
    omega_laser,
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


# def _propagate_state_pair_pulse(
#     state,
#     omega_laser,
#     t_pulse,
#     omega_ab,
#     pulse_phase=0.0,
#     k_sign=+1,
#     k=K_WAVEVECTOR,
#     z=0.0,
#     vz=0.0,
#     m_ground=0,
#     omega_0=2 * np.pi * TRANSITION_FREQUENCY,
# ):
#     """Apply a laser pulse to a _single pair_ of states

#     This is an internal function - library users should use :meth:`propegate_states_pulse` instead.

#     Calculate the effect on a single pair of states |a, m_ground⟩ and
#     |b, m_ground + k_sign⟩ coupled by a laser pulse, where |a⟩ is the
#     ground state and |b⟩ is the excited state.

#     Parameters
#     ----------
#     state : ndarray, shape (2,), complex
#         Input state vector [excited_amp, ground_amp] (Bordé's b, a)
#     omega_laser : float
#         Laser angular frequency in rad/s
#     t_pulse : float
#         Pulse duration in seconds
#     omega_ab : float
#         Rabi frequency (rad/s), defined as pi/(2*t_pi) per Bordé
#     pulse_phase : float
#         Laser phase in radians
#     k_sign : int
#         +1 for +k laser, -1 for -k laser
#     k : float
#         Wavevector magnitude (2*pi/wavelength)
#     z : float
#         Position along laser beam at t=0
#     vz : float
#         Velocity along laser beam direction
#     m : int
#         **Ground state** momentum quantum number (default: 0)
#     omega_0 : float
#         Atomic transition angular frequency

#     Returns
#     -------
#     ndarray, shape (2,), complex
#         Output state vector [excited_amp, ground_amp] (Bordé's b, a)
#     """
#     delta_E = constants.hbar * omega_0
#     Delta = omega_laser - omega_0
#     delta_recoil = constants.hbar * k**2 / (2 * MASS_ATOM)

#     # Equation 7: Omega_3 = Delta - k_sign*k*vz - [(m+k_sign)^2 - m^2]*delta_recoil
#     Omega_3 = (
#         Delta
#         - k_sign * k * vz
#         - ((m_ground + k_sign) ** 2 - m_ground**2) * delta_recoil
#     )

#     # Equation 8: Omega_0_val = -[(m+k_sign)^2 + m^2]*delta_recoil - (2*m + k_sign)*k*vz
#     Omega_0_val = (
#         -((m_ground + k_sign) ** 2 + m_ground**2) * delta_recoil
#         - (2 * m_ground + k_sign) * k * vz
#     )

#     # Pauli matrices
#     S0 = np.array([[1, 0], [0, 1]], dtype=complex)

#     # Eq 12: Generalized Rabi frequency
#     Omega = np.sqrt(Omega_3**2 + 4 * omega_ab**2)

#     # Equation 13: Matrix elements
#     A = np.cos(Omega * t_pulse / 2) + 1j * Omega_3 / Omega * np.sin(Omega * t_pulse / 2)
#     B = 2j * omega_ab / Omega * np.sin(Omega * t_pulse / 2)
#     C = B
#     D = np.cos(Omega * t_pulse / 2) - 1j * Omega_3 / Omega * np.sin(Omega * t_pulse / 2)

#     # Equation 10: Pulse operator
#     matrix_operator = np.array(
#         [[A, B * np.exp(-1j * pulse_phase)], [C * np.exp(1j * pulse_phase), D]]
#     )

#     # Equation 9: Apply phase evolution and pulse
#     vec_ba_s_t = expm(1j / 2 * Omega_0_val * S0 * t_pulse) @ matrix_operator @ state

#     # Transform back (Equation 4)
#     return U.conj().T @ vec_ba_s_t


def propagate_states_in_borde_representation(
    m_values: np.ndarray,
    squiggly_amplitudes: np.ndarray,
    state_is_ground: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    time_of_propegation: float,
    omega_laser: float,
    vz: float,
    k_sign=+1,
    k_wavevector=K_WAVEVECTOR,
):
    """
    Accumulate phase during free fall

    Following equation 5 of Borde's paper, which is broken down nicely in eq 14.

    We can ignore the velocity and spatial dependence - this was already taken
    care of by the unitary transformation to the Borde frame.

    Parameters
    ----------
    m_values : np.ndarray
        Momentum quantum numbers for each state row
    squiggly_amplitudes : np.ndarray
        Amplitudes in Borde representation
    state_is_ground : np.ndarray
        Boolean array indicating ground (True) or excited (False) state
    positions : np.ndarray, shape (N, 3)
        [x, y, z] positions of each state row (classical tracking)
    velocities : np.ndarray, shape (N, 3)
        [vx, vy, vz] velocities of each state row. vx and vy are constant.
        vz includes accumulated recoil kicks.
    time_of_propegation : float
        Time to propagate
    omega_laser : float
        Laser angular frequency
    vz : float
        Reference z-velocity (v_0) used for Borde phase calculations
    k_sign : int, optional
        Direction of laser, by default +1
    k_wavevector : float, optional
        Wavevector magnitude, by default K_WAVEVECTOR

    Returns
    -------
    tuple
        (m_values, squiggly_amplitudes_out, state_is_ground, positions_out, velocities)
        Positions are updated ballistically from velocities.
        Velocities are returned unchanged.
    """

    squiggly_amplitudes_out = np.empty_like(squiggly_amplitudes)
    positions_out = np.empty_like(positions)
    for idx in range(len(m_values)):
        is_ground = state_is_ground[idx]
        if is_ground:
            m_ground = m_values[idx]
        else:
            m_ground = m_values[idx] - k_sign

        Delta, delta_recoil, Omega_0_val, Omega_3 = _calculate_propagation_constants(
            omega_laser,
            k=k_wavevector,
            vz=vz,
            k_sign=k_sign,
            m_ground=m_ground,
        )

        phase = np.exp(1j * Omega_3 * time_of_propegation / 2)

        if is_ground:
            squiggly_amplitudes_out[idx] = squiggly_amplitudes[idx] * phase
        else:
            squiggly_amplitudes_out[idx] = squiggly_amplitudes[idx] * np.conj(phase)

        # Update position ballistically for all three dimensions.
        # velocities[idx, 2] already encodes v_0 + m * v_recoil accumulated from
        # recoil kicks during previous pulses, so no separate m-term is needed here.
        positions_out[idx] = positions[idx] + velocities[idx] * time_of_propegation

    return m_values, squiggly_amplitudes_out, state_is_ground, positions_out, velocities


def pulse_interaction_in_borde_representation(
    m_values: np.ndarray,
    squiggly_amplitudes: np.ndarray,
    internal_is_ground: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    pulse_detuning: float,
    t_pulse: float,
    pulse_rabi_freq,
    pulse_phase=0.0,
    k_sign=+1,
    k_wavevector=K_WAVEVECTOR,
    vz: float = 0.0,
):
    """
    Apply a laser pulse in the Borde representation

    Each pulse couples |a, m> (ground, momentum m) to |b, m+1> (excited,
    momentum m+1) for a +k laser, or |b, m-1> for a -k laser.

    Since we track each state independently (so that we can consider spatial
    overlap) this doubles the size of the state vector each pulse.

    Parameters
    ----------
    m_values : np.ndarray
        Momentum quantum numbers for each state row
    squiggly_amplitudes : np.ndarray
        Amplitudes in Borde representation
    internal_is_ground : np.ndarray
        Boolean array indicating ground (True) or excited (False) state
    positions : np.ndarray, shape (N, 3)
        [x, y, z] positions of each state row (classical tracking)
    velocities : np.ndarray, shape (N, 3)
        [vx, vy, vz] velocities of each state row. vx and vy are constant.
        vz includes accumulated recoil kicks from previous pulses.
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

    Returns
    -------
    tuple
        (new_m_values, new_squiggly_amplitudes, new_is_ground, new_positions, new_velocities)
        Positions are updated with midpoint approximation for m-changing branches.
        Velocities: vx and vy unchanged, vz updated by recoil kick for m-changing branches.
    """

    # Implement equation 13 / 14 / 15

    # Prepare output arrays -- each row branches into two
    N = squiggly_amplitudes.shape[0]
    new_num_rows = N * 2

    # Broadcast pulse_rabi_freq to a per-row array
    rabi_arr = np.broadcast_to(np.asarray(pulse_rabi_freq, dtype=float), (N,)).copy()

    new_squiggly_amplitudes = np.empty(new_num_rows, dtype=squiggly_amplitudes.dtype)
    new_m_values = np.empty(new_num_rows, dtype=m_values.dtype)
    new_positions = np.empty((new_num_rows, 3), dtype=positions.dtype)
    new_velocities = np.empty((new_num_rows, 3), dtype=velocities.dtype)
    new_is_ground = np.empty(new_num_rows, dtype=internal_is_ground.dtype)

    # Ground-output rows first, excited-output rows second
    ind_excited = N
    new_is_ground[:ind_excited] = True
    new_is_ground[ind_excited:] = False

    for idx in range(N):
        # Build input state vector for propagate_pulse
        # Borde notation: state = [excited_amp, ground_amp] (b, a)
        if internal_is_ground[idx]:
            # The ground state has m = m_a, so the relevant excited state for
            # the pulse is m = m_a +- 1
            m_ground = m_values[idx]
            m_excited = m_values[idx] + k_sign
            state = np.array([0, squiggly_amplitudes[idx]], dtype=complex)
        else:
            # This excited state has m = m_b, so the relevant ground state for
            # the pulse is m = m_b -+ 1
            m_ground = m_values[idx] - k_sign
            m_excited = m_values[idx]
            state = np.array([squiggly_amplitudes[idx], 0], dtype=complex)

        # Borde uses omega_ab = pi * RABI_FREQ, angular frequencies in rad/s
        omega_ab = np.pi * rabi_arr[idx]
        omega_laser = 2 * np.pi * (TRANSITION_FREQUENCY + pulse_detuning)

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

        amplitude_vector_out = prop_matrix @ state

        # Build 3D velocity vectors for the two output branches.
        # vz for each branch is computed from the reference velocity plus
        # m * RECOIL_VELOCITY, consistent with the invariant
        # velocities[idx, 2] == vz + m_values[idx] * RECOIL_VELOCITY.
        # vx and vy are taken from the input velocities (unchanged).
        vz_ground = vz + m_ground * RECOIL_VELOCITY
        vz_excited = vz + m_excited * RECOIL_VELOCITY

        vel_ground_3d = np.array([velocities[idx, 0], velocities[idx, 1], vz_ground])
        vel_excited_3d = np.array([velocities[idx, 0], velocities[idx, 1], vz_excited])

        # Ground-output branch: m = m_ground, velocity = vel_ground_3d
        new_squiggly_amplitudes[idx] = amplitude_vector_out[1]
        new_m_values[idx] = m_ground
        new_velocities[idx] = vel_ground_3d

        # Excited-output branch: m = m_excited, velocity = vel_excited_3d
        new_squiggly_amplitudes[ind_excited + idx] = amplitude_vector_out[0]
        new_m_values[ind_excited + idx] = m_excited
        new_velocities[ind_excited + idx] = vel_excited_3d

        # Update the positions. If this state didn't change (i.e. ground->ground
        # or excited->excited) we can just use the input velocity for the whole
        # pulse duration. If it did change, we use the midpoint approximation:
        # half the pulse with the old velocity, half the pulse with the new
        # velocity.
        if internal_is_ground[idx]:  # start in ground
            # ground->ground
            new_positions[idx] = positions[idx] + vel_ground_3d * t_pulse
            # ground->excited
            new_positions[ind_excited + idx] = (
                positions[idx]
                + vel_ground_3d * (t_pulse / 2)
                + vel_excited_3d * (t_pulse / 2)
            )
        else:  # start in excited
            # excited->excited
            new_positions[idx] = positions[idx] + vel_excited_3d * t_pulse
            # excited->ground
            new_positions[ind_excited + idx] = (
                positions[idx]
                + vel_ground_3d * (t_pulse / 2)
                + vel_excited_3d * (t_pulse / 2)
            )

    return (
        new_m_values,
        new_squiggly_amplitudes,
        new_is_ground,
        new_positions,
        new_velocities,
    )


def change_laser_frequency_in_borde_representation(
    m_values: np.ndarray,
    squiggly_amplitudes: np.ndarray,
    internal_is_ground: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
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
    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities
        The state arrays.  Only ``squiggly_amplitudes`` is modified; the others
        are returned unchanged.
    old_detuning_hz : float
        Detuning (Hz) of the Bordé frame the amplitudes are currently in.
    new_detuning_hz : float
        Detuning (Hz) of the Bordé frame to express the amplitudes in.
    time : float
        **Global** simulation time at which the frame change happens, in seconds.

    Returns
    -------
    tuple
        ``(m_values, squiggly_amplitudes, internal_is_ground, positions,
        velocities)``
    """
    delta_f = new_detuning_hz - old_detuning_hz
    phase_gnd = np.exp(-1j * np.pi * delta_f * time)
    phase_exc = np.exp(+1j * np.pi * delta_f * time)
    phase = np.where(internal_is_ground, phase_gnd, phase_exc)
    return (
        m_values,
        squiggly_amplitudes * phase,
        internal_is_ground,
        positions,
        velocities,
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
    m_values,
    squiggly_amplitudes,
    internal_is_ground,
    positions,
    velocities,
    pulse_detuning,
    t_pulse,
    on_axis_rabi_freq,
    beam_waist,
    pulse_phase=0.0,
    k_sign=+1,
    k_wavevector=K_WAVEVECTOR,
    vz=0.0,
    wavelength=TRANSITION_WAVELENGTH,
):
    """Apply a laser pulse with a full 3-D Gaussian (TEM00) intensity profile.

    Computes the per-row Rabi frequency at the pulse midpoint using the
    TEM00 Gaussian profile — including the Rayleigh-range z-dependence — then
    calls pulse_interaction_in_borde_representation with those per-row
    frequencies.  The beam waist is assumed to be located at z = 0 (the
    atom's initial position).

    Parameters
    ----------
    m_values : np.ndarray
        Momentum quantum numbers for each state row.
    squiggly_amplitudes : np.ndarray
        Amplitudes in Borde representation.
    internal_is_ground : np.ndarray
        Boolean array, True for ground-state rows.
    positions : np.ndarray, shape (N, 3)
        [x, y, z] positions of each state row.
    velocities : np.ndarray, shape (N, 3)
        [vx, vy, vz] velocities of each state row.
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

    Returns
    -------
    tuple
        (new_m_values, new_squiggly_amplitudes, new_is_ground, new_positions, new_velocities)
    """
    # Compute 3-D position at pulse midpoint for Gaussian Rabi calculation
    positions_mid = positions + velocities * (t_pulse / 2)
    rabi_per_row = gaussian_rabi(
        positions_mid, on_axis_rabi_freq, beam_waist, wavelength
    )
    return pulse_interaction_in_borde_representation(
        m_values,
        squiggly_amplitudes,
        internal_is_ground,
        positions,
        velocities,
        pulse_detuning=pulse_detuning,
        t_pulse=t_pulse,
        pulse_rabi_freq=rabi_per_row,
        pulse_phase=pulse_phase,
        k_sign=k_sign,
        k_wavevector=k_wavevector,
        vz=vz,
    )


def calculate_ground_and_excited_probabilities(
    m_values, internal_amplitude, internal_is_ground
):
    unique_m = np.unique(m_values)
    ground_prob = 0.0
    excited_prob = 0.0

    for m in unique_m:
        k_this_m = m_values == m
        total_gnd_amp = np.sum(internal_amplitude[k_this_m & internal_is_ground])
        total_exc_amp = np.sum(internal_amplitude[k_this_m & ~internal_is_ground])

        ground_prob += np.abs(total_gnd_amp) ** 2
        excited_prob += np.abs(total_exc_amp) ** 2

    return ground_prob, excited_prob


def do_clearout(
    m_values,
    squiggly_amplitudes,
    internal_is_ground,
    positions,
    velocities,
    rng=None,
):
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
    m_values : np.ndarray
        Momentum quantum numbers for each state row.
    squiggly_amplitudes : np.ndarray
        Amplitudes in the Bordé representation.
    internal_is_ground : np.ndarray
        Boolean array, True for ground-state rows.
    positions : np.ndarray, shape (N, 3)
        [x, y, z] positions of each state row.
    velocities : np.ndarray, shape (N, 3)
        [vx, vy, vz] velocities of each state row.
    rng : np.random.Generator, optional
        Random-number generator.  If ``None``, ``np.random.default_rng()``
        is used (avoids the legacy global random state).

    Returns
    -------
    None
        If the atom is projected to ground (discarded).
    tuple
        ``(m_values, squiggly_amplitudes, internal_is_ground, positions,
        velocities)`` with ground rows removed and excited amplitudes
        renormalised so the wavefunction has unit norm.
    """
    if rng is None:
        rng = np.random.default_rng()

    p_g, p_e = calculate_ground_and_excited_probabilities(
        m_values, squiggly_amplitudes, internal_is_ground
    )

    total_prob = p_g + p_e
    if np.isclose(total_prob, 0.0):
        # Empty state from a prior clearout -- treat as already discarded
        return None

    u = rng.uniform()
    if u < p_g / total_prob:
        # Projected to ground -- discard
        return None

    # Survived -- keep only excited rows and renormalise
    keep = ~internal_is_ground
    return (
        m_values[keep],
        squiggly_amplitudes[keep] * (1.0 / np.sqrt(p_e)),
        internal_is_ground[keep],
        positions[keep],
        velocities[keep],
    )


def run_clearout_trials(sequence_fn, n_trials, rng=None):
    """Run ``sequence_fn(rng)`` ``n_trials`` times.

    The closure must return either ``None`` (atom discarded mid-sequence)
    or the final state tuple.

    For surviving runs, each contributes its quantum-mechanical
    :math:`P_\\text{g}` and :math:`P_\\text{e}` weighted by
    :math:`1/n_\\text{trials}` (so the result equals the deterministic
    population breakdown in the limit :math:`n_\\text{trials} \\to \\infty`).

    Parameters
    ----------
    sequence_fn : callable
        ``sequence_fn(rng)`` -> ``None`` or state tuple.
    n_trials : int
        Number of Monte-Carlo trials to run.
    rng : np.random.Generator, optional
        Random-number generator.  If ``None``, ``np.random.default_rng()``
        is used.

    Returns
    -------
    tuple
        ``(p_ground, p_excited, p_discarded)`` -- population fractions.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    ground_tally = 0.0
    excited_tally = 0.0
    discard_tally = 0.0

    for _ in range(n_trials):
        result = sequence_fn(rng)
        if result is None:
            discard_tally += 1.0
        else:
            m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
                result
            )
            p_g, p_e = calculate_ground_and_excited_probabilities(
                m_values, squiggly_amplitudes, internal_is_ground
            )
            ground_tally += p_g
            excited_tally += p_e

    return (
        ground_tally / n_trials,
        excited_tally / n_trials,
        discard_tally / n_trials,
    )


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


def calc_mz_excitation(
    phi,
    detuning_hz=RECOIL_FREQUENCY_HZ,
    initial_velocity_z=0.0,
    time_between_pulses=200e-6,
):
    """Compute excitation fraction for Mach-Zehnder sequence.

    Pulse sequence: π/2 (phase=0) - π (phase=φ) - π/2 (phase=4φ)
    Uses proper quantum state evolution with coherent sum over paths.

    Parameters
    ----------
    phi : float
        Phase parameter in radians
    detuning_hz : float
        Laser detuning from resonance in Hz (default: recoil shift)
    initial_velocity_z : float
        Initial atom velocity in m/s
    time_between_pulses : float
        Time between pulses in seconds (default: 0.0)

    Returns
    -------
    float
        Excitation fraction after full sequence
    """

    pulse_sequence = build_mach_zehnder_pulse_sequence(
        phi=phi,
        detuning_hz=detuning_hz,
        time_between_pulses=time_between_pulses,
        rabi_frequency=RABI_FREQ,
        pulse_area_multiplier=1.0,
        k=+1,
    )

    return calculate_excited_fraction_for_pulse_sequence(
        pulse_sequence,
        initial_velocity_z=initial_velocity_z,
    )


if __name__ == "__main__":
    ######
    # Demo: single pulse on a stationary atom
    initial_velocity_z = 0.0
    t_propagate = 1e-3
    m_values, positions, velocities, internal_amplitude, internal_is_ground = (
        make_atom_states(initial_velocity_z=initial_velocity_z)
    )

    squiggly_amplitudes = transform_state_vector(
        m_values,
        internal_amplitude,
        internal_is_ground,
        omega_laser=2 * np.pi * TRANSITION_FREQUENCY,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )

    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
        propagate_states_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
            velocities,
            vz=initial_velocity_z,
            time_of_propegation=t_propagate,
            omega_laser=2 * np.pi * TRANSITION_FREQUENCY,
        )
    )

    m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = (
        pulse_interaction_in_borde_representation(
            m_values,
            squiggly_amplitudes,
            internal_is_ground,
            positions,
            velocities,
            pulse_detuning=0.0,
            t_pulse=T_PI,
            pulse_rabi_freq=RABI_FREQ,
            pulse_phase=0.0,
            k_sign=+1,
            k_wavevector=K_WAVEVECTOR,
            vz=initial_velocity_z,
        )
    )

    # Convert back to lab frame
    internal_amplitude_after_pulse = transform_state_vector(
        m_values,
        squiggly_amplitudes,
        internal_is_ground,
        omega_laser=2 * np.pi * TRANSITION_FREQUENCY,
        t=T_PI + t_propagate,
        z=0.0,
        vz=initial_velocity_z,
        inverse=True,
    )

    internal_amplitude_after_pulse, m_values, internal_is_ground

    ######

    _are_states_normalized(m_values, internal_amplitude_after_pulse, internal_is_ground)

    ######
    # Frequency scan: ±100 kHz around resonance with pi pulse

    detunings_hz = np.linspace(-100e3, 100e3, 200)
    excitation_fractions = [
        do_rabi_pulse(detuning, pulse_duration=T_PI, initial_velocity_z=0.0)
        for detuning in detunings_hz
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(detunings_hz / 1e3, excitation_fractions, "b-", linewidth=1.5)
    plt.axvline(0, color="r", linestyle="--", alpha=0.5, label="Resonance")
    plt.xlabel("Detuning (kHz)")
    plt.ylabel("Excitation Fraction")
    plt.title(r"Single $\pi$-pulse frequency scan ($v_z = 0$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-100, 100)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("/tmp/pi_pulse_freq_scan.png", dpi=150)
    print("Saved plot to /tmp/pi_pulse_freq_scan.png")

    ######
    # Mach-Zehnder interferometer: π/2 - π - π/2 sequence with phase scan
    # Pulse phases: 0, φ, 4φ
    # Detuning set to recoil shift

    plt.figure(figsize=(10, 6))

    for detuning in [0.0, RECOIL_FREQUENCY_HZ, 2 * RECOIL_FREQUENCY_HZ]:

        phis = np.linspace(0, 2 * np.pi, 500)
        mz_excitations = [
            calc_mz_excitation(
                phi,
                detuning_hz=detuning,
                initial_velocity_z=0.0,
                time_between_pulses=200e-6,
            )
            for phi in phis
        ]

        plt.plot(
            phis / np.pi,
            mz_excitations,
            label=f"Detuning = {detuning/1e3:.1f} kHz",
        )

    plt.xlabel(r"$\phi$ ($\pi$ rad)")
    plt.ylabel("Excitation Fraction")
    plt.title(
        r"Mach-Zehnder: $\pi/2$-$\pi$-$\pi/2$ sequence at recoil shift "
        r"($\phi_1=0$, $\phi_2=\phi$, $\phi_3=4\phi$)"
    )
    plt.xticks(
        [0, 0.5, 1.0, 1.5, 2.0], ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    )
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2)
    plt.ylim(0, 1.05)
    plt.legend()
    vs.tag_plot(small=True)

    plt.savefig("/tmp/mz_phase_scan.png", dpi=150)
    print("Saved plot to /tmp/mz_phase_scan.png")
