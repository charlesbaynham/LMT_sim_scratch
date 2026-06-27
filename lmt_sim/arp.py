"""Adiabatic rapid passage (ARP) pulses for the LMT simulation.

ARP needs a swept laser detuning ``Delta(t)``, which the simulation cannot
represent as a single time-varying pulse. We approximate the sweep as a
**staircase**: a sequence of short fixed-``(Delta, Omega, phi)`` sub-pulses, the
piecewise-constant case the Bordé machinery already supports.

Scope: single arm, two levels only. For a single co-propagating arm
(``k_sign=+1``) the states ``|ground, m>`` and ``|excited, m+1>`` form a *closed*
two-level system, so the full row-based composer's ``2**N`` row growth over ``N``
sub-pulses is pure redundancy. This module composes the **2x2** propagator
directly, but it does **not** reimplement the pulse physics: the per-pulse
propagator comes from
:func:`lmt_sim.lmt_simulation._single_pulse_propagator_2x2`, the same primitive
the production row-based composer applies per row.

Inter-block bookkeeping mirrors the (corrected) row path. Across a laser-detuning
step the row composer does **not** touch the amplitudes
(``change_laser_frequency_in_borde_representation`` only records the laser-phase
integral on the state); the staircase is therefore simply the product of the
per-block propagators. A chirp written as instantaneous detuning in each block's
Hamiltonian diagonal (``Omega_3``) is complete on its own -- applying an extra
inter-block frame rotation would double-count the laser-frequency change (the bug
diagnosed in ``docs/arp_frame_change_finding.md``). The composer reproduces the
textbook continuous-sweep ODE and Landau-Zener; ``tests/test_arp.py`` pins it to
the row composer and to those analytic limits.

The laser-phase integral the row path defers to the lab boundary
(``transform_state_vector`` applies ``exp(+/- i pi Phi)`` with
``Phi = sum_k detuning_k * dt_k``) is reproduced here on demand by the
``ref_detuning_hz`` argument of :func:`compose_arp_2x2`, so an imprinted phase can
be referenced to a fixed frame for cross-scan comparison.
"""

from dataclasses import dataclass

import numpy as np

from lmt_sim import lmt_simulation as sim


@dataclass(frozen=True)
class ARPSubPulse:
    """One fixed-parameter block of a staircase ARP pulse."""

    detuning_hz: float
    rabi_freq_hz: float
    duration: float


def resonant_centre_detuning_hz(m_ground=0, k_sign=+1, vz=0.0, k=sim.K_WAVEVECTOR):
    """Laser detuning (Hz) that places the two-level system on resonance.

    The dynamics see the recoil-shifted detuning ``Omega_3`` (Bordé Eq. 7), not
    the bare laser detuning, so ``detuning_hz = 0`` is *not* resonant. This
    returns the detuning for which ``Omega_3 = 0`` (the centre an ARP sweep
    should be symmetric about). For ``m_ground=0, k_sign=+1, vz=0`` this is
    ``RECOIL_FREQUENCY_HZ``.
    """
    _, _, _, omega_3_at_zero = sim._borde_frame_constants(
        0.0, k_sign=k_sign, k=k, vz=vz, m_ground=m_ground
    )
    # Omega_3(detuning) = 2*pi*detuning + Omega_3(0); solve Omega_3 = 0.
    return -omega_3_at_zero / (2 * np.pi)


def _sweep_detuning(t, T, delta_centre_hz, delta_sweep_hz, sweep_shape, tanh_beta):
    """Detuning (Hz) at time ``t`` in ``[0, T]`` for the chosen sweep shape."""
    if sweep_shape == "linear":
        return delta_centre_hz + delta_sweep_hz * (t / T - 0.5)
    if sweep_shape == "tanh":
        # Normalised so the endpoints reach exactly +/- delta_sweep/2.
        return delta_centre_hz + (delta_sweep_hz / 2.0) * (
            np.tanh(tanh_beta * (2.0 * t / T - 1.0)) / np.tanh(tanh_beta)
        )
    raise ValueError(f"Unknown sweep_shape: {sweep_shape!r}")


def _amplitude_envelope(t, T, omega0_hz, omega_shape):
    """Rabi frequency (Hz) at time ``t`` in ``[0, T]`` for the chosen envelope."""
    if omega_shape == "const":
        return np.full_like(np.asarray(t, dtype=float), omega0_hz)
    if omega_shape == "sin2":
        return omega0_hz * np.sin(np.pi * t / T) ** 2
    if omega_shape == "blackman":
        x = t / T
        window = 0.42 - 0.5 * np.cos(2 * np.pi * x) + 0.08 * np.cos(4 * np.pi * x)
        return omega0_hz * window
    raise ValueError(f"Unknown omega_shape: {omega_shape!r}")


def make_arp_subpulses(
    T,
    delta_sweep_hz,
    omega0_hz,
    *,
    n=400,
    sweep_shape="tanh",
    omega_shape="sin2",
    tanh_beta=3.0,
    m_ground=0,
    k_sign=+1,
    vz=0.0,
    delta_centre_hz=None,
):
    """Build the staircase of fixed-parameter sub-pulses for an ARP pulse.

    Parameters
    ----------
    T : float
        Total pulse duration (s).
    delta_sweep_hz : float
        Total swept detuning range (Hz), centred on ``delta_centre_hz``.
    omega0_hz : float
        Peak Rabi frequency (Hz).
    n : int, optional
        Number of sub-pulses (staircase resolution), by default 400.
    sweep_shape : {'tanh', 'linear'}, optional
        Detuning sweep shape, by default 'tanh'.
    omega_shape : {'sin2', 'const', 'blackman'}, optional
        Rabi-frequency envelope, by default 'sin2'.
    tanh_beta : float, optional
        Steepness of the tanh sweep, by default 3.0.
    m_ground, k_sign, vz : optional
        Single-arm parameters; used to find the resonant centre and passed
        through to the propagator.
    delta_centre_hz : float or None, optional
        Centre of the detuning sweep (Hz). ``None`` (default) uses
        :func:`resonant_centre_detuning_hz` so the sweep is symmetric about true
        two-level resonance.

    Returns
    -------
    list[ARPSubPulse]
        ``n`` sub-pulses sampled at bin centres ``t_k = (k + 0.5) * T / n``.
    """
    if delta_centre_hz is None:
        delta_centre_hz = resonant_centre_detuning_hz(
            m_ground=m_ground, k_sign=k_sign, vz=vz
        )

    dt = T / n
    t_k = (np.arange(n) + 0.5) * dt
    detunings = _sweep_detuning(
        t_k, T, delta_centre_hz, delta_sweep_hz, sweep_shape, tanh_beta
    )
    rabis = _amplitude_envelope(t_k, T, omega0_hz, omega_shape)

    if np.any(rabis <= 0.0):
        raise ValueError(
            "ARP sub-pulse Rabi frequency is non-positive; the bin-centre "
            "sampling should keep all envelope samples strictly positive."
        )

    return [
        ARPSubPulse(detuning_hz=float(d), rabi_freq_hz=float(r), duration=float(dt))
        for d, r in zip(detunings, rabis)
    ]


def compose_arp_2x2(
    subpulses,
    *,
    k_sign=+1,
    vz=0.0,
    m_ground=0,
    pulse_phase=0.0,
    ref_detuning_hz=None,
):
    """Compose the sub-pulses into a single 2x2 Bordé propagator.

    The returned matrix ``U`` acts on the amplitude vector ordered
    ``[c_excited, c_ground]`` (the convention of
    :func:`lmt_sim.lmt_simulation._single_pulse_propagator_2x2`).

    The staircase is the plain product of the per-block propagators -- no
    inter-block frame change. The chirp is carried entirely by the per-block
    detuning in each Hamiltonian diagonal (``Omega_3``); the row composer rebases
    the Bordé frame at a detuning step without touching the amplitudes
    (``change_laser_frequency_in_borde_representation``), so the Bordé-frame
    amplitudes are exactly this product. This reproduces the continuous-sweep ODE
    and Landau-Zener (see docs/arp_frame_change_finding.md).

    Parameters
    ----------
    subpulses : sequence[ARPSubPulse]
        The staircase, e.g. from :func:`make_arp_subpulses`.
    k_sign, vz, m_ground : optional
        Single-arm parameters passed to the per-pulse propagator.
    pulse_phase : float, optional
        Optical phase (rad), constant across the staircase for a pure-chirp ARP.
    ref_detuning_hz : float or None, optional
        ``None`` (default) leaves ``U`` in the instantaneous Bordé frame -- the
        product of the per-block propagators, directly comparable to the row
        composer in the Bordé representation.

        If given, apply the laser-phase integral the row path defers to the lab
        boundary, referenced to a fixed frame at ``ref_detuning_hz``:
        ``exp(-/+ i pi (Phi - ref_detuning_hz * T_total))`` (excited -, ground +,
        the same detuning-part sign as ``transform_state_vector(inverse=True)``),
        where ``Phi = sum_k detuning_k * dt_k`` (cycles) is the genuine integral
        of the laser detuning over the staircase. This fixes the imprinted phase
        to a frame independent of the (error-dependent) last sub-pulse, so phases
        can be compared across a parameter scan. (Note this is the rotating-frame
        imprinted phase: like the rest of this 2x2 composer it omits the trivial
        ``omega_0 * t`` transition-frequency evolution that the full lab transform
        also carries.)

    Returns
    -------
    numpy.ndarray
        The 2x2 complex propagator ``U``.
    """
    if len(subpulses) == 0:
        raise ValueError("compose_arp_2x2 needs at least one sub-pulse")

    U = np.eye(2, dtype=complex)
    phi_cycles = 0.0  # integral of the laser detuning, sum_k detuning_k * dt_k
    total_time = 0.0

    for sub in subpulses:
        P = sim._single_pulse_propagator_2x2(
            sub.detuning_hz,
            sub.duration,
            sub.rabi_freq_hz,
            pulse_phase=pulse_phase,
            k_sign=k_sign,
            vz=vz,
            m_ground=m_ground,
        )
        U = P @ U
        phi_cycles += sub.detuning_hz * sub.duration
        total_time += sub.duration

    if ref_detuning_hz is not None:
        # Laser-phase integral relative to a fixed reference frame, with the SAME
        # sign convention the lab boundary (transform_state_vector, inverse=True)
        # uses for the detuning part: excited exp(-i pi Phi), ground exp(+i pi Phi).
        # Offset so the reference is ref_detuning_hz rather than 0. This is NOT a
        # frame change; the detuning is already in each block's diagonal.
        residual_cycles = phi_cycles - ref_detuning_hz * total_time
        phase_exc = np.exp(-1j * np.pi * residual_cycles)
        phase_gnd = np.exp(+1j * np.pi * residual_cycles)
        U = np.diag([phase_exc, phase_gnd]) @ U

    return U


def arp_excited_ground_amplitudes(
    subpulses,
    *,
    initial_excited=0.0 + 0.0j,
    initial_ground=1.0 + 0.0j,
    **compose_kwargs,
):
    """Apply an ARP staircase to an initial state, returning ``(c_excited, c_ground)``.

    Convenience wrapper around :func:`compose_arp_2x2`. Defaults start in the
    ground state.
    """
    U = compose_arp_2x2(subpulses, **compose_kwargs)
    c_excited, c_ground = U @ np.array([initial_excited, initial_ground], dtype=complex)
    return c_excited, c_ground
