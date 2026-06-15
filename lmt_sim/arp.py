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
:func:`lmt_sim.lmt_simulation._single_pulse_propagator_2x2`. The inter-block
frame phase uses a local :func:`_frame_change_phases` (a paused-state copy of the
old, since-removed core primitive -- see below).

.. warning::

    **WORK IN PROGRESS / PAUSED -- the composer is currently physically wrong.**

    The production row-based composer no longer applies any inter-block frame
    change: it rebases the Bordé frame at a laser-frequency step without touching
    the amplitudes (``change_laser_frequency_in_borde_representation``), and the
    old ``_frame_change_phases`` primitive was removed from the core. This module
    still applies the old (wrong) inter-block frame change via a LOCAL copy of
    that function, so it is now out of step with the row composer and remains
    paused. :func:`compose_arp_2x2` applies the local frame change between
    sub-pulses.
    For a chirp this *double-counts* the laser-frequency change: the detuning is
    already carried in each block's Hamiltonian diagonal (``Omega_3``), so the
    extra frame rotation injects a spurious phase. The staircase then converges
    (n-independent) to the wrong answer -- e.g. it disagrees with the textbook
    continuous-sweep ODE and with Landau-Zener by a factor of 2 in the exponent.

    Removing the frame change (composing the per-block propagators directly)
    reproduces the ODE and Landau-Zener exactly. This is the corrected path, but
    it is **not** applied here yet: building the ARP composer is paused pending a
    deeper look at where/whether the row composer's frame change is correct (it
    is reused, as planned, for pulses separated by free evolution). See
    ``docs/arp_frame_change_finding.md`` for the full diagnosis and reproduction.
"""

from dataclasses import dataclass

import numpy as np

from lmt_sim import lmt_simulation as sim


def _frame_change_phases(old_detuning_hz, new_detuning_hz, time):
    r"""Per-state Bordé frame-change phases for a laser-frequency change.

    Returns ``(phase_excited, phase_ground)``: the factors to multiply the
    excited- and ground-state amplitudes by when re-expressing them from the
    ``old`` to the ``new`` laser-frequency frame at **global** ``time``. Excited
    gains ``exp(+i pi Df t)`` and ground ``exp(-i pi Df t)`` with
    ``Df = new - old`` (Hz).

    FIXME(frame-change): This phase is the WRONG tool for a chirp -- it
    double-counts the laser-frequency change (the detuning already lives in each
    block's Hamiltonian diagonal). It used to live in :mod:`lmt_sim.lmt_simulation`
    as the shared row/ARP frame-change primitive, but the row path has been fixed
    (it now rebases via ``change_laser_frequency_in_borde_representation`` and does
    not use this phase). This local copy is kept ONLY so the PAUSED ARP composer
    below keeps its previous (known-wrong) behaviour without depending on a symbol
    that no longer exists in the core. Do not use it for new code. See
    docs/arp_frame_change_finding.md.
    """
    delta_f = new_detuning_hz - old_detuning_hz
    phase_exc = np.exp(+1j * np.pi * delta_f * time)
    phase_gnd = np.exp(-1j * np.pi * delta_f * time)
    return phase_exc, phase_gnd


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
    :func:`lmt_sim.lmt_simulation._single_pulse_propagator_2x2`). The global
    clock starts at 0 and advances by each sub-pulse duration, mirroring
    ``iter_pulse_sequence_in_borde_representation`` so the frame bookkeeping is
    identical to the row-based composer.

    Parameters
    ----------
    subpulses : sequence[ARPSubPulse]
        The staircase, e.g. from :func:`make_arp_subpulses`.
    k_sign, vz, m_ground : optional
        Single-arm parameters passed to the per-pulse propagator.
    pulse_phase : float, optional
        Optical phase (rad), constant across the staircase for a pure-chirp ARP.
    ref_detuning_hz : float or None, optional
        If given, append a final frame change to this fixed reference frame at
        ``t = T_total`` so the result does not live in the (error-dependent)
        frame of the last sub-pulse -- important when comparing imprinted phases
        across a parameter scan.

    Returns
    -------
    numpy.ndarray
        The 2x2 complex propagator ``U``.
    """
    if len(subpulses) == 0:
        raise ValueError("compose_arp_2x2 needs at least one sub-pulse")

    U = np.eye(2, dtype=complex)
    current_detuning = subpulses[0].detuning_hz
    current_time = 0.0

    for sub in subpulses:
        if sub.detuning_hz != current_detuning:
            # FIXME(frame-change): This inter-block frame change is WRONG for a
            # back-to-back chirp. The detuning is already carried in each block's
            # Hamiltonian diagonal (Omega_3), so applying exp(-/+ i pi Df t) here
            # double-counts the laser-frequency change and converges to the wrong
            # physics (off from the continuous-sweep ODE / Landau-Zener by a
            # factor of 2 in the exponent). The correct staircase is the plain
            # product of the per-block propagators -- delete this block.
            # See docs/arp_frame_change_finding.md.
            phase_exc, phase_gnd = _frame_change_phases(
                current_detuning, sub.detuning_hz, current_time
            )
            U = np.diag([phase_exc, phase_gnd]) @ U
            current_detuning = sub.detuning_hz

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
        current_time += sub.duration

    if ref_detuning_hz is not None and ref_detuning_hz != current_detuning:
        # FIXME(frame-change): For the corrected (no inter-block frame change)
        # composer, re-referencing the imprinted phase across a parameter scan
        # needs an integral-of-laser-phase correction (exp(+/- i 2pi delta_err T)),
        # NOT this frame change. See docs/arp_frame_change_finding.md.
        phase_exc, phase_gnd = _frame_change_phases(
            current_detuning, ref_detuning_hz, current_time
        )
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
    c_excited, c_ground = U @ np.array(
        [initial_excited, initial_ground], dtype=complex
    )
    return c_excited, c_ground
