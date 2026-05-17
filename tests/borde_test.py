import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import expm


# # Borde's Optical Ramsey Fringes
#
# Following the notation in equations 6-12 of Borde's "Optical Ramsey Fringes with Travelling Waves" paper.

t_pi = 45e-6  # pi pulse duration from experiment
omega_ab = np.pi / (
    2 * t_pi
)  # Rabi frequency, defined weirdly with a factor of two by Borde

pulse_phase = 0.0  # phase of the laser pulse
k_sign = +1  # or -1 depending on the direction of the laser beam
k = 2 * np.pi / 698e-9  # wavevector of the clock light
z = 0  # position of the atom along the laser beam direction at t=0
vz = 0.0  # velocity of the atom along the laser beam direction
omega_0 = 2 * np.pi * constants.c / 698e-9  # angular frequency of the clock transition
M = 87 * constants.atomic_mass  # mass of the atom (strontium-87)


def propagate_pulse(
    state,
    omega_laser,
    t,
    omega_ab=omega_ab,
    pulse_phase=pulse_phase,
    k_sign=k_sign,
    k=k,
    z=z,
    vz=vz,
    omega_0=omega_0,
):
    delta_E = constants.hbar * omega_0  # energy difference between the two states
    Delta = omega_laser - omega_0
    delta_recoil = constants.hbar * k**2 / (2 * M)  # recoil frequency shift

    m = 0  # momentum state

    # Equation 7
    # Omega_1 = 2 * omega_ab * np.cos(pulse_phase)
    # Omega_2 = 2 * omega_ab * np.sin(pulse_phase)
    Omega_3 = Delta - k_sign * k * vz - ((m + k_sign) ** 2 - m**2) * delta_recoil

    # Equation 8
    Omega_0_val = -((m + k_sign) ** 2 + m**2) * delta_recoil - (2 * m + k_sign) * k * vz

    # Pauli matrices
    S0 = np.array([[1, 0], [0, 1]])
    # S1 = np.array([[0, 1], [1, 0]])
    # S2 = np.array([[0, -1j], [1j, 0]])
    # S3 = np.array([[1, 0], [0, -1]])

    # Equation 4 transformation
    U = np.exp(1j * (0 + delta_E) * t / constants.hbar) * np.diag(
        [
            np.exp(1j / 2 * (omega_laser * t - 2 * (m + 2) * k * (z + vz * t))),
            np.exp(1j / 2 * (-omega_laser * t - 2 * m * k * (z + vz * t))),
        ]
    )

    # Eq 12
    Omega = np.sqrt(Omega_3**2 + 4 * omega_ab**2)

    # Equation 13
    A = np.cos(Omega * t / 2) + 1j * Omega_3 / Omega * np.sin(Omega * t / 2)
    B = 2j * omega_ab / Omega * np.sin(Omega * t / 2)
    C = B
    D = np.cos(Omega * t / 2) - 1j * Omega_3 / Omega * np.sin(Omega * t / 2)

    # Equation 10
    matrix_operator = np.array(
        [[A, B * np.exp(-1j * pulse_phase)], [C * np.exp(1j * pulse_phase), D]]
    )

    # Equation 9
    vec_ba_s_t = expm(1j / 2 * Omega_0_val * S0 * t) @ matrix_operator @ state

    # Transform back (Equation 4)
    return U.conj().T @ vec_ba_s_t


def excitation_fraction(state):
    return np.real(state * np.conj(state))[0]


def calc_excitation_fraction(
    omega_laser,
    t,
    omega_ab=omega_ab,
    pulse_phase=pulse_phase,
    k_sign=k_sign,
    k=k,
    z=z,
    vz=vz,
    omega_0=omega_0,
):
    state = np.array([0, 1])  # ground state
    state = propagate_pulse(
        state,
        omega_laser,
        t,
        omega_ab=omega_ab,
        pulse_phase=pulse_phase,
        k_sign=k_sign,
        k=k,
        z=z,
        vz=vz,
        omega_0=omega_0,
    )
    return excitation_fraction(state)


d = 10
omega_lasers = np.linspace(-d * omega_ab, +d * omega_ab, 100)
omega_lasers += omega_0

pulse_params = dict(
    t=t_pi,
    omega_ab=omega_ab,
    pulse_phase=pulse_phase,
    k_sign=k_sign,
    k=k,
    z=z,
    vz=vz,
    omega_0=omega_0,
)

excitation_fractions = [
    calc_excitation_fraction(omega_laser, **pulse_params)
    for omega_laser in omega_lasers
]

M = 87 * constants.atomic_mass  # mass of the atom (strontium-87)
delta_recoil = constants.hbar * k**2 / (2 * M)  # recoil frequency shift

plt.figure()
plt.plot(
    (omega_lasers - omega_0) / (2 * np.pi),
    excitation_fractions,
    label="Excitation fraction",
)
plt.axvline(
    delta_recoil / (2 * np.pi), color="red", linestyle="--", label="Recoil shift"
)

plt.xlabel("Detuning (Hz)")
plt.ylabel("Excitation Fraction")
plt.title(r"Single $\pi$ pulse")
plt.legend()

# %%

# MZ pulse sequence with phase scan
# Phases: 0, phi, 4*phi for the three pulses respectively
# Laser detuning set to zero

phis = np.linspace(0, 2 * np.pi, 500)

base_params = dict(
    k_sign=k_sign,
    k=k,
    z=z,
    vz=vz,
    omega_0=omega_0,
)


def calc_phase_scan_excitation(phi, detuning=0.0):
    """Compute excitation fraction for a pi/2 - pi - pi/2 pulse sequence.

    Pulse phases are 0, phi, and 4*phi respectively.
    detuning is the laser detuning from resonance in rad/s.
    """
    state = np.array([0, 1])  # ground state
    omega_laser = omega_0 + detuning

    # pi pulse, phase = 0
    state = propagate_pulse(
        state,
        omega_laser,
        t_pi / 2,
        omega_ab=omega_ab,
        pulse_phase=0.0,
        **base_params,
    )

    # pi/2 pulse, phase = phi
    state = propagate_pulse(
        state,
        omega_laser,
        t_pi,
        omega_ab=omega_ab,
        pulse_phase=phi,
        **base_params,
    )

    # pi pulse, phase = 4*phi
    state = propagate_pulse(
        state,
        omega_laser,
        t_pi / 2,
        omega_ab=omega_ab,
        pulse_phase=4 * phi,
        **base_params,
    )

    return excitation_fraction(state)


# %% Scan phi at 5 different detunings

detunings = np.array([-2, -1, 0, 1, 2]) * delta_recoil  # rad/s
colors = plt.cm.viridis(np.linspace(0, 1, len(detunings)))

plt.figure()
for detuning, color in zip(detunings, colors):
    ef_phase_scan = [calc_phase_scan_excitation(phi, detuning=detuning) for phi in phis]
    detuning_hz = detuning / (2 * np.pi)
    plt.plot(phis / np.pi, ef_phase_scan, color=color, label=f"{detuning_hz:.0f} Hz")

plt.xlabel("phi (pi rad)")
plt.ylabel("Excitation Fraction")
plt.title(r"$\pi$ - $\pi/2$ - $\pi$ sequence, phase scan (phases: 0, $\phi$, $4\phi$)")
plt.xticks([0, 0.5, 1.0, 1.5, 2.0], ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
plt.legend(title="Detuning")


# %% Scan phi at recoil frequency

detuning = delta_recoil


plt.figure()


ef_phase_scan = [calc_phase_scan_excitation(phi, detuning=detuning) for phi in phis]
detuning_hz = detuning / (2 * np.pi)
plt.plot(phis / np.pi, ef_phase_scan)

plt.xlabel("phi (pi rad)")
plt.ylabel("Excitation Fraction")
plt.title(r"$\pi$ - $\pi/2$ - $\pi$ sequence, phase scan (phases: 0, $\phi$, $4\phi$)")
plt.xticks([0, 0.5, 1.0, 1.5, 2.0], ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
