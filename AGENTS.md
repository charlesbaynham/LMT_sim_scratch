# LMT Atom Interferometer Simulation

Minimal working example of a Large Momentum Transfer (LMT) atom interferometer simulation for Sr-87 on the 698 nm clock transition, using the formalism of Bordé (PhysRevA.30.1836).

## Architecture

The simulation tracks an ensemble of **state rows**, each representing a branch of the atom's wavefunction. A state row consists of:

- **Momentum quantum number `m`** `(N,)` int array — number of ℏk photon recoils relative to the initial momentum. Actual velocity of branch m is `v_0 + m * v_recoil`. 
- **Position** `(N,)` float array — z-position of each branch, propagated ballistically using `v(m) = v_0 + m * v_recoil`
- **Internal amplitude** `(N,)` complex128 array — complex coefficient for that branch
- **Internal label** `(N,)` bool array — `True` = ground state, `False` = excited state

The initial velocity `v_0` is a property of the atom (not per-row) and is passed as a parameter to the propagation and pulse functions.

Every laser pulse doubles the number of rows (each branch splits into ground/excited via the two-level Hamiltonian). Free evolution propagates positions ballistically and accumulates quantum phase.

## Key functions

### `make_atom_state_row(position_z, initial_velocity_z, c0, c1)`
Initialises two rows (ground + excited) for one atom. Both rows start at `m=0`.

### `propagate_state_freely(m_values, positions, internal_amplitude, internal_is_ground, time, initial_velocity_z=0, laser_detuning=0)`
Propagates all rows through a free-fall period of duration `time`. Updates positions ballistically using `v(m) = v_0 + m * v_recoil` and applies phase evolution to each amplitude:

1. **Kinetic phase**: $\exp(-i\, M\, v(m)^2\, t / 2\hbar)$ — from the kinetic energy of each branch, where `v(m) = v_0 + m * v_recoil`.
2. **Internal energy phase** (rotating frame): The detuning for each row is $\delta_i = \delta_\text{laser} - v(m) / \lambda$, including the Doppler shift. Ground-state rows accumulate $\exp(+i\pi\delta_i t)$ and excited-state rows accumulate $\exp(-i\pi\delta_i t)$.

Returns `(m_values, new_positions, new_amplitude)`.

### `do_pulse(m_values, positions, internal_amplitude, internal_is_ground, initial_velocity_z=0, laser_direction=+1, pulse_detuning=0, pulse_duration=T_PI, pulse_rabi_freq=RABI_FREQ, pulse_phase=0)`
Applies a laser pulse using Bordé's 2×2 formalism. For each row, builds the rotating-frame Hamiltonian:

$$H/\hbar = \pi\bigl(-\delta_\text{eff}(m)\,\sigma_z + \Omega\cos\phi\,\sigma_x + \Omega\sin\phi\,\sigma_y\bigr)$$

where the **effective detuning includes the m-dependent recoil shift** (Bordé Eqs. 7-8):

$$\delta_\text{eff}(m) = \Delta - v(m)/\lambda - (2m + d)\,\delta_\text{rec}$$

with $d = \pm 1$ being the laser direction and $\delta_\text{rec} = \hbar k^2/(2M)$ the single-photon recoil frequency. Computes $U = \exp(-i\,t\,H/\hbar)$ via `scipy.linalg.expm`. Doubles the row count and updates `m` for the excited branch (`m → m + d` for ground→excited transitions).

## Conventions

- All frequencies (Rabi, detuning) are in **Hz** (not angular). The factor of $2\pi$ from $\omega = 2\pi f$ combines with the $1/2$ from the Hamiltonian to give an overall $\pi$ prefactor.
- Momentum is parametrised as integer `m`: actual velocity is `v_0 + m * RECOIL_VELOCITY` where `RECOIL_VELOCITY = hbar * k / M`.
- `RECOIL_FREQUENCY_HZ = hbar * k^2 / (2M) / (2*pi)` is the recoil frequency in Hz.
- Detuning sign convention: `delta = laser_freq - atom_freq`, so a positive-velocity atom sees a **negative** Doppler shift contribution (`-v/lambda`).
- `laser_direction = +1` means a +k photon kick (ground→excited gives +1 recoil); `-1` for the opposite.
- User-facing state-vector functions must preserve representation: functions named as Bordé-frame operations take and return Bordé-frame amplitudes, while lab-frame APIs must take and return lab-frame amplitudes. Only explicit transform helpers should change representation.
- Mixed sequence lists should use `Pulse` for clock pulses and `Clearout` for clearout events. `Clearout.duration` is descriptive metadata only; clearout physics should match `do_clearout(...)` without adding extra evolution.
