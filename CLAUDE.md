# LMT Atom Interferometer Simulation

## IMPORTANT: Do not bypass intentional guards

This codebase contains deliberate error guards — `ValueError`, `NotImplementedError`, assertion failures, and similar checks — that mark **known physics limitations**. These are tripwires, not bugs to fix.

**When you hit one of these errors, stop and report it.** Do not:
- Reimplement the failing function's logic outside the guarded path
- Write a custom loop that skips the check
- Catch and suppress the exception
- Find an alternative route that avoids triggering the guard

The guards exist because the underlying physics is not yet correctly handled. Working around them produces **silently wrong results** — numbers that look plausible but are physically incorrect. A loud failure is always better than a silent wrong answer in a physics simulation.

If you encounter a guard while implementing something, your response should be: explain what guard was hit, what physics limitation it protects, and leave it to the human to decide how to proceed. Do not attempt to make the code "work" by circumventing the check.

Examples of guards to respect (not exhaustive):
- `NotImplementedError: Arm-restricted / simultaneous pulses are a stand-in for shaped pulses ...` in `lmt_sim/lmt_sequence.py` — full quantum propagation of shaped pulses is not implemented
- `ValueError: All states discarded, increase discard_threshold` in `discard_and_renormalise_state_vector` — the renormalisation has nothing left to keep

Note: changing the laser frequency between pulses **is** now supported. The
lab<->Bordé transform (`transform_state_vector`) uses the genuine integral of the
laser detuning, carried on `AtomState` as `(t_ref, detuning_ref_hz,
accumulated_detuning_cycles)` and advanced by
`change_laser_frequency_in_borde_representation` (which records the step without
touching the amplitudes). See `docs/arp_frame_change_finding.md`. The ARP composer
in `lmt_sim/arp.py` is the one piece still paused on this.

## Notebooks are jupytext scripts, not `.ipynb`

This is a working repo, not an archive, so the explorations in `notebooks/` are
plain Python scripts in the [jupytext](https://jupytext.readthedocs.io)
`py:percent` format (`# %%` cell markers), **not** `.ipynb` files. There are no
stored outputs to keep in sync — the script *is* the source of truth.

- Run a script top-to-bottom from the `notebooks/` directory, e.g.
  `cd notebooks && uv run python <name>.py` (the scripts do
  `sys.path.insert(0, "..")`). `tests/test_scripts.py` executes every script
  this way under a headless matplotlib backend.
- Edit the `.py` directly. If you want the interactive cell experience, open it
  with the Jupytext-aware Jupyter (`uv run jupytext --to notebook <name>.py` for
  a throwaway `.ipynb`), but **do not commit `.ipynb` files** — only the scripts.
- A script that must not be run as-is (e.g. it depends on data we don't have)
  should be parked: put a `# --- PARKED: ...` comment explaining why immediately
  after the imports, followed by `raise NotImplementedError(...)`. A parked
  script is a known-broken state. `test_scripts.py` detects the `# --- PARKED`
  marker and marks that script `xfail` (strict), so it surfaces as an expected
  failure rather than a hard red. Removing the marker (un-parking the script)
  automatically holds it to the normal must-run bar again.




Minimal working example of a Large Momentum Transfer (LMT) atom interferometer simulation for Sr-87 on the 698 nm clock transition, using the formalism of Bordé (PhysRevA.30.1836).

## Environment

This is python code, managed by UV. Call "uv sync" at the start of your session and prefix calls with `uv run xxx`.

## Vendored dependencies

This project uses [vendored-refs](https://github.com/charlesbaynham/claude-code-vendored-refs) to manage external repositories.
Dependencies are cloned into `vendor/` on session start based on `.claude/vendored-refs.yaml`.
Do not modify vendored code directly — changes will be lost on next clone.

The tracked repository is:

- `icl_experiments` from `https://gitlab.com/aion-physics/code/artiq/experiment-repositories/icl_experiments.git`
- always cloned from `master`

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
