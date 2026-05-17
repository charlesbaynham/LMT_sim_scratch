# Spec: Gaussian-beam Rabi frequency (transverse profile)

**Roadmap milestone:** "Implement Gaussian-shaped beams, i.e. varying Rabi frequency over the atom's position in the XY plane" (`docs/roadmap.md`).

## Context

The LMT simulation currently treats every atom as if it sat at the centre of a flat-top laser -- `pulse_rabi_freq` is a single scalar applied uniformly to every state row. Real LMT experiments use a TEM00 Gaussian beam, so atoms displaced from the beam axis see a reduced Rabi frequency. This is one of the dominant contrast-loss mechanisms for finite-size atom clouds and needs to be captured before doing the realistic LMT and RAP studies later in the roadmap.

This spec is a deliberate trim of the existing March plan (`docs/20260325_gaussian_beam_plan.md`): out goes the `transverse_temperature` parameter, the `beam_waist=1e6` sentinel, the elliptical-beam and beam-pointing TODOs, and the Rayleigh-range / z-dependence. In stays a clean transverse-only Gaussian, with positions promoted to `(N, 3)` so a later milestone can add the z-dependence without another round of API churn.

## Key addition: 3D velocity tracking

**This spec now includes 3D velocity tracking.** Velocities are tracked as an `(N, 3)` array alongside positions. They are constant in time (no forces, ballistic motion). The only velocity change is vz getting recoil kicks during pulses (existing behaviour). vx and vy never change.

x and y positions propagate ballistically using these velocities: `r(t) = r_0 + v * t`. This applies during both pulses and free evolution. The midpoint approximation for pulse centre now uses velocity: `r_center = r + v * t_pulse/2`.

**Future direction:** Velocities will be initialized from a Maxwell-Boltzmann distribution at cloud temperature. That is NOT in this milestone -- we are adding the tracking infrastructure now so the temperature initialization can be added later without API churn.

## Physics

Transverse Gaussian intensity profile (no z-dependence -- beam treated as collimated along z):

```
Omega(x, y) = Omega_0 * exp(-(x^2 + y^2) / w^2)
```

where Omega_0 is the on-axis Rabi frequency and w is the beam waist. The beam axis is the z-axis; the beam is centred on the origin. The atom's transverse position evolves ballistically during the pulse (via vx, vy), so the per-row Rabi is computed at the pulse centre using the midpoint approximation.

## Data model

Promote `positions` from a 1D z-only array to an `(N, 3)` array with columns `[x, y, z]`.

Add `velocities` as an `(N, 3)` array with columns `[vx, vy, vz]`.

- `x` and `y` propagate ballistically: `x(t) = x_0 + vx * t`, `y(t) = y_0 + vy * t`
- `z` continues to propagate ballistically via `v_z + m * v_recoil` exactly as today
- `vx` and `vy` are constant (never change)
- `vz` changes only via recoil kicks during pulses (existing behaviour)
- The midpoint approximation for m-changing branches in `pulse_interaction_in_borde_representation` now applies to all three dimensions using velocities

## API changes

### `lmt_sim/lmt_simulation.py`

**`make_atom_states`**
```python
def make_atom_states(
    position_x: float = 0.0,
    position_y: float = 0.0,
    position_z: float = 0.0,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
    initial_velocity_z: float = 0.0,
    c0: complex = 1,
    c1: complex = 0,
):
    ...
    positions = np.array(
        [[position_x, position_y, position_z],
         [position_x, position_y, position_z]],
        dtype=np.float64,
    )  # shape (2, 3)
    velocities = np.array(
        [[velocity_x, velocity_y, initial_velocity_z],
         [velocity_x, velocity_y, initial_velocity_z]],
        dtype=np.float64,
    )  # shape (2, 3)
    return m_values, positions, velocities, internal_amplitude, internal_is_ground
```

**`propagate_states_in_borde_representation`**
Signature: add `velocities` parameter. Body update: propagate all three dimensions ballistically:
```python
positions_out[idx] = positions[idx] + velocities[idx] * t
```
Velocities remain unchanged (returned as-is).

**`pulse_interaction_in_borde_representation`**
- `positions` is now `(N, 3)`.
- `velocities` is now `(N, 3)`.
- `pulse_rabi_freq` accepts a scalar (today's behaviour) or a 1D `(N,)` array of per-row Rabi frequencies. When an array is passed, row `idx` uses `pulse_rabi_freq[idx]`. Implementation: at the top of the function, broadcast to a length-N array (`np.broadcast_to(pulse_rabi_freq, (N,))`), then use `pulse_rabi_freq[idx]` inside the per-row loop.
- Position updates apply to all three dimensions using ballistic propagation with velocities. For same-m branches: `positions_out = positions + velocities * t_pulse`. For m-changing branches, use midpoint approximation with velocities for all dimensions.
- Velocity updates: vx and vy unchanged. vz gets +/- recoil (existing behaviour).

**New helper: `gaussian_rabi`** (pure-physics primitive)
```python
def gaussian_rabi(positions: np.ndarray, on_axis_rabi: float, beam_waist: float) -> np.ndarray:
    """Per-row Rabi frequency from TEM00 transverse intensity profile.

    Omega(x, y) = Omega_0 * exp(-(x^2 + y^2) / w^2)
    """
    r2 = positions[:, 0] ** 2 + positions[:, 1] ** 2
    return on_axis_rabi * np.exp(-r2 / beam_waist ** 2)
```

**New helper: `do_gaussian_pulse`** (the wrap-with-helper user-facing entry point)
```python
def do_gaussian_pulse(
    m_values,
    squiggly_amplitudes,
    internal_is_ground,
    positions,                       # (N, 3)
    velocities,                      # (N, 3)
    pulse_detuning,
    t_pulse,
    on_axis_rabi_freq,
    beam_waist,                      # required, no default
    pulse_phase=0.0,
    k_sign=+1,
    k_wavevector=K_WAVEVECTOR,
    vz=0.0,
):
    # Compute position at pulse centre for Gaussian Rabi calculation
    positions_mid = positions + velocities * t_pulse / 2
    rabi_per_row = gaussian_rabi(positions_mid, on_axis_rabi_freq, beam_waist)
    return pulse_interaction_in_borde_representation(
        m_values, squiggly_amplitudes, internal_is_ground, positions, velocities,
        pulse_detuning=pulse_detuning,
        t_pulse=t_pulse,
        pulse_rabi_freq=rabi_per_row,
        pulse_phase=pulse_phase,
        k_sign=k_sign,
        k_wavevector=k_wavevector,
        vz=vz,
    )
```

No default for `beam_waist` -- if you want flat-top behaviour, call `pulse_interaction_in_borde_representation` directly.

## File-by-file changes

### `lmt_sim/lmt_simulation.py`
- `make_atom_states`: signature & body, as above (lines ~50-91).
- `propagate_states_in_borde_representation`: add velocities parameter, ballistic propagation for all 3 dimensions (line ~390).
- `pulse_interaction_in_borde_representation`: add velocities parameter; scalar-or-array `pulse_rabi_freq` handling at the top of the loop; position updates use ballistic propagation for all 3 dimensions; velocity updates (vx, vy unchanged, vz recoil) (lines ~395-520).
- Add `gaussian_rabi` and `do_gaussian_pulse` as new top-level functions.
- Update `do_rabi_pulse`, `calc_mz_excitation`, and the `__main__` demo to construct `(N, 3)` positions and velocities and pass them through (these are the existing internal callers).

### `test_states_vector.py`
- `test_mz_randomized_population_conserved_every_step` already calls `make_atom_states` and threads positions through -- just needs the `(N, 3)` shape and velocities (no behavioural change, since x = y = 0 and vx = vy = 0 by default). Population conservation must continue to hold.
- Add new tests (see Verification).

### `test_pulse.py`
- This file imports `propagate_states_pulse` and `propagate_states_freely`, both of which no longer exist in `lmt_sim/lmt_simulation.py`. It is already broken on the main branch -- out of scope for this milestone, leave it alone (or delete in a separate cleanup commit if the user prefers).

### Notebooks
- `rabi_flop_with_temperature.ipynb`, `mach_zehnder_with_temperature.ipynb`, `sliced_rabi_flop_with_temperature.ipynb`, `slice_duration_vs_survival.ipynb`: these call `make_atom_states` and `pulse_interaction_in_borde_representation` directly. They will continue to work as-is because `make_atom_states` still returns positions with defaults of zero -- but the positions array shape changes from `(2,)` to `(2, 3)` and velocities `(2, 3)` are now returned. Any code that does shape-dependent operations on positions needs a small update.
- Add a follow-up cell in one notebook (suggest `rabi_flop_with_temperature.ipynb`) that demonstrates Rabi flops at varying transverse offsets using `do_gaussian_pulse`, for visual confirmation that the new code path is wired up correctly.

## Verification

1. **Pure-physics primitive**: `gaussian_rabi`
   - `gaussian_rabi(positions with r=0, Omega_0, w) == Omega_0` exactly.
   - `gaussian_rabi(positions with r=w, Omega_0, w) ~ Omega_0 / e`.
   - `gaussian_rabi(positions with r=5w, Omega_0, w) < 1e-10 * Omega_0`.

2. **Backward compatibility**: `test_mz_randomized_population_conserved_every_step` (parametrised over 100 seeds) continues to pass after the `(N, 3)` change. Positions default to the origin, velocities default to zero, so `do_gaussian_pulse` is not exercised here; this test guards the scalar-Rabi path.

3. **Array-Rabi path**: a new test calls `pulse_interaction_in_borde_representation` with `pulse_rabi_freq` as an `(N,)` array (all entries equal) and confirms the result matches the scalar-Rabi path bit-for-bit (or to floating-point tolerance).

4. **Gaussian-pulse end-to-end**: a new test runs `do_gaussian_pulse` on a single pi-pulse from the ground state with the atom at the beam centre -- should give full population transfer (within numerical tolerance). Repeat at `r = w`: pulse area is `Omega_0/e * t_pulse`, expected excitation is `sin^2(pi/(2e)) ~ 0.310`.

5. **Ballistic propagation**: a new test verifies that `propagate_states_in_borde_representation` correctly updates x, y, z positions using velocities: `position = initial + velocity * time`.

6. **Velocity tracking**: a new test verifies that vx and vy remain constant through pulses and propagation, and that vz only changes via recoil during pulses.

7. **Cloud-contrast sanity check** (notebook, not pytest): ensemble of atoms at varying r in a Mach-Zehnder sequence shows reduced contrast vs. on-axis. No analytic target, just monotonic decrease with cloud size relative to beam waist.

8. **ASCII check**: all code uses ASCII characters only (no Unicode math symbols, no Greek letters in variable names).

## Out of scope (future milestones)

- z-dependent beam profile (Rayleigh range, w(z)). The `(N, 3)` positions layout is forward-compatible.
- Transverse velocity / cloud expansion in xy (velocities are tracked but initialized to zero in this milestone).
- Elliptical beams (w_x != w_y).
- Beam axis offset from origin / beam pointing.
- Continuous Rabi variation during a pulse (needed only if `vx * t_pulse` becomes comparable to w, which is far from current parameters).
- Coupling to a per-atom transverse temperature for initial position/velocity sampling -- this belongs in a separate "transverse thermal effects" milestone if/when needed.
- Maxwell-Boltzmann velocity distribution initialization -- tracked as future work. The velocity infrastructure is in place; temperature initialization is the next change.

## ASCII-only requirement

- Use only ASCII characters in all code
- No Unicode math symbols, no Greek letters in variable names
- Use `omega` not Omega, `lambda_` not lambda, etc.
- Write equations in plain ASCII in comments and docstrings

## Critical files

- `lmt_sim/lmt_simulation.py` -- all core changes
- `test_states_vector.py` -- backward-compat sanity + new tests
- `docs/roadmap.md` -- tick the first item once delivered
