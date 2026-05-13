# Spec: Gaussian-beam Rabi frequency (transverse profile)

**Roadmap milestone:** "Implement Gaussian-shaped beams, i.e. varying Rabi frequency over the atom's position in the XY plane" (`docs/roadmap.md`).

## Context

The LMT simulation currently treats every atom as if it sat at the centre of a flat-top laser — `pulse_rabi_freq` is a single scalar applied uniformly to every state row. Real LMT experiments use a TEM00 Gaussian beam, so atoms displaced from the beam axis see a reduced Rabi frequency. This is one of the dominant contrast-loss mechanisms for finite-size atom clouds and needs to be captured before doing the realistic LMT and RAP studies later in the roadmap.

This spec is a deliberate trim of the existing March plan (`docs/20260325_gaussian_beam_plan.md`): out goes the `transverse_temperature` parameter, the `beam_waist=1e6` sentinel, the elliptical-beam and beam-pointing TODOs, and the Rayleigh-range / z-dependence. In stays a clean transverse-only Gaussian, with positions promoted to `(N, 3)` so a later milestone can add the z-dependence without another round of API churn.

## Physics

Transverse Gaussian intensity profile (no z-dependence — beam treated as collimated along z):

$$\Omega(x, y) = \Omega_0 \, \exp\!\left(-\frac{x^2 + y^2}{w^2}\right)$$

where $\Omega_0$ is the on-axis Rabi frequency and $w$ is the beam waist. The beam axis is the z-axis; the beam is centred on the origin. The atom's transverse position is constant for the duration of a pulse (no transverse velocity in this milestone), so the midpoint approximation is exact and the per-row Rabi can be computed once and held constant across the pulse.

## Data model

Promote `positions` from a 1D z-only array to an `(N, 3)` array with columns `[x, y, z]`.

- `x` and `y` are constant per state row (no transverse velocity).
- `z` continues to propagate ballistically via `v_z + m·v_recoil` exactly as today.
- The midpoint approximation for m-changing branches in `pulse_interaction_in_borde_representation` continues to apply to the z component only.

## API changes

### `lmt_simulation.py`

**`make_atom_states`**
```python
def make_atom_states(
    position_x: float = 0.0,
    position_y: float = 0.0,
    position_z: float = 0.0,
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
    return m_values, positions, internal_amplitude, internal_is_ground
```

**`propagate_states_in_borde_representation`**
Signature unchanged in terms of named parameters. Body update: when computing `positions_out[idx]`, only update the z column:
```python
positions_out[idx, :2] = positions[idx, :2]                 # x, y unchanged
positions_out[idx, 2]  = positions[idx, 2] + velocity * t   # z propagates
```

**`pulse_interaction_in_borde_representation`**
- `positions` is now `(N, 3)`.
- `pulse_rabi_freq` accepts a scalar (today's behaviour) or a 1D `(N,)` array of per-row Rabi frequencies. When an array is passed, row `idx` uses `pulse_rabi_freq[idx]`. Implementation: at the top of the function, broadcast to a length-N array (`np.broadcast_to(pulse_rabi_freq, (N,))`), then use `pulse_rabi_freq[idx]` inside the per-row loop.
- Position updates apply only to the z column (x and y carried through unchanged for both same-m and m-changing output branches).

**New helper: `gaussian_rabi`** (pure-physics primitive)
```python
def gaussian_rabi(positions: np.ndarray, on_axis_rabi: float, beam_waist: float) -> np.ndarray:
    """Per-row Rabi frequency from TEM00 transverse intensity profile.

    Ω(x, y) = Ω₀ exp(-(x² + y²) / w²)
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
    pulse_detuning,
    t_pulse,
    on_axis_rabi_freq,
    beam_waist,                      # required, no default
    pulse_phase=0.0,
    k_sign=+1,
    k_wavevector=K_WAVEVECTOR,
    vz=0.0,
):
    rabi_per_row = gaussian_rabi(positions, on_axis_rabi_freq, beam_waist)
    return pulse_interaction_in_borde_representation(
        m_values, squiggly_amplitudes, internal_is_ground, positions,
        pulse_detuning=pulse_detuning,
        t_pulse=t_pulse,
        pulse_rabi_freq=rabi_per_row,
        pulse_phase=pulse_phase,
        k_sign=k_sign,
        k_wavevector=k_wavevector,
        vz=vz,
    )
```

No default for `beam_waist` — if you want flat-top behaviour, call `pulse_interaction_in_borde_representation` directly.

## File-by-file changes

### `lmt_simulation.py`
- `make_atom_states`: signature & body, as above (lines ~50–91).
- `propagate_states_in_borde_representation`: only the position-update line changes (line ~390).
- `pulse_interaction_in_borde_representation`: scalar-or-array `pulse_rabi_freq` handling at the top of the loop; position updates target column 2 only (lines ~395–520).
- Add `gaussian_rabi` and `do_gaussian_pulse` as new top-level functions.
- Update `do_rabi_pulse`, `calc_mz_excitation`, and the `__main__` demo to construct `(N, 3)` positions and pass them through (these are the existing internal callers).

### `test_states_vector.py`
- `test_mz_randomized_population_conserved_every_step` already calls `make_atom_states` and threads positions through — just needs the `(N, 3)` shape (no behavioural change, since x = y = 0 by default). Population conservation must continue to hold.
- Add new tests (see Verification).

### `test_pulse.py`
- This file imports `propagate_states_pulse` and `propagate_states_freely`, both of which no longer exist in `lmt_simulation.py`. It is already broken on the main branch — out of scope for this milestone, leave it alone (or delete in a separate cleanup commit if the user prefers).

### Notebooks
- `rabi_flop_with_temperature.ipynb`, `mach_zehnder_with_temperature.ipynb`, `sliced_rabi_flop_with_temperature.ipynb`, `slice_duration_vs_survival.ipynb`: these call `make_atom_states` and `pulse_interaction_in_borde_representation` directly. They will continue to work as-is because `make_atom_states` still returns positions with defaults of zero — but the positions array shape changes from `(2,)` to `(2, 3)`. Any code that does shape-dependent operations on positions needs a small update.
- Add a follow-up cell in one notebook (suggest `rabi_flop_with_temperature.ipynb`) that demonstrates Rabi flops at varying transverse offsets using `do_gaussian_pulse`, for visual confirmation that the new code path is wired up correctly.

## Verification

1. **Pure-physics primitive**: `gaussian_rabi`
   - `gaussian_rabi(positions with r=0, Ω₀, w) == Ω₀` exactly.
   - `gaussian_rabi(positions with r=w, Ω₀, w) ≈ Ω₀ / e`.
   - `gaussian_rabi(positions with r=5w, Ω₀, w) < 1e-10 · Ω₀`.

2. **Backward compatibility**: `test_mz_randomized_population_conserved_every_step` (parametrised over 100 seeds) continues to pass after the `(N, 3)` change. Positions default to the origin, so `do_gaussian_pulse` is not exercised here; this test guards the scalar-Rabi path.

3. **Array-Rabi path**: a new test calls `pulse_interaction_in_borde_representation` with `pulse_rabi_freq` as an `(N,)` array (all entries equal) and confirms the result matches the scalar-Rabi path bit-for-bit (or to floating-point tolerance).

4. **Gaussian-pulse end-to-end**: a new test runs `do_gaussian_pulse` on a single π-pulse from the ground state with the atom at the beam centre — should give full population transfer (within numerical tolerance). Repeat at `r = w`: pulse area is `Ω₀/e · t_pulse`, expected excitation is `sin²(π/(2e)) ≈ 0.310`.

5. **Cloud-contrast sanity check** (notebook, not pytest): ensemble of atoms at varying r in a Mach-Zehnder sequence shows reduced contrast vs. on-axis. No analytic target, just monotonic decrease with cloud size relative to beam waist.

## Out of scope (future milestones)

- z-dependent beam profile (Rayleigh range, $w(z)$). The `(N, 3)` positions layout is forward-compatible.
- Transverse velocity / cloud expansion in xy.
- Elliptical beams ($w_x \neq w_y$).
- Beam axis offset from origin / beam pointing.
- Continuous Rabi variation during a pulse (needed only if `v_x · t_pulse` becomes comparable to $w$, which is far from current parameters).
- Coupling to a per-atom transverse temperature for initial position/velocity sampling — this belongs in a separate "transverse thermal effects" milestone if/when needed.

## Critical files

- `lmt_simulation.py` — all core changes
- `test_states_vector.py` — backward-compat sanity + new tests
- `docs/roadmap.md` — tick the first item once delivered
