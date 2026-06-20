# Plan: Gaussian Beam Profile (TEM00) for LMT Simulation

## TL;DR

Add Gaussian beam shape (TEM00 transverse mode) to the LMT simulation. The beam intensity varies with transverse position, affecting the Rabi frequency. Use a midpoint approximation where the atom's position at pulse center determines the Rabi frequency for the entire pulse.

**Backward compatibility:** Set `beam_waist` to a very large value (e.g., 1e6 m) to approximate a flat-top beam (uniform intensity).

## Physics Background

### Gaussian Beam (TEM00) Profile

For a Gaussian beam propagating along the z-axis, the intensity at transverse position $(x, y)$ is:

$$I(x, y, z) = I_0 \left(\frac{w_0}{w(z)}\right)^2 \exp\left(-\frac{2(x^2 + y^2)}{w(z)^2}\right)$$

where:
- $I_0$ is the peak intensity at the beam waist
- $w_0$ is the beam waist radius (at $z=0$)
- $w(z) = w_0 \sqrt{1 + (z/z_R)^2}$ is the beam radius at position $z$
- $z_R = \pi w_0^2 / \lambda$ is the Rayleigh range

The Rabi frequency scales with the square root of intensity:

$$\Omega(x, y, z) = \Omega_0 \frac{w_0}{w(z)} \exp\left(-\frac{x^2 + y^2}{w(z)^2}\right)$$

where $\Omega_0$ is the on-axis Rabi frequency at the waist.

### Approximation Strategy

**Current approach (constant Rabi):** The pulse uses a constant Rabi frequency throughout.

**New approach (Gaussian beam):**
1. Track 3D position $(x, y, z)$ of each state branch
2. At pulse center, compute the atom's transverse position $(x, y)$
3. Compute Rabi frequency from Gaussian beam formula
4. Use this Rabi frequency as constant throughout the pulse

**TODO:** Consider implementing continuous variation where the Rabi frequency is updated as the atom moves through the beam during the pulse.

## Implementation Plan

### Phase 1: Core Infrastructure Changes

#### 1.1 Extend State Tracking to 3D

**File:** `lmt_sim/lmt_simulation.py`

Currently positions are 1D (z-only). Extend to track $(x, y, z)$ for each state row.

- Modify `make_atom_states()` to accept 3D positions and velocities
- Return `positions` as shape `(N, 3)` array with columns `[x, y, z]`
- Return `velocities` as shape `(N, 3)` array with columns `[vx, vy, vz]`
- Add `transverse_temperature` parameter (default 1e-20 K, effectively zero)
- Update all callers to handle 3D positions and velocities

**Decision 1: Position Array Format**

**Decision:** Use `(N, 3)` array for all 3D positions and velocities
- `positions` array has shape `(N, 3)` with columns `[x, y, z]`
- Same format for velocities: `(N, 3)` with columns `[vx, vy, vz]`
- Helper function to extract individual dimensions when needed
- Update all functions that accept positions to use this format

**Rationale:** More compact, consistent with NumPy conventions, easier to pass around

#### 1.2 Add Gaussian Beam Parameters

Add constants/configuration for:
- `BEAM_WAIST_X`: Beam waist radius in x-direction (m)
- `BEAM_WAIST_Y`: Beam waist radius in y-direction (m) — for circular beams, equal to X
- `ON_AXIS_RABI_FREQ`: Rabi frequency at beam center
- `BEAM_WAVELENGTH`: Laser wavelength (may differ from atomic transition)

**Note:** For initial implementation, assume circular Gaussian ($w_x = w_y = w_0$).

#### 1.3 Implement Gaussian Beam Function

```python
def calculate_gaussian_rabi_frequency(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_z: np.ndarray,
    on_axis_rabi: float,
    waist_radius: float,
    wavelength: float,
) -> np.ndarray:
    """
    Calculate Rabi frequency for each state branch based on Gaussian beam profile.

    For backward compatibility, set waist_radius to a very large value (e.g., 1e6 m)
    to approximate a flat-top beam (uniform intensity).

    Returns array of Rabi frequencies (Hz) for each branch.
    """
```

Compute:
1. Rayleigh range: $z_R = \pi w_0^2 / \lambda$
2. Beam radius at z: $w(z) = w_0 \sqrt{1 + (z/z_R)^2}$
3. Normalized intensity: $(w_0/w(z))^2 \exp(-2(x^2+y^2)/w(z)^2)$
4. Rabi frequency: $\Omega_0 (w_0/w(z)) \exp(-(x^2+y^2)/w(z)^2)$

**Backward compatibility:** When `waist_radius` is very large (e.g., 1e6 m), the exponential term approaches 1 and $w(z) \approx w_0$, giving uniform Rabi frequency $\Omega_0$ everywhere.

### Phase 2: Update Pulse Functions

#### 2.1 Modify `pulse_interaction_in_borde_representation()`

**Current signature:**
```python
def pulse_interaction_in_borde_representation(
    m_values,
    squiggly_amplitudes,
    internal_is_ground,
    positions,  # Currently 1D: z-position
    pulse_detuning,
    t_pulse,
    pulse_rabi_freq,  # Single value
    ...
):
```

**New signature:**
```python
def pulse_interaction_in_borde_representation(
    m_values,
    squiggly_amplitudes,
    internal_is_ground,
    positions,  # Now (N, 3) array: [x, y, z]
    pulse_detuning,
    t_pulse,
    pulse_rabi_freq,  # Base Rabi frequency (on-axis at waist)
    beam_waist=1e6,  # Default 1e6 m = flat-top beam (backward compatible)
    beam_wavelength=None,  # If None, use TRANSITION_WAVELENGTH
    ...
):
```

**Behavior:**
1. If `beam_waist` is very large (e.g., >= 1e6 m):
   - Use constant `pulse_rabi_freq` for all branches (backward compatible)
2. If `beam_waist` is smaller (reasonable value, e.g., < 1e3 m):
   - Extract $(x, y, z)$ at pulse center (midpoint approximation)
   - Compute per-branch Rabi frequency from Gaussian formula
   - Use these Rabi frequencies in the pulse interaction

**Note:** No Boolean flag needed — the waist value itself controls the behavior.

#### 2.2 Update Position Tracking During Pulses

Currently positions are updated classically during pulses with midpoint approximation for velocity. Keep this, but now track 3D position evolution.

For transverse motion (x, y):
- If atoms have initial transverse velocity, update $x$ and $y$ positions
- For most LMT simulations, transverse velocity may be zero (1D model)
- But we should support it for completeness

**Position update during pulse:**
- $x_{new} = x + v_x \cdot t_{pulse}$
- $y_{new} = y + v_y \cdot t_{pulse}$
- $z_{new} = z + v_z \cdot t_{pulse}$ (already implemented)

### Phase 3: Update Propagation Functions

#### 3.1 Modify `propagate_states_in_borde_representation()`

Extend to handle 3D position propagation:
- Accept `velocities` parameter (or compute from momentum)
- Update x, y, z positions during free evolution
- For now, assume ballistic motion (no external forces)

**Note:** This enables tracking of transverse spreading of the atomic cloud.

### Phase 4: Update High-Level Functions

#### 4.1 Update `do_rabi_pulse()`

Add parameters:
- `initial_position_x=0.0`
- `initial_position_y=0.0`
- `initial_velocity_x=0.0`
- `initial_velocity_y=0.0`
- `transverse_temperature=1e-20`  # Effectively zero (K), backwards compatible
- `beam_waist=1e6`  # Default 1e6 m for flat-top beam (backward compatible)

#### 4.2 Update `calc_mz_excitation()`

Same additions as above. Thread 3D positions through the pulse sequence.

### Phase 5: Update Notebooks

#### 5.1 `mach_zehnder_with_temperature.ipynb`

- Add initial transverse position sampling (e.g., from Gaussian distribution)
- Add `transverse_temperature` parameter (default 1e-20 K for backward compatibility)
- Add `beam_waist` parameter (default 1e6 m for backward compatible flat-top beam)
- Plot showing how transverse position affects interference contrast

#### 5.2 `rabi_flop_with_temperature.ipynb`

- Add transverse position distribution
- Compare Rabi flops for on-axis vs off-axis atoms
- Show ensemble average with Gaussian beam effect

### Phase 6: Update Tests

#### 6.1 New Tests

- Test Gaussian beam Rabi frequency calculation
- Test that on-axis atoms get expected Rabi frequency
- Test that far-off-axis atoms get reduced Rabi frequency
- Test 3D position propagation
- Test backward compatibility (very large waist ≈ uniform Rabi)

#### 6.2 Update Existing Tests

- Update `test_states_vector.py` to use 3D positions
- Ensure backward compatibility (tests pass with `beam_waist=1e6`)

## Relevant Files

| File | Changes |
|------|---------|
| `lmt_sim/lmt_simulation.py` | Core changes: 3D positions, Gaussian beam function, updated pulse/propagation |
| `test_states_vector.py` | Update to 3D positions, add Gaussian beam tests |
| `mach_zehnder_with_temperature.ipynb` | Add transverse position, Gaussian beam option |
| `rabi_flop_with_temperature.ipynb` | Add transverse position, Gaussian beam option |

## Key Decisions

### Decision 1: Position Array Format
- **Option A:** `(N, 3)` array — More compact, consistent
- **Option B:** Three separate 1D arrays — Easier migration from current code
- **Recommendation:** Option A with helper functions

### Decision 2: Backward Compatibility Mechanism

**Decision:** Use `beam_waist` parameter with default value of `1e6` meters
- Large waist (1e6 m) approximates flat-top beam (uniform intensity)
- No Boolean flag needed — the waist value itself controls behavior
- Beam waist is always centered on origin r = (0, 0, 0)
- Add as function parameter with default, not just global constant

**Rationale:** Cleaner API, physical intuition (large waist = collimated beam), avoids extra flags

### Decision 3: Beam Waist Specification

**Decision:** Function parameter with global constant default
- Global constant `DEFAULT_BEAM_WAIST = 1e6`  (meters)
- Function parameter `beam_waist` defaults to `DEFAULT_BEAM_WAIST`
- Users can override per-function call or change global default
- Beam always centered on origin r = (0, 0, 0)

**Rationale:** Flexibility where needed, backward compatible defaults, clean API

### Decision 4: Transverse Velocity

**Decision:** Include transverse velocity with independent temperature control
- Add `transverse_temperature` parameter separate from vertical temperature
- Default to `1e-20` K (effectively zero, atoms stationary in transverse directions)
- For backwards compatibility, keep transverse motion frozen by default
- When needed, user can set to same value as vertical temperature or different

**Rationale:** Supports full 3D physics when needed, backwards compatible (no transverse motion by default), flexible for different simulation scenarios

## Verification Plan

1. **Unit test:** Gaussian Rabi frequency formula matches analytical expectation
2. **Consistency:** On-axis atom gets full Rabi frequency
3. **Consistency:** Far off-axis atom gets near-zero Rabi frequency
4. **Backward compatibility:** All existing tests pass with `beam_waist=1e6` (default value)
5. **Integration:** Notebook runs successfully with Gaussian beam enabled (reasonable waist)

## TODO Items

1. [ ] Consider implementing continuous Rabi frequency variation during pulse
2. [ ] Add support for elliptical beams ($w_x \neq w_y$)
3. [ ] Consider beam pointing/alignment effects (beam axis offset from z-axis)
4. [ ] Add visualization of beam profile in notebooks

## Implementation Notes

**Backward compatibility detail:**
- Setting `beam_waist = 1e6` (1 megameter) makes the beam essentially flat over any practical lab scale
- The Rayleigh range becomes enormous, so $w(z) \approx w_0$ everywhere
- The Gaussian exponential $\exp(-(x^2+y^2)/w^2) \approx 1$ for any realistic $(x,y)$
- Result: Uniform Rabi frequency = `pulse_rabi_freq` everywhere

**Midpoint approximation detail:**
- For pulse of duration $t_p$, position at center is $r(t_p/2)$
- Use this position to compute Rabi frequency
- Apply this Rabi frequency as constant for entire pulse
- After pulse, update position to $r(t_p)$ using velocity

**Performance considerations:**
- Computing Gaussian Rabi for every branch adds some overhead
- Consider caching if same positions are used repeatedly
- For now, compute fresh each pulse (cleanest)

---

*Plan generated: 2026-03-25*
*Updated: 2026-03-25 (four decisions made: (N,3) format, beam_waist=1e6 default, centered at origin, independent transverse_temperature)*
