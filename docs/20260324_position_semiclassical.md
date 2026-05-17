# Plan: Track wavepacket positions through interferometer

## TL;DR

Add classical position tracking to the LMT simulation. Positions are threaded through `pulse_interaction_in_borde_representation` and `propagate_states_in_borde_representation` as an additional array alongside `m_values`, `squiggly_amplitudes`, and `internal_is_ground`. Positions update based on (initial_velocity + m * v_recoil) * duration. For pulses that change m-state, the transitioning branch uses the midpoint approximation (old m for first half, new m for second half of pulse).

## Steps

### Phase 1: Core library changes in `lmt_sim/lmt_simulation.py`

1. **`propagate_states_in_borde_representation`** — Add `positions` parameter and return it
   - New parameter: `positions: np.ndarray` 
   - For each state row, compute velocity as `vz + m_values[idx] * RECOIL_VELOCITY`
   - Update position: `positions_out[idx] = positions[idx] + velocity * time_of_propegation`
   - Return 4-tuple: `(m_values, squiggly_amplitudes_out, state_is_ground, positions_out)`

2. **`pulse_interaction_in_borde_representation`** — Add `positions` parameter and return it, with midpoint approximation for m-changing branches
   - New parameter: `positions: np.ndarray`
   - Allocate `new_positions = np.empty(new_num_rows, ...)` (doubled like other arrays)
   - For the **same-m output branch** (ground output from ground input, or excited output from excited input): position updates as `positions[idx] + (vz + m_same * RECOIL_VELOCITY) * t_pulse`
   - For the **m-changing output branch**: use midpoint approximation — first half at old m velocity, second half at new m velocity: `positions[idx] + (vz + m_old * RECOIL_VELOCITY) * (t_pulse / 2) + (vz + m_new * RECOIL_VELOCITY) * (t_pulse / 2)`
   - Return 4-tuple: `(new_m_values, new_squiggly_amplitudes, new_is_ground, new_positions)`

3. **`make_atom_states`** — Already returns `positions`, no change needed. Callers just need to stop discarding it.

### Phase 2: Update internal callers in `lmt_sim/lmt_simulation.py`

4. **`do_rabi_pulse`** — Thread `positions` through `pulse_interaction_in_borde_representation` (currently doesn't call propagate). Positions not used for result, but API must match.

5. **`calc_mz_excitation`** — Thread `positions` through all pulse and propagation calls. Stop discarding `_positions` from `make_atom_states`.

6. **`__main__` block** — Update the demo code at bottom of file to thread positions through.

### Phase 3: Update notebook callers

7. **`mach_zehnder_with_temperature.ipynb`** — Update `calc_mz_excitation_borde` to:
   - Accept `initial_position_z=0.0` parameter
   - Stop discarding `_positions` from `make_atom_states`
   - Thread `positions` through all pulse/propagation calls (receive and pass 4-tuples)
   - Optionally return final positions alongside excitation fraction

8. **`rabi_flop_with_temperature.ipynb`** — Update single pulse call to use 4-tuple return.

### Phase 4: Update tests

9. **`test_states_vector.py`** — Update all calls to `pulse_interaction_in_borde_representation` and `propagate_states_in_borde_representation` to pass `positions` and unpack 4-tuple returns. Initialize positions from `make_atom_states`. 

## Relevant files

- `2026-03-02 LMT sim/lmt_simulation.py` — Core changes: `propagate_states_in_borde_representation` (line 319), `pulse_interaction_in_borde_representation` (line 365), `do_rabi_pulse` (line 651), `calc_mz_excitation` (line 737), `__main__` demo (line 877). Uses `RECOIL_VELOCITY` constant (already defined).
- `2026-03-02 LMT sim/mach_zehnder_with_temperature.ipynb` — `calc_mz_excitation_borde` function, import list
- `2026-03-02 LMT sim/rabi_flop_with_temperature.ipynb` — Single pulse call
- `2026-03-02 LMT sim/test_states_vector.py` — Test function `test_mz_randomized_population_conserved_every_step`

## Verification

1. Run `pytest test_states_vector.py` — existing population conservation tests must still pass (positions don't affect amplitudes)
2. For a single propagation step: verify position delta equals `(vz + m * RECOIL_VELOCITY) * dt` for a known m and dt
3. For a Mach-Zehnder with `T_FREE > 0`: verify that wavepackets at different m-states separate spatially (different final positions for m=0 vs m=1)
4. For a pulse with no free evolution (`T_FREE = 0`): positions should still change by a small amount proportional to `t_pulse`

## Decisions

- Positions are purely classical tracking — they do NOT affect the quantum amplitudes or phases (consistent with the Bordé plane-wave formalism, where spatial effects enter as a separate overlap integral later)
- The midpoint approximation for m-changing branches during a pulse treats the atom as being at the old m-state velocity for `t_pulse/2`, then the new m-state velocity for `t_pulse/2`
- `transform_state_vector` and `calculate_ground_and_excited_probabilities` are unchanged (they don't need positions)
- `make_atom_states` already returns positions, so its API doesn't change
