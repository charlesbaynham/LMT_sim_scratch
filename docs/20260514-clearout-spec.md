# Plan: Atom clearout between pulses

## Context

In real LMT atom-interferometer experiments, a *clearout* (or "blow-away") pulse between interferometer pulses physically removes atoms that are in the ground internal state, e.g. by pushing them away with a near-resonant beam on a non-clock transition. The atoms that remain are those that were measured in the excited state at that moment.

This simulation currently tracks one atom's full coherent wavefunction (an ensemble of branched state rows in Bordé's representation) and reports deterministic final probabilities for ground vs excited. After this change, each run must also be able to be *discarded* mid-sequence, and the final outcome of one simulation run is one of **three** classes: `ground`, `excited`, or `discarded`.

Per the design discussion the user chose **Monte-Carlo projection**: a single clearout call samples a Bernoulli outcome from the current `P(ground) : P(excited)` ratio. On `ground` the run is discarded (returns `None`); on `excited` the ground rows are dropped and the surviving (excited) amplitudes are renormalised so the wavefunction is unit norm again. Discarded runs are short-circuited so no further quantum operations occur. To recover the three-class population the user runs many trials and aggregates.

## Design

### New public function `do_clearout`

Location: `lmt_simulation.py`, near `pulse_interaction_in_borde_representation` and `propagate_states_in_borde_representation`.

```python
def do_clearout(
    m_values, squiggly_amplitudes, internal_is_ground,
    positions, velocities,
    rng=None,
):
    """Projective measurement in the {ground, excited} basis.

    Per-atom Monte Carlo: samples one outcome from the current
    P(ground):P(excited) ratio.

    Returns
    -------
    None if the atom is projected to ground (discarded), else the
    same 5-tuple as the pulse/propagate functions, with ground rows
    removed and excited amplitudes renormalised so the wavefunction
    has unit norm.
    """
```

Implementation notes:
- Reuse `calculate_ground_and_excited_probabilities` (already at `lmt_simulation.py:873`) to compute `p_g, p_e` — this matches the coherent-sum-within-m convention used everywhere else.
- `rng` is an optional `np.random.Generator`; default uses `np.random.default_rng()` (avoid the legacy global state).
- Sample `u = rng.uniform()`; threshold against `p_g / (p_g + p_e)` so the function is robust if amplitudes drift slightly off-norm (e.g. earlier clearouts followed by floating-point pulses).
- On survive: `keep = ~internal_is_ground`; slice each array; multiply amplitudes by `1/sqrt(p_e)`.
- Frame-independence: the per-row phase from `transform_state_vector` is the same for all rows sharing `(m, is_ground)`, so projection in the Bordé frame is identical to projection in the lab frame. No frame transform needed inside `do_clearout`. Document this in the docstring.

### Convention: `None` as the discarded sentinel

A clearout that returns `None` means "this run is over." The user's sequence code becomes:

```python
state = do_clearout(*state, rng=rng)
if state is None:
    return "discarded"
# unpack and continue
m_values, squiggly_amplitudes, internal_is_ground, positions, velocities = state
```

This trivially implements the "no more quantum ops on discarded atoms" optimisation — the discarded branch returns early before any further pulse/propagate call. No per-row "alive" flag is needed and existing pulse/propagate functions need no changes.

### Aggregation helper for the three classes

Add a small helper (kept tight — single responsibility) that runs a user-supplied per-trial closure many times and tallies the three outcome classes:

```python
def run_clearout_trials(sequence_fn, n_trials, rng=None):
    """Run `sequence_fn(rng)` `n_trials` times. The closure must return
    either None (atom discarded mid-sequence) or the final state tuple.

    Returns (p_ground, p_excited, p_discarded) — the population fractions.
    For surviving runs, each contributes its quantum-mechanical p_g, p_e
    weighted by 1/n_trials (so the result equals the deterministic
    population breakdown in the limit n_trials -> infinity)."""
```

This is the only new aggregation surface; everything else stays a primitive.

### Edge cases to handle in the implementation

- `p_g + p_e == 0` (empty state from a prior clearout): treat as already-discarded; return `None`.
- Zero-sized arrays passed in: must not divide by zero. Guard with the same `p_g + p_e == 0` check.
- `p_e` very small but nonzero on survive path: renormalisation amplifies numerical noise; document but no special handling — same regime as any near-zero post-selection.

## Files to modify / add

- `lmt_simulation.py` — add `do_clearout` and `run_clearout_trials`. No changes to existing pulse / propagate / probability functions.
- `test_states_vector.py` — append a new section `# Clearout tests` with the tests below.
- `clearout_sanity_checks.ipynb` (**new** notebook) — sanity plots, see below.

## Tests (append to `test_states_vector.py`)

All tests use a seeded `np.random.default_rng(seed)` so they're deterministic.

1. `test_clearout_pure_ground_always_discards` — start with `c0=1, c1=0`, run `do_clearout` 50 times with different seeds, every call returns `None`.
2. `test_clearout_pure_excited_never_discards` — start with `c0=0, c1=1`, every call returns a non-`None` tuple; the returned amplitudes equal the input excited row (up to renormalisation, which is a no-op here); ground rows are absent.
3. `test_clearout_renormalises_to_unit_norm` — start from a random superposition (loop over a few seeds), on the survive branch assert `calculate_ground_and_excited_probabilities` returns `p_g=0`, `p_e≈1`.
4. `test_clearout_discard_rate_matches_initial_population` — random `c0, c1`, run `do_clearout` 5000 times, assert discard fraction ≈ `|c0|^2` within `4/sqrt(N)`.
5. `test_clearout_drops_ground_rows` — after a π/2 pulse (so we have both ground and excited rows with various `m`), on the survive path verify `len(m_out) == sum(~internal_is_ground_in)` and `not internal_is_ground_out.any()`.
6. `test_clearout_then_pulse_consistent` — after a clearout survive, a subsequent π pulse should produce a state with total population 1 (no NaN, no normalisation drift).
7. `test_clearout_empty_state_returns_none` — feeding zero-length arrays returns `None`.
8. `test_clearout_mc_matches_deterministic_dropground` — Mach-Zehnder with a clearout between the π and final π/2: run 5000 trials with `run_clearout_trials` and compare `(p_g, p_e, p_d)` against the deterministic baseline computed by manually doing `drop ground rows / don't renormalise` and reading off the three population sums. Should match within `4/sqrt(N)`.
9. `test_clearout_frame_independence` — call `do_clearout` (a) on Bordé-frame amplitudes and (b) on lab-frame amplitudes from the same state with the same rng seed; the resulting `(p_g, p_e)` of the survived state and the discard decision should match.

## Sanity notebook: `clearout_sanity_checks.ipynb`

Cells:

1. Imports + constants, reusing `lmt_simulation` (mirrors header of `mach_zehnder_with_temperature.ipynb`).
2. **Plot 1 — discard rate vs initial excited fraction.** Sweep `|c1|^2` from 0 to 1; for each, run 2000 MC trials of `make_atom_states → do_clearout`; plot empirical `P_discarded` with `4/sqrt(N)` error bars against the analytical line `1 - |c1|^2`. Should overlap.
3. **Plot 2 — Mach-Zehnder with clearout, fringe contrast.** Run the standard π/2-π-π/2 MZ phase scan (reuse pattern from `calc_mz_excitation`, `lmt_simulation.py:968`) but insert a `do_clearout` between the π and final π/2 pulses. Plot `P_ground`, `P_excited`, `P_discarded` vs interferometer phase φ on the same axes. Expectations: `P_discarded` ≈ constant ≈ 0.5 (atoms in ground after the π are blown away regardless of φ); the surviving fringe in `P_excited` has reduced absolute amplitude but full contrast relative to the surviving population.
4. **Plot 3 — MC convergence.** Pick one φ from plot 2; sweep `n_trials ∈ {50, 200, 1000, 5000, 20000}`; plot `|MC estimate − deterministic baseline|` vs `n_trials` on log-log; overlay `1/sqrt(N)` reference line.
5. **Plot 4 — clearout commutes with free propagation.** Compare `do_clearout` immediately followed by free propagation vs free propagation followed by `do_clearout` (same rng seed and free-fall duration). Plot both result distributions for a Mach-Zehnder; should overlay within MC noise. Demonstrates the documented frame/time independence of the projection.

## Verification

- `uv run pytest test_states_vector.py -k clearout` — all new tests pass.
- `uv run pytest` — full suite stays green; existing tests untouched.
- Open `clearout_sanity_checks.ipynb`, Run All. Visually confirm each plot matches its expected shape per the descriptions above.
- Spot-check that the existing `calc_mz_excitation` path still returns the same values it did before (no incidental regression in the unchanged code path).
