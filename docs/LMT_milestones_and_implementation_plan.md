# LMT Interferometry Simulation: Milestones and Implementation Plan

**Document Date:** 2026-03-07
**Based on:** [[2026-03-06 LMT Monte Carlo]] notes
**Goal:** Reach accurate 10 LMT simulation with all major noise sources

---

## Overview

The simulation now correctly implements MT (Multi-pulse Transition) using Borde's formalism with unitary transforms decomposing the problem into 2×2 time-independent matrix equations. The current implementation tracks momentum eigenstates |ℏkm⟩ for ground (a) and excited (b) states with semi-classical position tracking.

---

## Priority 1: Validation and Core Functionality (Complete First)

### 1.1 Multi-LMT Validation
**Status:** Not started
**Purpose:** Verify simulation behavior with increasing LMT order
**Tasks:**
- Test with N=2, N=5, N=10 LMTs
- Verify phase accumulation scales correctly with momentum transfer
- Check that recoil shift effects compound properly
- Validate interferometer closure (output ports sum to input population)

**Success Criteria:**
- Population conservation within numerical precision (< 0.1%)
- Phase scales linearly with N (ℏk_eff = Nℏk)
- Recoil shifts accumulate correctly

### 1.2 Computational Baseline
**Status:** In progress
**Purpose:** Establish performance baseline before adding complexity
**Tasks:**
- Profile current code for N=10, N=50, N=100
- Measure execution time vs N scaling
- Document memory usage patterns

**Success Criteria:**
- Documented baseline for comparison
- Identify scaling bottlenecks

---

## Priority 2: Thermal Effects (High Impact, Required for Realism)

### 2.1 Classical Mixed States for Temperature Dephasing
**Status:** Not started
**Purpose:** Model thermal distribution of atoms
**Tasks:**
- Implement Maxwell-Boltzmann velocity distribution sampling
- Run ensemble of trajectories with different initial velocities
- Calculate ensemble-averaged interference contrast
- Compare with analytical predictions for known limits

**Physics:**
- Sample velocities from P(v) ∝ v² exp(-mv²/2kT)
- Track thermal dephasing timescale
- Extract temperature-dependent contrast reduction

**Success Criteria:**
- Contrast vs temperature curve matches theory
- Dephasing time consistent with Doppler width

---

## Priority 3: Spatial Effects (Critical for Large N)

### 3.1 Wavepacket Spatial Overlap
**Status:** Planned
**Purpose:** Model finite spatial extent and cloud separation
**Implementation Plan:**
1. Start with Gaussian wavepacket defined by trap harmonic oscillator ground state
2. Convert to momentum spread ℏ/(2σ_x)
3. Propagate position spread during free evolution
4. Multiply phase by spatial overlap integral

**Physics:**
- Initial size: σ_x = √(ℏ/2mω_trap)
- Expansion: σ(t) = σ_x √(1 + (ℏt/2mσ_x²)²)
- Overlap: O = exp(-Δx²/4σ²) where Δx is arm separation

**Success Criteria:**
- Overlap factor correctly reduces contrast at large N
- Smooth transition from coherent to incoherent regime

### 3.2 Spatially-Varying Rabi Frequency
**Status:** Code infrastructure exists
**Purpose:** Model atom movement through Gaussian beam profile
**Tasks:**
- Use existing 3D position tracking
- Calculate Rabi frequency Ω(r) = Ω₀ exp(-r²/w²)
- Integrate over pulse duration with position evolution

**Success Criteria:**
- Rabi frequency correctly modulated by transverse position
- Edge atoms experience reduced pulse area

---

## Priority 4: Laser Noise (Medium Priority)

### 4.1 Velocity Slicing
**Status:** Not started
**Purpose:** Model finite momentum resolution from laser linewidth
**Tasks:**
- Implement velocity-dependent Rabi frequency
- Add Doppler shift compensation
- Model Raman/pulse pair detuning effects

### 4.2 Laser Frequency and Phase Noise
**Status:** Not started
**Purpose:** Model technical noise sources
**Tasks:**
- Add random phase fluctuations to pulse sequence
- Implement frequency noise power spectral density
- Correlate noise between pulses (common-mode rejection)

**Success Criteria:**
- Phase noise produces contrast reduction scaling with N²
- Common-mode noise partially suppressed

---

## Priority 5: Computational Optimizations (Required for N>10)

### 5.1 Eigenstate Consolidation
**Status:** Planned (reluctantly)
**Purpose:** Reduce 2^N scaling to linear N scaling
**Approach:**
- Combine all momentum eigenstates after each pulse
- Track as single state with accumulated phase
- Sacrifices: spatial overlap accuracy, wavefront aberration handling

**Trade-offs:**
- ✅ Enables N=100+ simulations
- ⚠️ Cannot properly handle spatial overlap (coherent vs incoherent parasitics)
- ⚠️ Precludes wavefront aberration modeling (k-vector changes)

**Decision Point:** Implement if N=10 baseline is too slow

### 5.2 State Trimming
**Status:** Idea stage
**Purpose:** Reduce state space by removing negligible components
**Tasks:**
- Track population in each eigenstate
- Prune states with occupancy < threshold (e.g., 10⁻⁶)
- Track total pruned population as error budget
- Only keep states that could overlap with main components

**Success Criteria:**
- < 5% population loss to pruning
- > 10× speedup for N=50+

---

## Priority 6: Advanced Physics (Future Work)

### 6.1 Clear-Out Pulses
**Status:** Considered
**Purpose:** Remove unwanted momentum states
**Benefit:** Simplifies computation, improves contrast
**Implementation:** Additional π pulses to eject parasitic paths

### 6.2 Finite Pulse Duration and Pulse Shaping
**Status:** Framework exists
**Current:** Transition treated as instantaneous at pulse start
**Better:** Transition at pulse center
**Best:** Full time evolution during pulse (already in formalism)

**Additional:**
- Moving atoms through beams (Ω and k vary during pulse)
- Solution: Segment pulse into time bins
- Treat each bin as static, small enough for constant dynamics

**⚠️ Composite / ARP pulses make this acute (TODO — knowingly deferred):**
`CompositePulse` (see `lmt_sim/lmt_sequence.py`, `lmt_sim/lmt_simulation.py`
`composite_pulse_interaction_in_borde_representation`) applies a whole ARP
frequency sweep as a SINGLE branching event. Such pulses can be **hundreds of µs
long**, over which the atom moves appreciably — yet the current implementation
evaluates the atom's position (and hence the beam profile / Rabi) **once** for
the whole pulse and imparts the discrete recoil at a single instant
(`momentum_kick_fraction`, default the midpoint). This is a poor approximation
for long pulses and we are shipping it knowingly. The proper fix:
- re-evaluate the atom's position **per sub-slice**, not once per pulse, and
- evaluate that position at each slice's **midpoint** (not its start or end),
- which also lets the Gaussian-beam Rabi vary slice-by-slice (ties into §3.2).
The code carries matching `TODO`s at `_branch_row_with_propagator` and the
trajectory flip step in `compute_spacetime_trajectory`.

### 6.3 Wavefront Aberrations
**Status:** Deferred (breaks consolidation simplification)
**Purpose:** Model imperfect optics
**Challenge:** k-vector changes per pulse → no well-resolved |ℏkm⟩ basis
**Possible Approach:**
- Convert to local momentum basis during each pulse
- Transform back to global basis after
- Requires careful handling of interference

---

## Implementation Timeline

| Phase | Milestones | Target Completion |
|-------|-----------|-------------------|
| **Phase 1** | 1.1, 1.2, 2.1 | 2 weeks |
| **Phase 2** | 3.1, 3.2 | 2 weeks |
| **Phase 3** | 4.1, 4.2 | 1 week |
| **Phase 4** | 5.1 (if needed), 5.2 | 1 week |
| **Phase 5** | 6.1, 6.2 | Future |

**Total to 10 LMT with noise:** 4-6 weeks
**Stretch goal (100 LMT):** Requires Phase 4 optimization

---

## Key Technical Decisions

### Decision 1: Consolidate States?
- **Now:** No - keep full 2^N for accuracy
- **Trigger:** If N=10 takes > 10 minutes
- **Alternative:** Implement state trimming first

### Decision 2: Spatial Overlap Priority
- **Critical for:** Large N where arm separation exceeds wavepacket size
- **Current plan:** Implement before consolidating states
- **Risk:** May force keeping full 2^N scaling

### Decision 3: Wavefront Aberrations
- **Current:** Deferred
- **Reason:** Probably not limiting factor yet
- **Revisit:** After baseline noise sources characterized

---

## Current Sanity Checks Passing

✅ Single π pulse shows recoil shift offset
✅ MZ interferometer without propagation shows expected behavior
✅ Peak excitation offset correctly modeled

---

## Next Immediate Actions

1. **Run N=2,5,10 LMT validation** - verify scaling
2. **Profile performance** - establish baseline
3. **Implement thermal ensemble** - most impactful noise source
4. **Decide on consolidation** based on performance data

---

*Document generated from Charles' research notes on 2026-03-07*
