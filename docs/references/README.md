# Reference papers

Source papers for the Bordé two-level / recoil formalism this simulation
implements. (`*.pdf` is ignored repo-wide; `.gitignore` has a scoped negation so
papers in this directory are tracked.)

## Primary

**Borde_1984_PhysRevA.30.1836_optical_ramsey_traveling_waves.pdf**
Ch. J. Bordé, Ch. Salomon, S. Avrillier, A. Van Lerberghe, Ch. Bréant, D. Bassi
& G. Scoles, *"Optical Ramsey fringes with traveling waves,"*
Phys. Rev. A **30**, 1836 (1984).

This is the paper the equation references throughout `lmt_sim` point at:

- **Eq. 7** — recoil-shifted detuning `Ω₃` → `_borde_frame_constants`
- **Eq. 8** — common phase `Ω₀` → `_borde_frame_constants`
- **Eq. 12** — generalized Rabi frequency `Ω² = Ω₃² + 4Ω_ba²`
- **Eq. 13** — the 2×2 transition matrix `A,B,C,D` → `_calculate_interaction_constants`
- **Eqs. 4, 5/14** — lab↔Bordé frame transform and free-evolution phase

Sign convention: Bordé writes the generator with a `+i`
(Eq. 6/9: `∂_t ψ = +(i/2)(Ω·σ + Ω₀)ψ`), so the effective Hamiltonian in
`ψ = exp(-iHt)ψ₀` is `H = -½ Ω·σ` — i.e. `H/ℏ = -½Ω₃σ_z - ω_ab(cosφ σ_x + sinφ σ_y)`,
both diagonal and coupling carrying the same minus sign. The code follows this
(it reproduces Eq. 13 verbatim); see the note under `do_pulse` in `CLAUDE.md`.

## Corroborating (modern review)

**Cronin_Schmiedmayer_Pritchard_2009_RevModPhys.81.1051_optics_interferometry_atoms_molecules.pdf**
A. D. Cronin, J. Schmiedmayer & D. E. Pritchard, *"Optics and interferometry with
atoms and molecules,"* Rev. Mod. Phys. **81**, 1051 (2009); open access as
arXiv:0712.3703.

Corroborates the formalism the code uses: the detuning convention
`Δ = ω_laser − ω_atom`, the generalized Rabi frequency `Ω_R = √(Ω₁² + Δ²)`, the
detuned-pulse population `P_b = ½(Ω₁/Ω_R)²[1 − cos Ω_R t]` (the squared magnitude
of Bordé's Eq. 13 off-diagonal element), and the Ramsey–Bordé recoil splitting of
the interferometer paths. (A review fixes populations and the Rabi structure, not
the off-diagonal phase sign — that is pinned by the primary source above.)
