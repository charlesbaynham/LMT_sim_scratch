# LMT simulation scratch

Simulation of large-momentum-transfer (LMT) atom interferometer pulse sequences
for Sr-87, including a Bordé-representation pulse/propagation engine
(`lmt_sim/lmt_simulation.py`), a sequence layer (`lmt_sim/lmt_sequence.py`), and
synthetic camera imaging (`lmt_sim/imaging.py`). Design notes live in `docs/`.

---

## ⚠️ BIG TODO — proper Gaussian wave-packet tracking for imaging

**The synthetic camera imaging (`lmt_sim/imaging.py`) is currently a heuristic.**
Within one atom, branch amplitudes are binned onto the pixel grid and the
**complex field is convolved with a single fixed Gaussian** (the single-atom wave
packet, `SINGLE_ATOM_WAVEPACKET_SIGMA_M ≈ 10 µm`) before squaring; different atoms
in the ensemble add incoherently. So branches within ~a packet width interfere
and well-separated ones do not — the **wave-packet size**, not the arbitrary
pixel size, sets the interference range. (A literal "same pixel interferes" rule
silently destroys interferometer fringes whenever recombining arms land in
adjacent pixels, so we deliberately do not do that.)

This is still **not** physically correct. It uses one fixed blur for every
branch (real packets have their own size and expand ballistically per the
uncertainty principle), and it ignores the momentum (`m`-dependent) phase
gradient across each packet that suppresses interference between different-`m`
branches even when they spatially overlap.

**What we actually need (deliberately deferred — do not paper over it):**
give every branch a finite **Gaussian wave packet** (initial in-trap size,
expanding ballistically), propagate those packets **with their momentum phase**,
and overlap the **complex fields**. Interference then falls out of real spatial
overlap. **This is the entire reason the simulation carries per-branch position
and velocity** through every pulse and free-propagation step — that bookkeeping
exists precisely so wave-packet tracking can be added here.

Until that lands, every image from `lmt_sim.imaging` is an approximation. The
same warning is repeated at the top of `lmt_sim/imaging.py`.

### Placeholder blur size (`SINGLE_ATOM_WAVEPACKET_SIGMA_M ≈ 10 µm`)

As a stand-in for the (untracked) wave-packet extent at readout, images are
blurred by a single-atom Gaussian. Rough estimate for Sr-87 released from the
low harmonic-oscillator levels (n = 0..10) of an optical trap with ~10 µm beam
waist (trap frequency ~1 kHz):

- in-trap position spread `σ₀ ≈ 0.2–1.1 µm`;
- the uncertainty principle fixes a velocity spread `σ_v = ħ / (2 m σ₀)`;
- after ~4 ms of free fall, `σ(t) = sqrt(σ₀² + (σ_v t)²) ≈ 6 µm (n=0) … 28 µm (n=10)`.

`~10 µm` is a representative ground-to-low-`n` value. It is just a blur scale,
not a substitute for real wave-packet tracking (see above).
