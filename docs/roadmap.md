

# Roadmap

* [x] Implement Gaussian-shaped beams, i.e. varying Rabi frequency over the atom's position in the XY plane
* Sim LMT with realistic params
* Try Rapid adiabatic passage
* [ ] TODO: Simulate shaped (optimal-control) pulses properly, by integrating
  the time-dependent Hamiltonian over the recorded amplitude/phase profiles
  (e.g. `JessePulse` / `JessePulseLMT` from icl_experiments). Until then,
  shaped pulses are modelled as plain pi pulses with a corrected Stark shift
  (`Pulse.stark_rabi_frequency`), and multi-arm shaped pulses as several
  SIMULTANEOUS arm-restricted pi pulses (`Pulse.restrict_to_m_ground` +
  `Pulse.simultaneous_with_previous`) whose off-resonant cross-talk with the
  other arms is suppressed in code -- see
  `notebooks/true_trajectory_double_launch_rid74108.ipynb`. These stand-ins
  are supported by the trajectory inference only; the full quantum propagator
  raises `NotImplementedError` on them.
