Set of functions to calculate the electronic structure of a spiral magnetic in face rotating at a constant frequency embedded into a s-wave superconductor

Parameters used in functions

Continuum Model

omega: Float, energy in Green's function
kx: Float, momentum along the interface, in units of kf
kf: Float, fermi momentum
km: Float, spiral pitch of interface, in units of kf
Delta: Float, superconductor pairing, should be real
B: Float, effective zeeman field, should be less than Delta for a gapped system
sigma: +-1, index for the spin sectors
Cm: Float, unitless magnetic scattering
theta: Float between -pi and pi, out of plane angle of the spiral

Tight Binding Model

omega: Float, energy in Green's function
kx: Float between -pi and pi, momentum along the interface
t: Float, bandwidth, set to 1 to make everything unitless
mu: Float, chemical potential, in units of t
km: Float between - pi and pi, spiral pitch of interface
Delta: Float, superconductor pairing, should be real, in units of t
B: Float, effective zeeman field, should be less than Delta for a gapped system, in units of t
sigma: +-1, index for the spin sectors
Vm: Float, magnetic scattering, in units of t
theta: Float between -pi and pi, out of plane angle of the spiral
