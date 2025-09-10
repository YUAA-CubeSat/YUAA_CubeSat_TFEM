# Thermal finite-element model for the Yale Undergrad Aerospace Association's CubeSat. 
Simple Python model that breaks up the CubeSat into lumped masses to simulate the on-orbit thermal environment.

This code offers the following capabilities:
* Define lumped masses with specific heat capacities, absorptivities, emissivities, and optionally one flat external surface.
* Connect lumped masses with thermal conductances.
* Numerically propagate the rotation of a rigid body to simulate on-orbit attitude.
* Query the SGP4 near-Earth satellite orbital propagator and JPL ephemerides to obtain the satellite's position relative to the Earth and Sun.
* Finally, package all of this together to numerically propagate a satellite in its orbit around Earth while tracking heating from the Sun and Earth, radiation loss to space, and equalization of heat among connected components.

## Dependencies
The `requirements.txt` file in this repo lists ALL dependencies, even the optional ones.
### Numerical integration
* This code uses [pint](https://pint.readthedocs.io/en/stable/) to allow initializing material properties without having to worry about unit conversions. `pint` is not used after initialization, so in principle you could do without it.
* `scipy`, `numpy`, and `pandas`. Nothing fancy.
### Orbital model
* This repo includes a simple orbital model which assumes a circular Earth orbit with a constant beta angle. This model has no dependencies.
* To accurately track the satellite relative to the Earth and Sun, including seasonal changes, this code uses:
   * [sgp4](https://pypi.org/project/sgp4/) for finding the position of the satellite relative to the Earth for a given time and two-line elements specifying the orbital parameters.
   * [jplephem](https://pypi.org/project/jplephem/) for finding the position of the Earth relative to the Sun for a given time. Read the jplephem documentation to understand how to select an ephemeris kernel and where to download it from (they are too big to host on GitHub). `de440.bsp` is a good default.
### Visualization
* This repo includes some simple plotting functions written for `matplotlib` that could help visualize thermal results.
* I found `matplotlib` to be too slow for visualizing rotation kinematics. This repo therefore includes an optional 3D animation script written for `vtk`.

## Contents
* `constants.py`: physical, material, and astronomical constants defined as `pint` Quantities.
* `lumped_mass.py`: class definitions for lumped mass objects. Linear subclassing is used to progressively expand the functionality of a lumped mass.
* `orbit.py`: class definitions for orbital models. Includes a simple circular orbit with constant beta angle, and a more accurate orbit using SGP4 and jplephem.
* `rotation_sim.py`: class definition for a rotating rigid body. Rotation is propagated with Euler's equation in body coordinates, tracking the body rotation matrix (in inertal coordinates) and the body angular velocity vector (in body coordinates).
* `rotation_sim_test.py`: a simple script that propagates the rigid body rotation for a minute and then generates an interactive 3D animation using vtk.
### Unit conventions

## Scope, limitations, extensions


## References
Much of the methodology and some of the math for this project is taken from: \
[1] Versteeg, Casper, and David Cotten. n.d. “Preliminary Thermal Analysis of Small Satellites.” https://smallsat.uga.edu/images/documents/papers/Preliminary_Thermal_Analysis_of_Small_Satellites.pdf.

Material values are sourced from all over, including datasheets stored in the YUAA CubeSat private Google Drive, see citations in-code.
