# Floppy Box Monte Carlo

This is a code to do Floppy Box Monte Carlo simulations as described in
`J. Chem. Phys. 137, 214101 (2012); doi: 10.1063/1.4767529`.

The method is meant to be used to get **candidate** crystal structures,
so it changes the volume and skewness of the simulation box, and ramps
up the pressure to very high values to pack the particles together.

The point of the algorithm is to simulate a small number of particles to
effectively simulate a unit cell of a crystal. This way one obtains the
close-packed structure.

# Features and issues

- It can perform floppy box simulations, so deformations and scalings of the box are enabled.
- It can reproduce the simple cubic and face-centered cubic for monodisperse hard-spheres.
- It **can not** be used with **any** anisotropic shape. Anisotropic particles require specialized overlapping algorithms, as well as rotations, that are not included in this code.
- The implementation is relatively fast, obviously it will scale with the number of particles as $O(N^2)$ because cell-lists cannot be implemented, and images must be checked one by one. But a simple test on an M4 chip from Apple resulted in around ~30 minutes for simulating 10 million steps for $N = 4$ monodisperse hard-spheres.

# Observations

- Large language models, specifically Claude Sonnet 4, ChatGPT 4o-mini-high and ChatGPT o3 were used to code parts of this implementation. Specifically, the part of the image lists. Indeed, image lists turned out to be the hardest part to implement and are not simple at all. Care must be taken in correctly implementing shifts, the minimum-image convention has to account for all 27 neighboring images, and the image lists have to include more information on the neighbors.
- Having skew boxes makes everything more complicated, so lattice reduction is a must, although most of the time the final configuration might not appear very much cubic and will most certainly be skewed.
