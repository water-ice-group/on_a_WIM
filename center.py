# #!/usr/bin/env python3

import sys
import MDAnalysis as mda
import numpy as np

pdb = sys.argv[1]
trj = sys.argv[2]

# initialize topology first for cell dimensions:
u = mda.Universe(pdb)
dimensions = u.dimensions

# initalize full trajectory and assign cell dimensions:
u = mda.Universe(pdb, trj)

u.dimensions = dimensions
u.add_TopologyAttr('charges')
sel = u.select_atoms('name OW H')

step = 2
with mda.Writer('centered.dcd', n_atoms=u.atoms.n_atoms) as w:
    for ts in u.trajectory[::step]:
        com = sel.center_of_mass(pbc=True)
        u.atoms.positions-=com
        w.write(u.atoms)
