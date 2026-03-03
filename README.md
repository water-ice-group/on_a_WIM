# on_a_WIM
Willard-Chandler Interfacial Module - python module for generating instantaneous interfaces using the Willard Chandler formalism.

https://pubs.acs.org/doi/10.1021/jp909219k

Based on MDAnalysis. Requires input in .pdb/.dcd format. Coordinates must be centered such that the centre of mass resides at zero. 

---

# Contents
* `WillardChandler.py`: Main file.
* `utilities.py`: Helper functions for loading trajectory.
* `interface`: WC code for generating interfaces.
* `density.py`: Functions for obtaining density profiles using WC interface.
* `orientation.py`: Functions for obtaining orientational profiles.
* `hbondz.py`: Functions for obtaining hydrogen bonding profiles.
* `rdf.py`: Functions for obtaining near-surface RDFs (still in progress).
* `run.py`: Example script showing how to use code for simple air-water interface.
* `ref_coords.pdb`/`test.dcd`: Test system.       
