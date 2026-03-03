# utilities for module
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.transformations.wrap import wrap,unwrap
import numpy as np
import tqdm as tqdm
import os
from joblib import Parallel, delayed
import multiprocessing

co_bond_cutoff = 1.6
hydronium_oh_cutoff = 1.2

class AtomPos: 
    """Load and prepare atomic positions from trajectory.
    
    Args:
        universe: MDAnalysis universe object.
        start_step (int, optional): Starting frame index.
        end_step (int, optional): Ending frame index.
    """

    def __init__(self,universe,start_step=None,end_step=None):
        
        self._u = universe
        self._start = start_step if start_step is not None else (0)
        self._end = end_step if end_step is not None else (len(self._u.trajectory)-1)


    ############################################################################
    ################ Prepare atomic positions for analysis #####################
    ############################################################################

    def prepare(self):
        """Prepare trajectory for water and carbon dioxide systems.
        
        Returns:
            tuple: (opos, h1pos, h2pos, cpos, ocpos1, ocpos2, box_dim)
        """
        if os.path.isdir('./outputs'):
            print('Output directory detected.')
        else:
            os.mkdir('./outputs')

        print()
        print('Obtaining atom coordinates.')
        opos,h1pos,h2pos,cpos,ocpos1,ocpos2,box_dim = self.positions()
        print()
        return (opos,h1pos,h2pos,cpos,ocpos1,ocpos2,box_dim)
    


    def prepare_undefined(self):
        """Prepare trajectory for metaD run where OC bond is not defined.
        
        Returns:
            tuple: (opos, hpos, cpos, ocpos, box_dim)
        """
        if os.path.isdir('./outputs'):
            print('Output directory detected.')
        else:
            os.mkdir('./outputs')

        print('\nObtaining atom coordinates.')
        opos, hpos, cpos, ocpos, box_dim = self.pos_Oc_undefined()
        print()
        return (opos,hpos,cpos,ocpos,box_dim)
    


    def prepare_unorg(self):
        """Prepare trajectory for unorganised water with hydronium ions.
        
        Returns:
            tuple: (opos, hpos, h3opos, box_dim)
        """
        if os.path.isdir('./outputs'):
            print('Output directory detected.')
        else:
            os.mkdir('./outputs')
        print('\nObtaining atom coordinates.')
        opos,hpos,h3opos,box_dim = self.positions_unorg()
        print()
    
        return (opos,hpos,h3opos,box_dim)
        



    def positions(self):
        """Load trajectory for water and carbon dioxide.
        
        Returns:
            tuple: (opos_traj, h1_traj, h2_traj, cpos_traj, ocpos_traj, hcpos_traj, box_dim)
        """
        opos_traj = []
        h1_traj = []
        h2_traj = []
        cpos_traj = []
        ocpos_traj = []
        hcpos_traj = []
        box_dim = []
        length = len(self._u.trajectory[self._start:self._end])
        print('Parsing through frames.')
        print(f'Total: {length}.')
        
        for ts in self._u.trajectory[self._start:self._end]:
            oh_dist = distance_array(self._u.select_atoms('name' + ' OW').positions, # distance array loaded from module
                                    self._u.select_atoms('name' + ' H').positions, 
                                    box=self._u.dimensions)
            idx = np.argpartition(oh_dist, 3, axis=-1)

            opos = self._u.select_atoms('name' + ' OW').positions
            h1pos = self._u.select_atoms('name' + ' H')[idx[:, 0]].positions
            h2pos = self._u.select_atoms('name' + ' H')[idx[:, 1]].positions
            opos_traj.append(opos)
            h1_traj.append(h1pos)
            h2_traj.append(h2pos)

            cpos = self._u.select_atoms('name' + ' C').positions
            ocpos = self._u.select_atoms('name' + ' OC').positions
            hcpos = self._u.select_atoms('name' + ' HC').positions
            cpos_traj.append(cpos)
            ocpos_traj.append(ocpos)
            hcpos_traj.append(hcpos)

            box_dim.append(self._u.dimensions)

        return (opos_traj, h1_traj, h2_traj, cpos_traj, ocpos_traj, hcpos_traj, box_dim)




    def pos_Oc_undefined(self):
        """Load trajectory for metaD run where OC bond is not defined.
        
        Identifies bonded/unbonded oxygens based on distance to carbon.
        
        Returns:
            tuple: (opos_traj, hpos_traj, cpos_traj, ocpos_traj, box_dim)
        """
        opos_traj = []
        hpos_traj = []
        cpos_traj = []
        ocpos_traj = []
        box_dim = []
        length = len(self._u.trajectory[self._start:self._end])
        print('Parsing through frames.')
        print(f'Total: {length}.')
        
        for ts in self._u.trajectory[self._start:self._end]:

            cpos = self._u.select_atoms('name' + ' C').positions
            cpos_traj.append(cpos)

            co_dist = distance_array(self._u.select_atoms('name' + ' C').positions, # distance array loaded from module
                                    self._u.select_atoms('name' + ' O').positions, 
                                    box=self._u.dimensions)
            
            bonded_O = np.where(co_dist[0] < co_bond_cutoff)[0]
            ocpos = self._u.select_atoms('name' + ' O')[bonded_O].positions
            ocpos_traj.append(ocpos)

            unbonded_O = np.where(co_dist[0] > co_bond_cutoff)[0]
            opos = self._u.select_atoms('name' + ' O')[unbonded_O].positions
            opos_traj.append(opos)

            hpos = self._u.select_atoms('name' + ' H').positions
            hpos_traj.append(hpos)

            box_dim.append(self._u.dimensions)
        
        return (opos_traj, hpos_traj, cpos_traj, ocpos_traj, box_dim)



    def positions_unorg(self):
        """Load trajectory for water with hydronium ions.
        
        Cannot perform molecule aggregation due to hydronium presence.
        
        Returns:
            tuple: (opos_traj, hpos_traj, h3opos_traj, box_dim)
        """
        opos_traj = []
        hpos_traj = []
        h3opos_traj = []
        box_dim = []

        length = len(self._u.trajectory[self._start:self._end])
        print('Parsing through frames.')
        print(f'Total: {length}.')
        

        for ts in self._u.trajectory[self._start:self._end]:

            oh_dist = distance_array(self._u.select_atoms('name' + ' OW').positions, # distance array loaded from module
                        self._u.select_atoms('name' + ' H').positions, 
                        box=self._u.dimensions)

            opos = self._u.select_atoms('name' + ' OW').positions
            hpos = self._u.select_atoms('name' + ' H').positions

            hydronium_indices = self.hydronium_crit(oh_dist)
            h3opos = self._u.select_atoms('name' + ' OW')[hydronium_indices].positions

            opos_traj.append(opos)
            hpos_traj.append(hpos)
            h3opos_traj.append(h3opos)
            box_dim.append(self._u.dimensions)
        
        return (opos_traj, hpos_traj, h3opos_traj, box_dim)
    



    def hydronium_crit(self, dist_arr):
        """Identify hydronium ions based on O-H distances.
        
        Hydronium ions have 3 hydrogens within the cutoff distance.
        
        Args:
            dist_arr (ndarray): Distance array between oxygens and hydrogens.
            
        Returns:
            ndarray: Indices of oxygen atoms identified as hydronium.
        """
        indices = np.argpartition(dist_arr, 3, axis=1)[:, :3]  
        elements = dist_arr[np.arange(len(dist_arr))[:, None], indices]
    
        below_threshold = np.all(elements < hydronium_oh_cutoff, axis=1)  
        row_indices = np.where(below_threshold)[0]  

        return row_indices
        


