# utilities for module
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.transformations.wrap import wrap,unwrap
import numpy as np
import tqdm as tqdm
import os
from joblib import Parallel, delayed
import multiprocessing


class AtomPos: 
    
    def __init__(self,universe,start_step=None,end_step=None):
        
        self._u = universe
        self._start = start_step if start_step is not None else (0)
        self._end = end_step if end_step is not None else (len(self._u.trajectory)-1)
        
    def prepare(self):
        
        if os.path.isdir('./outputs'):
            print('Output directory detected.')
        else:
            os.mkdir('./outputs')

        print()
        print('Obtaining atom coordinates.')
        opos,h1pos,h2pos,cpos,ocpos1,ocpos2,box_dim = self.positions()
        print()
    
        return (opos,h1pos,h2pos,cpos,ocpos1,ocpos2,box_dim)
    
    def prepare_unorg(self):
        
        if os.path.isdir('./outputs'):
            print('Output directory detected.')
        else:
            os.mkdir('./outputs')

        print()
        print('Obtaining atom coordinates.')
        opos,hpos,h3opos,box_dim = self.positions_unorg()
        print()
    
        return (opos,hpos,h3opos,box_dim)
        

    def positions(self):

        '''Load trajectory for water and carbon dioxide.'''

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
            c_oc_dist = distance_array(self._u.select_atoms('name' + ' C').positions, # distance array loaded from module
                                    self._u.select_atoms('name' + ' OC').positions, 
                                    box=self._u.dimensions)

            cpos = self._u.select_atoms('name' + ' C').positions
            ocpos = self._u.select_atoms('name' + ' OC').positions
            hcpos = self._u.select_atoms('name' + ' HC').positions
            cpos_traj.append(cpos)
            ocpos_traj.append(ocpos)
            hcpos_traj.append(hcpos)

            box_dim.append(self._u.dimensions)

        return (opos_traj, h1_traj, h2_traj, cpos_traj, ocpos_traj, hcpos_traj, box_dim)


    def positions_unorg(self):

        '''Load trajectory for water. Account for hydronium ions (cannot perform molecule aggregation).'''
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

        inp = dist_arr
        threshold = 1.2 # A. Set the distance threshold for OH distance in hydronium ions
        indices = np.argpartition(inp, 3, axis=1)[:, :3]  
        elements = inp[np.arange(len(inp))[:, None], indices] 
        below_threshold = np.all(elements < threshold, axis=1)  
        row_indices = np.where(below_threshold)[0]  

        return row_indices
        


