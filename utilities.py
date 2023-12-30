# utilities for module
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.transformations.wrap import wrap,unwrap
import numpy as np
import tqdm as tqdm
import os


class AtomPos: 
    
    def __init__(self,universe,end_step=None):
        
        self._u = universe
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
        

    def positions(self):
        '''Load trajectory for water.'''
        opos_traj = []
        h1_traj = []
        h2_traj = []
        cpos_traj = []
        ocpos1_traj = []
        ocpos2_traj = []
        box_dim = []

        length = len(self._u.trajectory[:self._end])
        print('Parsing through frames.')
        print(f'Total: {length}.')

        for ts in self._u.trajectory[:self._end]:
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
            

            try: 
                idx = np.argpartition(c_oc_dist, 3, axis=-1)
                cpos = self._u.select_atoms('name' + ' C').positions
                oc1pos = self._u.select_atoms('name' + ' OC')[idx[:, 0]].positions
                oc2pos = self._u.select_atoms('name' + ' OC')[idx[:, 1]].positions

                cpos_traj.append(cpos)
                ocpos1_traj.append(oc1pos)
                ocpos2_traj.append(oc2pos)

            except:  # allow exception for single co2 molecule (partitioning breaks for this.)
                cpos = self._u.select_atoms('name' + ' C').positions
                ocpos = self._u.select_atoms('name' + ' OC').positions

                cpos_traj.append(cpos)
                ocpos1_traj.append(ocpos)

            box_dim.append(self._u.dimensions)
    
        return (opos_traj,h1_traj,h2_traj,cpos_traj,ocpos1_traj,ocpos2_traj,box_dim)

