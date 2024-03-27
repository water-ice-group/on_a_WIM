# utilities for module

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
        opos,hpos,box_dim = self.positions_unorg()
        print()
    
        return (opos,hpos,box_dim)
        

    def positions(self):

        '''Load trajectory for water and carbon dioxide.'''

        opos_traj = []
        h1_traj = []
        h2_traj = []
        cpos_traj = []
        ocpos1_traj = []
        ocpos2_traj = []
        box_dim = []
        length = len(self._u.trajectory[self._start:self._end])
        print('Parsing through frames.')
        print(f'Total: {length}.')
        
        def process_frame(ts):
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
        num_cores = int(multiprocessing.cpu_count()/2)
        
        print()
        print(f'Number of cores: {num_cores}')
        print()
        print('Processing frames.')
        result = Parallel(n_jobs=num_cores,backend='threading')(delayed(process_frame)(ts) for ts in self._u.trajectory[self._start:self._end])
        print(opos_traj)
        return (opos_traj, h1_traj, h2_traj, cpos_traj, ocpos1_traj, ocpos2_traj, box_dim)


    def positions_unorg(self):

        '''Load trajectory for water. Account for hydronium ions (cannot perform molecule aggregation).'''
        opos_traj = []
        hpos_traj = []
        box_dim = []
        length = len(self._u.trajectory[self._start:self._end])
        print('Parsing through frames.')
        print(f'Total: {length}.')
        
        def process_frame(ts):
            opos = self._u.select_atoms('name' + ' OW').positions
            hpos = self._u.select_atoms('name' + ' H').positions
            opos_traj.append(opos)
            hpos_traj.append(hpos_traj)
            box_dim.append(self._u.dimensions)
        
        num_cores = int(multiprocessing.cpu_count()/2)
        
        print()
        print(f'Number of cores: {num_cores}')
        print()
        print('Processing frames.')
        result = Parallel(n_jobs=num_cores, backend='threading')(delayed(process_frame)(ts) for ts in self._u.trajectory[self._start:self._end])
        
        return (opos_traj, hpos_traj, box_dim)