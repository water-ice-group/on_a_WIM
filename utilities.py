# utilities for module
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.transformations.wrap import wrap,unwrap
import numpy as np
import tqdm as tqdm

class AtomPos: 
    
    def __init__(self,universe,end_step=None):
        
        self._u = universe
        self._end = end_step if end_step is not None else (len(self._u.trajectory)-1)
        
        
        
    def prepare(self):
        
        print('Unwrapped coordinates')
        unopos,unh1pos,unh2pos,uncpos,unocpos = self.positions()
        print()

        print('Wrapped coordinates')
        self.wrap()
        opos,h1pos,h2pos,cpos,ocpos = self.positions()
    
        return (unopos,unh1pos,unh2pos,opos,h1pos,h2pos,uncpos,unocpos,cpos,ocpos)
        
    def wrap(self):
        
        ag = self._u.atoms
        transform = wrap(ag)
        self._u.trajectory.add_transformations(transform)
        

    def positions(self):
        '''Load trajectory for water.'''
        opos_traj = []
        h1_traj = []
        h2_traj = []
        cpos_traj = []
        ocpos_traj = []

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
                ocpos_traj.append(oc1pos)

            except:
                cpos = self._u.select_atoms('name' + ' C').positions
                ocpos = self._u.select_atoms('name' + ' OC').positions

                cpos_traj.append(cpos)
                ocpos_traj.append(ocpos)
    
        return (opos_traj,h1_traj,h2_traj,cpos_traj,ocpos_traj)
    
    
    def carbon(self):
        '''Load trajectory for the carbon species.
        Returns arrays of C and OC positions for each frame '''
        cpos_traj = []
        ocpos_traj = []
        for ts in self._u.trajectory[:self._end]:
            cpos = self._u.select_atoms('name C').positions
            ocpos = self._u.select_atoms('name OC').positions
            cpos_traj.append(cpos)  
            ocpos_traj.append(ocpos)
    
        return (cpos_traj,ocpos_traj)
    
    
    def conclude(self):
        ag = self._u.atoms
        transform = unwrap(ag)
        self._u.trajectory.add_transformations(transform)

