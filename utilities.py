# utilities for module
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.transformations.wrap import wrap,unwrap
import numpy as np

class AtomPos: 
    
    def __init__(self,universe,end_step=None):
        
        self._u = universe
        self._end = end_step if end_step is not None else (len(self._u.trajectory)-1)
        
        
        
    def prepare(self):
        
        unopos = self.water()[0]
        unh1pos = self.water()[1]
        unh2pos = self.water()[2]
        self.wrap()
        opos = self.water()[0]
        h1pos = self.water()[1]
        h2pos = self.water()[2]
        cpos = self.carbon()[0]
    
        return (unopos,unh1pos,unh2pos,opos,h1pos,h2pos,cpos)
        
    def wrap(self):
        
        ag = self._u.atoms
        transform = wrap(ag)
        self._u.trajectory.add_transformations(transform)
        
    
        
    

    def water(self):
        '''Load trajectory for water.'''
        opos_traj = []
        h1_traj = []
        h2_traj = []
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
    
        return (opos_traj,h1_traj,h2_traj)
    
    
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

