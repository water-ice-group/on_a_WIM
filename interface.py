# script for calculating the Willard-Chandler interface
# https://pubs.acs.org/doi/pdf/10.1021/jp909219k


import numpy as np
import csv
import os
from MDAnalysis.analysis.distances import distance_array
from utilities import AtomPos
from scipy import interpolate


class WC_Interface:
    
    '''Module for generating a Willard-Chandler interface and using this
    interface to calculate properties such as density and orientation.'''

    def __init__(self, universe, grid_spacing=100, **kwargs):
        
        self._u = universe
        self._gs = grid_spacing

    

    ##########################################################################
    ###################### Generate the WC interface #########################
    ##########################################################################
    
    def gaussian(self,r,eps=2.4,dimens=3): 
        '''Function for generating the coarse-grained density at a 
        single point (Eq. 2).'''
        
        function = (2*np.pi*eps**2)**(-dimens/2) * np.exp(-(r**2)/(2*eps**2))
        return function
        
        
        
    def grid_spacing(self):
        '''Create the spatial positions extending the entirety of the box.'''

        # obtain rough boundaries to the slab
        # frame = np.array(self._u.trajectory[0])
        # dist_array = distance_array(frame,frame,box=self._u.dimensions)
        # dim = np.max(dist_array)/2

        # z_coords = np.linspace(0,dim,self._gs/2) + np.linspace(self._u.dimensions[2],self._u.dimensions[2]-dim,self._gs/2)

        
        grid = []
        for i in np.linspace(0,self._u.dimensions[0],int(self._u.dimensions[0])):
            for j in np.linspace(0,self._u.dimensions[1],int(self._u.dimensions[1])):
                for k in np.linspace(0,self._u.dimensions[2],self._gs):
                    grid.append([i,j,k])
        
        return grid
        
 
    def CG_field(self,manifold,opos):
        '''Return the CG field for each of the spatial points (Eq 3).'''

        density_field = []
        array = np.array(manifold)
        dist =  distance_array(array,opos,box=self._u.dimensions)
        
        dens_array = self.gaussian(dist)
        density_field = np.sum(dens_array,axis=1)
        
        return density_field
                
    
    def criteria(self,O_atoms,grid,crit=0.016):
        '''Identify the quasi-2D surface by equating the points in the density
        field to a particular critical value, chosen here to be half the 
        density of water.'''
        
        manifold = grid
        field = self.CG_field(manifold,O_atoms)
        inter_lower = [] # need to account for both surfaces.
        inter_upper = []

        
        for i in range(int(len(field)/self._gs)):
            z_field = field[i*self._gs:(i+1)*self._gs]
            z_pos = manifold[i*self._gs:(i+1)*self._gs]
            
            div = int(len(z_field)/2)
            
            lower_field = z_field[:div]
            upper_field = z_field[div:]
            lower_pos = z_pos[:div]
            upper_pos = z_pos[div:]
            
            
            diff_lower = abs(lower_field - crit)
            min_z = min(diff_lower)
            min_index = np.where(diff_lower == min_z)[0][0]
            inter_lower.append(lower_pos[min_index])
            
            diff_upper = abs(upper_field - crit)
            min_z = min(diff_upper)
            min_index = np.where(diff_upper == min_z)[0][0]
            inter_upper.append(upper_pos[min_index])
            
        inter = inter_lower + inter_upper
        return inter

    ##########################################################################
    ################################# Splining ###############################
    ##########################################################################
                
                
    def spline(self,WC_inter):
        '''Spline the interface to obtain finer grid.'''
    
        WC_upper = WC_inter[int(len(WC_inter)/2):]
        WC_lower = WC_inter[:int(len(WC_inter)/2)]

        # do upper first 
        
        x_upper = [i[0] for i in WC_upper]
        y_upper = [i[1] for i in WC_upper]
        z_upper = [i[2] for i in WC_upper]
        

        # interpolate
        tck_upper = interpolate.bisplrep(x_upper, y_upper, z_upper)
        
        xy = self._u.dimensions[0]
        mesh = complex(0,100)
        
        # interpolate over new 100x100 grid
        xnew_edges, ynew_edges = np.mgrid[0:xy:mesh, 0:xy:mesh]
        xnew = xnew_edges[:-1, :-1] + np.diff(xnew_edges[:2, 0])[0] / 2.
        ynew = ynew_edges[:-1, :-1] + np.diff(ynew_edges[0, :2])[0] / 2.
        
        znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck_upper)
        
        coordinates_upper = []
        for i in range(len(znew)):
            for j in range(len(znew)):
                entry = [xnew[i,j],ynew[i,j],znew[i,j]]
                coordinates_upper.append(entry)
            
            
        # construct lower surface
            
        x_lower = [i[0] for i in WC_lower]
        y_lower = [i[1] for i in WC_lower]
        z_lower = [i[2] for i in WC_lower]
        

                    
        # interpolate
        tck_lower = interpolate.bisplrep(x_lower, y_lower, z_lower)
        

        
        znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck_lower)
        
        coordinates_lower = []
        for i in range(len(znew)):
            for j in range(len(znew)):
                entry = [xnew[i,j],ynew[i,j],znew[i,j]]
                coordinates_lower.append(entry)

        WC_surf = coordinates_upper + coordinates_lower
        return WC_surf
    




    ##########################################################################
    ############################ Visualisation ###############################
    ##########################################################################

    def save_coords(self,WC):
        '''Save the coordinates of the WC interface in a format readable 
        by VMD fo visualisation.'''
    
    
        iter_count = 0
        tot_frames = len(WC)
        
        if os.path.isfile('./outputs/surface.xyz'):
            os.remove('./outputs/surface.xyz')
        
        for i in WC: # parse through the frames
            iden = 'S'
            array = [0] * len(i)
        
            for x in range(len(i)):
                array[x] = [str(iden), i[x][0],i[x][1],i[x][2]]
            
            iter_count += 1

            
            with open('./outputs/surface.xyz', 'a') as f:
                writer  = csv.writer(f,delimiter=' ',quoting=csv.QUOTE_NONNUMERIC,
                                     quotechar=' ')
                line12 = [[str(len(i))],[f'Timeframe {iter_count}/{tot_frames}.']]
                writer.writerows(line12)
                writer.writerows(array)
                

                
                

        


        
        
        
