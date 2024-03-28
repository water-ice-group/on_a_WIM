# script for calculating the Willard-Chandler interface
# https://pubs.acs.org/doi/pdf/10.1021/jp909219k


import numpy as np
import csv
import os
from MDAnalysis.analysis.distances import distance_array
from utilities import AtomPos
from scipy import interpolate
import MDAnalysis as mda
from MDAnalysis.lib import distances


class WC_Interface:
    
    '''Module for generating a Willard-Chandler interface and using this
    interface to calculate properties such as density and orientation.'''

    def __init__(self, universe, grid_spacing=100, lower_z=10, upper_z=30, **kwargs):
        
        self._u = universe          # load universe
        self._gs = grid_spacing     # load the grid spacing along z
        self._lz = lower_z          # load the lower bounds for interface detection. 
        self._uz = upper_z          # load the upper bounds for interface detection.
    

    ##########################################################################
    ###################### Generate the WC interface #########################
    ##########################################################################
    
    def gaussian(self,r,eps=2.4,dimens=3): 
        '''Function for generating the coarse-grained density at a 
        single point.'''
        
        function = (2*np.pi*eps**2)**(-dimens/2) * np.exp(-(r**2)/(2*eps**2))
        return function
        
        
        
    def grid_spacing(self):
        '''Create the spatial positions extending the entirety of the box.'''
        '''Create once at beginning => Difficulty with variable box sizes (NPT).'''

        grid = []
        x = self._u.dimensions[0]
        x_spacing = int(1*self._u.dimensions[0])
        y = self._u.dimensions[1]
        y_spacing = int(1*self._u.dimensions[1])

        for i in np.linspace(0,x - (x/x_spacing),x_spacing):
            for j in np.linspace(0,y - (y/y_spacing),y_spacing):
                for k in np.linspace(self._lz,self._uz,self._gs): # need to include 5A buffer to prevent zero point interference. 
                    grid.append([i,j,k])

        grid = np.array(grid)
        
        return grid
        
 
    def CG_field(self,manifold,opos,boxdim):
        '''Return the CG field for each of the spatial points.'''

        density_field = []
        array = np.array(manifold)
        opos_wrap = distances.apply_PBC(opos,boxdim)
        dist =  distance_array(array,opos,box=boxdim) # hopefully providing box dimensions will take care of wrapping etc. 

        dens_array = self.gaussian(dist) # return gaussian for each of the grid points (rows) calculated per atom (columns)
        density_field = np.sum(dens_array,axis=1) # sum the gaussians along the columns (atoms) for each row (grid point)
        
        return density_field
                
    
    def criteria(self,O_atoms,grid,boxdim=None,crit=0.016):
        '''Identify the quasi-2D surface by equating the points in the density
        field to a particular critical value, chosen here to be half the 
        density of water.'''
        
        
        field = self.CG_field(grid,O_atoms,boxdim)
        manifold = grid
        inter_tot = np.zeros(shape=(int(len(field)/self._gs),3))

        for i in range(int(len(field)/self._gs)):

            # extract field values at different z along point in the x/y.
            z_field = field[i*self._gs:(i+1)*self._gs]

            # extract corresponding z coordinates along point in x/y frame. 
            z_pos = manifold[i*self._gs:(i+1)*self._gs]

            
            diff = abs(z_field - crit)
            min_z = min(diff)
            min_idx = np.where(diff == min_z)[0][0]

            inter_tot[i] = z_pos[min_idx]

        return inter_tot


    ##########################################################################
    ################################# Deformation ############################
    ##########################################################################


    def dist_surf_deform(self,WC):

        '''Take single frame: return the z coordinates relative to average 
        position.'''

        z_coord = [i[2] for i in WC]
        avg = np.average(z_coord)
        output = [i-avg for i in z_coord]

        return output




    ##########################################################################
    ################################# Splining ###############################
    ##########################################################################
                
                
    def spline(self,WC_inter):
        '''Spline the interface to obtain finer grid.'''

        x = [i[0] for i in WC_inter]
        y = [i[1] for i in WC_inter]
        z = [i[2] for i in WC_inter]
        

        # interpolate
        tck = interpolate.bisplrep(x, y, z)
        
        xy = self._u.dimensions[0]
        mesh = complex(0,100)
        
        # interpolate over new 100x100 grid
        xnew_edges, ynew_edges = np.mgrid[0:xy:mesh, 0:xy:mesh]
        xnew = xnew_edges[:-1, :-1] + np.diff(xnew_edges[:2, 0])[0] / 2.
        ynew = ynew_edges[:-1, :-1] + np.diff(ynew_edges[0, :2])[0] / 2.
        
        znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
        
        coordinates = []
        for i in range(len(znew)):
            for j in range(len(znew)):
                entry = [xnew[i,j],ynew[i,j],znew[i,j]]
                coordinates.append(entry)
            
        
        return coordinates
    
    ##########################################################################
    #################### Coordinates & Visualisation #########################
    ##########################################################################

    def gener_WC_univ(self,WC):
        '''Create universe object containing coordinates.'''
        '''WC object has structure [frames[positions]]'''

        no_points = len(WC[0])
        sol = mda.Universe.empty(no_points,trajectory=True)
        sol.add_TopologyAttr('name', ['S']*no_points)
    
        coordinates = np.array(WC)

        sol.load_new(coordinates)

        return sol

    def save_coords(self,WC):
        '''Utility to save generated interface.'''

        wc_univ = self.gener_WC_univ(WC)
        sel = wc_univ.select_atoms('all')

        with mda.Writer("./outputs/ref_inter.pdb",len(WC[0])) as W:
            sel.dimensions = [self._u.dimensions[0], self._u.dimensions[1], self._uz, 90.0, 90.0, 90.0]
            W.write(sel)

        with mda.Writer("./outputs/inter.dcd",len(WC[0])) as W:
            for ts in wc_univ.trajectory:
                W.write(sel)

    def load_coords(self):
        pdb = './outputs/ref_inter.pdb'
        trj = './outputs/inter.dcd'

        if os.path.isfile(pdb) and os.path.isfile(trj):
            print('Loading files.')
            u = mda.Universe(pdb,trj)

            return u
        
        else:
            print('No interface files detected in ./outputs.')


                
                

        


        
        
        
