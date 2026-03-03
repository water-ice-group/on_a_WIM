import numpy as np
import os
from MDAnalysis.analysis.distances import distance_array
from scipy import interpolate
import MDAnalysis as mda
from MDAnalysis.lib import distances
import warnings

# Suppress MDAnalysis warnings
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis')


default_epsilon = 2.4
default_crit_dens = 0.016

class WC_Interface:
    """Module for generating a Willard-Chandler interface.
    
    Uses coarse-grained density fields to identify the instantaneous
    liquid-vapour interface.
    
    Args:
        universe: MDAnalysis universe object.
        grid_spacing (int): Number of grid points along z-axis.
        lower_z (float): Lower bound for interface detection.
        upper_z (float): Upper bound for interface detection.
    """

    def __init__(self, universe, grid_spacing=100, lower_z=10, upper_z=30):
        
        self._u = universe          # load universe
        self._gs = grid_spacing     # load the grid spacing along z
        self._lz = lower_z          # load the lower bounds for interface detection. 
        self._uz = upper_z          # load the upper bounds for interface detection.
    









    ##########################################################################
    ###################### Generate the WC interface #########################
    ##########################################################################
    
    def gaussian(self,r,eps=default_epsilon,dimens=3): 
        """Calculate coarse-grained density contribution at a point.
        
        Args:
            r (ndarray): Distance array.
            eps (float): Gaussian width parameter.
            dimens (int): Dimensionality of the system.
            
        Returns:
            ndarray: Gaussian density contributions.
        """
        function = (2*np.pi*eps**2)**(-dimens/2) * np.exp(-(r**2)/(2*eps**2))
        return function
        
        
        
    def grid_spacing(self):
        """Create spatial grid extending over the simulation box.
        
        Note: Creates grid once at beginning. May have issues with 
        variable box sizes (NPT).
        
        Returns:
            ndarray: Grid points with shape (n_points, 3).
        """
        x_dim = self._u.dimensions[0]
        y_dim = self._u.dimensions[1]
        x_spacing = int(x_dim)
        y_spacing = int(y_dim)

        grid = []
        for i in np.linspace(0, x_dim, x_spacing):
            for j in np.linspace(0, y_dim, y_spacing):
                for k in np.linspace(self._lz, self._uz, self._gs):
                    grid.append([i, j, k])

        return np.array(grid)
        
 
    def CG_field(self,manifold,opos,boxdim):
        """Calculate coarse-grained density field at grid points.
        
        Args:
            manifold (ndarray): Grid points.
            opos (ndarray): Oxygen atom positions.
            boxdim (ndarray): Box dimensions for PBC.
            
        Returns:
            ndarray: Density field values at each grid point.
        """
        array = np.array(manifold)
        # opos_wrap = distances.apply_PBC(opos, boxdim)
        
        # Calculate distances from grid points to atoms
        dist = distance_array(array, opos, box=boxdim)
        
        # Sum Gaussian contributions from all atoms
        dens_array = self.gaussian(dist)
        density_field = np.sum(dens_array, axis=1)

        return density_field
                
    
    def criteria(self,O_atoms,grid,boxdim=None,crit=default_crit_dens):
        """Identify the interface by finding points at critical density.
        
        Locates the quasi-2D surface where the coarse-grained density
        equals the critical value (half bulk water density).
        
        Args:
            O_atoms (ndarray): Oxygen atom positions.
            grid (ndarray): Grid points.
            boxdim (ndarray): Box dimensions.
            crit (float): Critical density value.
            
        Returns:
            ndarray: Interface coordinates (lower and upper surfaces).
        """
        field = self.CG_field(grid, O_atoms, boxdim)
        n_xy_points = int(len(field) / self._gs)

        inter_lower = np.zeros(shape=(n_xy_points, 3))
        inter_upper = np.zeros(shape=(n_xy_points, 3))

        for i in range(n_xy_points):

            # extract field values at different z along point in the x/y.
            z_field = field[i*self._gs:(i+1)*self._gs]
            # extract corresponding z coordinates along point in x/y frame. 
            z_pos = grid[i*self._gs:(i+1)*self._gs]

            div = len(z_field)//2
            lower_field, upper_field = z_field[:div], z_field[div:]
            lower_pos, upper_pos = z_pos[:div], z_pos[div:]

            # Find closest point to critical density (lower interface)
            diff_lower = np.abs(lower_field - crit)
            min_idx_lower = np.argmin(diff_lower)
            inter_lower[i] = lower_pos[min_idx_lower]

            # Find closest point to critical density (upper interface)
            diff_upper = np.abs(upper_field - crit)
            min_idx_upper = np.argmin(diff_upper)
            inter_upper[i] = upper_pos[min_idx_upper]

        out = np.concatenate((inter_lower,inter_upper),axis=0)
        return out












    ##########################################################################
    ################################# Deformation ############################
    ##########################################################################

    def dist_surf_deform(self,WC):
        """Calculate surface deformation relative to mean position.
        
        Args:
            WC (ndarray): Interface coordinates for a single frame.
            
        Returns:
            ndarray: Z-coordinates relative to average position.
        """
        WC_array = np.asarray(WC)
        z_coord = WC_array[:, 2]
        avg = np.mean(z_coord)
        return z_coord - avg










    ##########################################################################
    ################################# Splining ###############################
    ##########################################################################
                
                
    def spline(self,WC_inter,mesh_size=100):
        """Spline the interface to obtain a finer grid.
        
        Args:
            WC_inter (ndarray): Interface coordinates.
            mesh_size (int): Number of points in interpolated grid.
            
        Returns:
            list: Interpolated coordinates on finer grid.
        """

        WC_array = np.array(WC_inter)
        x = WC_array[:, 0]
        y = WC_array[:, 1]
        z = WC_array[:, 2]

        # interpolate
        tck = interpolate.bisplrep(x, y, z)
        
        xy = self._u.dimensions[0]
        mesh = complex(0,mesh_size)
        
        # interpolate over new mesh_size x mesh_size grid
        xnew_edges, ynew_edges = np.mgrid[0:xy:mesh, 0:xy:mesh]
        xnew = xnew_edges[:-1, :-1] + np.diff(xnew_edges[:2, 0])[0] / 2.
        ynew = ynew_edges[:-1, :-1] + np.diff(ynew_edges[0, :2])[0] / 2.
        
        # evaluate spline
        znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
        

        # build coordinate list
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
        """Create MDAnalysis universe containing interface coordinates.
        
        Args:
            WC (list): Interface coordinates with structure [frames[positions]].
            
        Returns:
            Universe: MDAnalysis universe with interface trajectory.
        """

        no_points = len(WC[0])
        surf_u = mda.Universe.empty(no_points,trajectory=True)
        surf_u.add_TopologyAttr('name', ['S']*no_points) # use 'S' for visualisation purposes. 
    
        coordinates = np.array(WC)
        surf_u.load_new(coordinates)
        return surf_u




    def save_coords(self,WC):
        """Save generated interface coordinates.
        
        Args:
            WC (list): Interface coordinates for all frames.
        """
        wc_univ = self.gener_WC_univ(WC)
        sel = wc_univ.select_atoms('all')

        box_dims = [
            self._u.dimensions[0],
            self._u.dimensions[1],
            self._uz,
            90.0, 90.0, 90.0
        ]

        with mda.Writer("./outputs/ref_inter.pdb",len(WC[0])) as W:
            sel.dimensions = box_dims
            W.write(sel)

        with mda.Writer("./outputs/inter.dcd",len(WC[0])) as W:
            for ts in wc_univ.trajectory:
                W.write(sel)




    def load_coords(self):
        """Load previously saved interface coordinates.
        
        Returns:
            Universe: MDAnalysis universe with interface trajectory,
                     or None if files not found.
        """
        pdb = './outputs/ref_inter.pdb'
        trj = './outputs/inter.dcd'

        if os.path.isfile(pdb) and os.path.isfile(trj):
            print('Loading files.')
            u = mda.Universe(pdb,trj)

            return u
        
        else:
            print('No interface files detected in ./outputs.')
            return None


                
                

        


        
        
        
