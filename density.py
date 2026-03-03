import numpy as np
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib import distances
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


class Density:
    """Calculate the density of species relative to the loaded WC interface.
    
    Args:
        universe: MDAnalysis universe object.
    """
    
    def __init__(self,universe):
        self._u = universe
        
    

    def proximity(self,WC_inter,inp,boxdim,upper=25,result='mag',cutoff=False):
        """Obtain the proximities of molecules to the WC interface.
        
        Uses simple z-axis normal vector for distance calculation.
        Input is a SINGLE FRAME.
        
        Args:
            WC_inter (ndarray): Interface coordinates.
            inp (ndarray): Molecule positions.
            boxdim (ndarray): Box dimensions for PBC.
            upper (float): Upper z-boundary (unused, kept for API compatibility).
            result (str): Return type - 'mag', 'vect', or 'both'.
            cutoff (bool): Whether to apply cutoff (unused, kept for API compatibility).
            
        Returns:
            list or ndarray or tuple: Depending on `result`:
                - 'mag': List of signed distances
                - 'vect': Array of vectors from interface to molecule
                - 'both': Tuple of (mag, vect)
        """


        ######################################################################
        ######### Obtain the proximity of molecules to an interface ##########
        ######################################################################

        # try:
        #     dist_mat = distance_array(inp, WC_inter, box=boxdim) 
        # except:
        #     dist_mat = distance_array(inp, np.array(WC_inter), box=boxdim)
        # surf_div = len(WC_inter)//2
        WC_array = np.asarray(WC_inter)  # Handles both list and array
        dist_mat = distance_array(inp, WC_array, box=boxdim)
        surf_div = len(WC_array) // 2


        # find closest interface point for each molecule
        proxim = np.min(dist_mat, axis=1)
        loc = [np.argmin(dist_mat[i]) for i in range(len(proxim))]
        
        mag = []
        vect_list = []

        for i in range(len(inp)):
            
            # obtain vector pointing from interface to molecule
            vect = distances.minimize_vectors(inp[i]-WC_inter[loc[i]],box=boxdim)
            vect_list.append(vect) # unaltered surface -> molecule vector

            if loc[i] >= surf_div:
                vect = -vect # reverse vector for upper interface

            # calculate dot product with z-axis normal
            norm = [0,0,1]
            prox = np.dot(vect,norm)
            mag.append(prox)
            
        
        if result == 'mag':
            return mag
        elif result == 'vect':
            return np.array(vect_list) 
        elif result == 'both':
            return (mag,np.array(vect_list))
        else:
            raise ValueError("Invalid result type. Choose 'mag', 'vect', or 'both'.")
        

    # ------------------------------------------------------------------------------
    # Redacted freature. This was an attempt to calculate proximity using local surface normals: minimal change to the resulting profiles. 
    # The code is retained here for reference but is not currently used in the analysis.
    # ------------------------------------------------------------------------------

    # def proximity_normal(self,WC_inter,inp,boxdim,upper=25,result='mag',cutoff=False):
    #     '''Obtain the proximities of each particular molecule to the WC interface.'''
    #     '''Input of a SINGLE FRAME into the function.'''


    #     ######################################################################
    #     ######### Obtain the proximity of molecules to an interface ##########
    #     ######################################################################

    #     try:
    #         dist_mat = distance_array(inp, WC_inter, box=boxdim) 
    #     except:
    #         dist_mat = distance_array(inp, np.array(WC_inter), box=boxdim)

    #     surf_div = len(WC_inter)//2
    #     normals_1 = self.calculate_normal(WC_inter[:surf_div])
    #     normals_2 = self.calculate_normal(WC_inter[surf_div:])
    #     normals = np.concatenate((normals_1,normals_2),axis=0)

    #     proxim = np.min(dist_mat,axis=1)
    #     loc = [(np.where(dist_mat[i] == proxim[i])[0][0]) for i in range(len(proxim))] 
        
    #     mag = []
    #     vect_list = []

    #     for i in range(len(inp)):
            
    #         # obtain vector pointing from interface to molecule
    #         vect = distances.minimize_vectors(inp[i]-WC_inter[loc[i]],box=boxdim)
    #         vect_list.append(vect) # unaltered surface -> molecule vector

    #         if loc[i] >= surf_div:
    #             vect = -vect # reverse vector 

    #         # obtain normal vector at interface
    #         norm = normals[loc[i]]

    #         # calculate dot product
    #         prox = np.dot(vect,norm)
    #         mag.append(prox)
            
        
            
    #     if result == 'mag':
    #         return mag
    #     elif result == 'vect':
    #         return np.array(vect_list) 
    #     elif result == 'both':
    #         return (mag,np.array(vect_list))
    


    # def calculate_normal(self, grid):
    #     # Infer n and m assuming grid is of shape (n*m, 3)
    #     total_points = grid.shape[0]
        
    #     # If n and m are not known, assume it's a square grid for simplicity
    #     # Otherwise, you can infer them from the grid structure if it's rectangular
    #     m = int(np.sqrt(total_points))  # Number of columns
    #     n = total_points // m  # Number of rows, assuming it's a rectangular grid

    #     # Reshape the grid into a 2D form
    #     x = grid[:, 0].reshape((n, m))
    #     y = grid[:, 1].reshape((n, m))
    #     z = grid[:, 2].reshape((n, m))

    #     # Calculate gradients, manually enforcing periodicity
    #     dz_dx = (z[(np.arange(n) + 1) % n, :] - z[(np.arange(n) - 1) % n, :]) / 2
    #     dz_dy = (z[:, (np.arange(m) + 1) % m] - z[:, (np.arange(m) - 1) % m]) / 2

    #     dx_dx = (x[(np.arange(n) + 1) % n, :] - x[(np.arange(n) - 1) % n, :]) / 2
    #     dy_dx = (y[(np.arange(n) + 1) % n, :] - y[(np.arange(n) - 1) % n, :]) / 2

    #     dx_dy = (x[:, (np.arange(m) + 1) % m] - x[:, (np.arange(m) - 1) % m]) / 2
    #     dy_dy = (y[:, (np.arange(m) + 1) % m] - y[:, (np.arange(m) - 1) % m]) / 2

    #     # Tangent vectors
    #     tangent_x = np.stack([dx_dx, dy_dx, dz_dx], axis=-1)  # Tangent vector in x direction
    #     tangent_y = np.stack([dx_dy, dy_dy, dz_dy], axis=-1)  # Tangent vector in y direction

    #     # Cross product of the tangent vectors to get normal vectors
    #     normals = np.cross(tangent_x, tangent_y, axis=-1)

    #     # Normalize the normal vectors
    #     magnitudes = np.linalg.norm(normals, axis=-1)
    #     normals /= magnitudes[..., np.newaxis]

    #     return normals.reshape(-1, 3)









    def hydroniums(self,ox,hy,boxdim,cutoff=1.5):
        """Identify hydronium ions based on O-H distances.
        
        Args:
            ox (ndarray): Oxygen atom positions.
            hy (ndarray): Hydrogen atom positions.
            boxdim (ndarray): Box dimensions for PBC.
            cutoff (float): O-H distance cutoff for bonding.
            
        Returns:
            list: Positions of identified hydronium oxygen atoms.
        """

        dist_mat = distance_array(ox, hy, box=boxdim)

        within_threshold_mask = dist_mat <= cutoff
        within_threshold_rows = np.any(within_threshold_mask, axis=1)
        atoms_within_distance = np.where(within_threshold_rows)[0]

        poss_hydro = []
        for i in range(len(atoms_within_distance)):
            if len(atoms_within_distance[i]) == 3:
                poss_hydro.append(ox[i])
        

        if len(poss_hydro) == 0:
            return []
        elif len(poss_hydro) == 1:
            return poss_hydro
        elif len(poss_hydro) > 1:
            dist_mat = distance_array(np.array(poss_hydro), hy, box=boxdim)
            smallest_values = np.partition(dist_mat, 3, axis=1)[:, :3]
            column_indices = np.argsort(dist_mat, axis=1)[:, :3]
            result = [(smallest_values[i, j], i, column_indices[i, j]) for i in range(len(dist_mat)) for j in range(3)]
            
            sumation = [sum(result[i][0]) for i in result]
            proxim = np.min(sumation) # obtain min for each row/atom. 
            loc = [(np.where(dist_mat[i] == proxim[i])[0][0]) for i in range(len(proxim))]
            return poss_hydro[loc]

        else:
            return []






def dens_plot(data_Oxygen,data_Carbon=None,lower=-15,upper=15):
    """Plot density profiles.
    
    Args:
        data_Oxygen (tuple): (density, distance) arrays for oxygen/water.
        data_Carbon (tuple, optional): (density, distance) arrays for carbon species.
        lower (float): Lower x-axis limit.
        upper (float): Upper x-axis limit.
    """
    smooth = 4
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(data_Oxygen[1][:-1:smooth],data_Oxygen[0][::smooth],'--',
            color='r',
            label = r'$\rho \mathrm{(H_2O)}$')
    zeros = [0]*len(data_Oxygen[0][:-1:smooth])
    ax.fill_between(data_Oxygen[1][:-1:smooth],zeros,data_Oxygen[0][::smooth],
                    color='red',
                    alpha=0.2)

    if data_Carbon is not None:
        ax.plot(data_Carbon[1][:-1:smooth],data_Carbon[0][::smooth],'--',
                color='black',
                label = r'$\rho \mathrm{(CO_2)}$')
        zeros = [0]*len(data_Carbon[1][:-1:smooth])
        ax.fill_between(data_Carbon[1][:-1:smooth],zeros,data_Carbon[0][::smooth],
                        color='black',
                        alpha=0.2)
        
    ax.set_xlim(lower,upper)
    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)',size=12)
    ax.set_ylabel('Density (g/ml)',size=12)
    ax.tick_params(axis="x",which='both',direction="in",labelsize=12)
    ax.tick_params(axis="y",which='both',direction="in",labelsize=12)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.04))

    ax.legend(loc='upper right')
    plt.savefig('./outputs/dens_plot.pdf',dpi=400,bbox_inches='tight',facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
        
