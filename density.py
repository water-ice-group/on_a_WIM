# script for calculating the Willard-Chandler interface
# https://pubs.acs.org/doi/pdf/10.1021/jp909219k


import numpy as np
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib import distances
from scipy import interpolate
from scipy.interpolate import griddata
from interface import WC_Interface
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


class Density:
    '''Calculate the density of species relative to the loaded WC interface.'''
    
    def __init__(self,universe):
        
        self._u = universe
        
    

    def proximity(self,WC_inter,inp,boxdim,upper=25,result='mag',cutoff=False):
        '''Obtain the proximities of each particular molecule to the WC interface.'''
        '''Input of a SINGLE FRAME into the function.'''


        ######################################################################
        ######### Obtain the proximity of molecules to an interface ##########
        ######################################################################

        try:
            dist_mat = distance_array(inp, WC_inter, box=boxdim) 
        except:
            dist_mat = distance_array(inp, np.array(WC_inter), box=boxdim)

        normals = self.calculate_normal(WC_inter)

        proxim = np.min(dist_mat,axis=1)
        loc = [(np.where(dist_mat[i] == proxim[i])[0][0]) for i in range(len(proxim))] 
        
        mag = []
        vect_list = []

        for i in range(len(inp)):
            
            # obtain vector pointing from interface to molecule
            vect = distances.minimize_vectors(inp[i]-WC_inter[loc[i]],box=boxdim)

            # obtain normal vector at interface
            norm = normals[loc[i]]

            # calculate dot product
            prox = np.dot(vect,norm)
            
            mag.append(prox)
            vect_list.append(vect)
        
            
        if result == 'mag':
            return mag
        elif result == 'vect':
            return np.array(vect_list) # vector from the interface to the molecule
        elif result == 'both':
            return (mag,np.array(vect_list))
        


    def calculate_normal(self,grid):

        x = grid[:, 0]
        y = grid[:, 1]
        z = grid[:, 2]

        # Calculate the periodic virtual grid
        x_extended = np.concatenate((x[-1:], x, x[:1]))
        y_extended = np.concatenate((y[-1:], y, y[:1]))
        z_extended = np.concatenate((z[-1:], z, z[:1]))

        # Calculate gradients along x and y axes using central differences on the extended grid
        dx = np.gradient(x_extended)
        dy = np.gradient(y_extended)

        # Initialize an array to store normal vectors
        normals = np.zeros_like(grid)

        # Calculate the normal vectors using cross product
        normals[:, 0] = -dy[1:-1]  # x component of the normal vector
        normals[:, 1] = dx[1:-1]   # y component of the normal vector
        normals[:, 2] = z          # z vector. Not true normal vector. Useful for directionality of distances. 

        # Normalize the normal vectors
        magnitudes = np.sqrt(np.sum(normals**2, axis=1))
        normals /= magnitudes[:, np.newaxis]

        return normals



def hydroniums(self,ox,hy,boxdim,cutoff=1.5):

    dist_mat = distance_array(ox, hy, box=boxdim)

    within_threshold_mask = dist_mat <= cutoff
    within_threshold_rows = np.any(within_threshold_mask, axis=1)
    atoms_within_distance = np.where(within_threshold_rows)[0]

    poss_hydro = []
    for i in range(len(atoms_within_distance)):
        if len(atoms_within_distance[i]) == 3:
            poss_hydro.append(ox[i])
    
    if len(poss_hydro) == 1:
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
    
    # FONT -----------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "custom"
    # ----------------------------------------------------------------------------

    smooth = 4
    fig, ax = plt.subplots()
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
    ax.set_xlabel('Distance ($\mathrm{\AA}$)',size=12)
    ax.set_ylabel('Density (g/ml)',size=12)
    ax.tick_params(axis="x",which='both',direction="in",labelsize=12)
    ax.tick_params(axis="y",which='both',direction="in",labelsize=12)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.04))


    ax.legend(loc='upper right')
    plt.savefig('./outputs/dens_plot.pdf',dpi=400,bbox_inches='tight',facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
        
