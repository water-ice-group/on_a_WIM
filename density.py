# script for calculating the Willard-Chandler interface
# https://pubs.acs.org/doi/pdf/10.1021/jp909219k


import numpy as np
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib import distances
from scipy import interpolate
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
        
        pos = []
        wrap = distances.apply_PBC(inp,boxdim) # obtained the wrapped coordinates
        wrap = np.array(wrap)
        if cutoff==True:
            for i in range(len(wrap)):
                if (wrap[i][2] >= 0) and (wrap[i][2] < (2*upper)): # check that coordinates fall within given range of evaluation. 
                    pos.append(wrap[i]) # append the unwrapped coordinates?
            pos = np.array(pos)
        if cutoff==False:
            pos = wrap

        #WC_spline = np.array(WC_Interface(self._u).spline(WC_inter))  # obtain finer grid for better resolution of distances. 
        #WC_spline = WC_inter

        try:
            dist_mat = distance_array(pos, WC_inter, box=boxdim) # should account for the wrapping. 
        except:
            dist_mat = distance_array(pos, np.array(WC_inter), box=boxdim) # should account for the wrapping. 
        proxim = np.min(dist_mat,axis=1) # obtain min for each row/atom. 
        loc = [(np.where(dist_mat[i] == proxim[i])[0][0]) for i in range(len(proxim))] # obtain splined interface coordinate closest to the ox positions. 
        
        mag = []
        vect_list = []

        for i in range(len(pos)):
            
            z_unit  = [0,0,1]
            vect = distances.minimize_vectors(WC_inter[loc[i]]-pos[i],box=boxdim)

            scal_proj = np.dot(z_unit,vect)
            mag.append(scal_proj)
            vect_list.append(vect)
                
        mag_prox = [0]*len(mag)
        for i in range(len(mag)):
            if mag[i] < 0:
                mag_prox[i] = proxim[i]
            else:
                mag_prox[i] = -proxim[i]
            
        if result == 'mag':
            return mag_prox # distance (magnitude denotes inside or outside slab)
        elif result == 'vect':
            return np.array(vect_list)
        elif result == 'both':
            return (mag_prox,np.array(vect_list))

# center = boxdim[:3]/2 
# init = distances.apply_PBC((WC_spline[loc[i]] - pos[i]) + center,box=boxdim)
# vect = init - center

# surf_wrap_c = distances.apply_PBC(WC_spline[loc[i]]+center,box=boxdim)
# pos_wrap_c  = distances.apply_PBC(pos[i]+center,box=boxdim)
# vect = surf_wrap_c - pos_wrap_c


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
        
