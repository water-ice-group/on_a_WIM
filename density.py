# script for calculating the Willard-Chandler interface
# https://pubs.acs.org/doi/pdf/10.1021/jp909219k


import numpy as np
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.analysis.distances import dist
from MDAnalysis.transformations.wrap import wrap
from scipy import interpolate
from interface import WC_Interface
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


class Density:
    '''Calculate the density of species relative to the loaded WC interface.'''
    
    def __init__(self,universe):
        
        self._u = universe
        
    

    def proximity(self,WC_inter,pos,result='mag'):
        '''Obtain the proximities of each particular molecule to the WC interface.'''
        
        pos = np.array(pos)
        WC_spline = np.array(WC_Interface(self._u).spline(WC_inter))
        
        dist_mat = distance_array(pos, WC_spline, box=self._u.dimensions)
        proxim = np.min(dist_mat,axis=1) # obtain min for each row/atom. 
        loc = [(np.where(dist_mat[i] == proxim[i])[0][0]) for i in range(len(proxim))]
        
        mag = []
        vect_list = []
        surf_div = int(len(WC_spline)/2)
        for i in range(len(pos)):
            if loc[i] < surf_div: # upper surface first
                z_unit  = [0,0,-1]
                #wrap_pos = wrap([pos[i]])
                vect = pos[i] - WC_spline[loc[i]]
                scal_proj = np.dot(vect,z_unit)
                mag.append(scal_proj)
                vect_list.append(vect)

            else:
                z_unit  = [0,0,1]
                #wrap_pos = wrap([pos[i]])
                vect = pos[i] - WC_spline[loc[i]]
                scal_proj = np.dot(vect,z_unit)
                mag.append(scal_proj)
                vect_list.append(vect)

                
        mag_prox = [0]*len(mag)
        for i in range(len(mag)):
            if mag[i] < 0:
                mag_prox[i] = - proxim[i]
            else:
                mag_prox[i] = proxim[i]
            
        if result == 'mag':
            return mag_prox # distance (magnitude denotes inside or outside slab)
        elif result == 'vect':
            return np.array(vect_list)
        elif result == 'both':
            return (mag_prox,np.array(vect_list))
        




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
                label = r'$\rho \mathrm{(CO_2)}$ (Norm)')
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
        