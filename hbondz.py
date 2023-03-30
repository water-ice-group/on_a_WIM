# define functions for calculating the number of HBonds per distance. 


import numpy as np
from density import Density
import math as m
import matplotlib.pyplot as plt
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import calc_angles
from scipy import stats


class Hbondz:
    '''calculate the number of H bonds in system as a function of 
    z-coordinate.'''
    
    def __init__(self, universe, **kwargs):

        self._u = universe


        
        
    def count(self, opos, h1pos, h2pos, wc,lower,upper,bins):
        '''For each timeframe, determine the HBond count.'''
        
        # need to determine whether this is an instance of double counting or not. 
        opos_dist = distance_array(opos,opos,box=self._u.dimensions) 
        crit_1a,crit_1b = np.where( (opos_dist>0) & (opos_dist <= 3.3) )
        
        angle_array = calc_angles(opos[crit_1a],h1pos[crit_1a],opos[crit_1b],box=self._u.dimensions)
        angle_array = np.rad2deg(angle_array)
        
        crit2 = np.where(angle_array > 150.0)
        ox_idx = crit_1a[crit2]
        acc_idx = crit_1b[crit2]
        list_1 = [opos[i] for i in ox_idx]
        list_2 = [opos[i] for i in acc_idx]

        
        angle_array = calc_angles(opos[crit_1a],h2pos[crit_1a],opos[crit_1b],box=self._u.dimensions)
        angle_array = np.rad2deg(angle_array)

        crit2 = np.where(angle_array > 150.0)
        ox_idx = crit_1a[crit2]
        acc_idx = crit_1b[crit2]
        list_3 = [opos[i] for i in ox_idx]
        list_4 = [opos[i] for i in acc_idx]

        O_hbond_list = list_1 + list_2 + list_3 + list_4

        dens = Density(self._u)
        dist = dens.proximity(wc,O_hbond_list,'mag')
        dist_norm = dens.proximity(wc,opos,'mag')
        hist,xrange = np.histogram(dist,bins=bins,range=[lower,upper])
        hist_norm,xrange = np.histogram(dist_norm,bins=bins,range=[lower,upper])

        hist_final = []
        for i in range(len(hist)):
            if (hist_norm[i] != 0):
                result = hist[i]/hist_norm[i]
                hist_final.append(result)
            else:
                result = hist[i]
                hist_final.append(result)

        return hist_final


    

def hbondPlot(hist,lower,upper):
    fig, ax = plt.subplots()
    ax.plot(hist[1],hist[0],'.-')
    ax.set_xlabel('Distance / $\mathrm{\AA}$')
    ax.set_ylabel('HBond count')
    ax.set_xlim(lower,upper)
    ax.set_ylim(0,4)
    plt.savefig('./outputs/hbond_profile.pdf',dpi=400,bbox_inches='tight',facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
