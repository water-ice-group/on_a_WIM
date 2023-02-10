# define functions for calculating the number of HBonds per distance. 


import numpy as np
from density import Density
import math as m
import matplotlib.pyplot as plt
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import calc_angles



class Hbondz:
    '''calculate the number of H bonds in system as a function of 
    z-coordinate.'''
    
    def __init__(self, universe, **kwargs):

        self._u = universe


        
        
    def count(self, opos, h1pos, h2pos, wc, wrap_opos):
        '''For each timeframe, determine the HBond count.'''
        
        opos_dist = distance_array(opos,opos,box=self._u.dimensions)
        crit_1a,crit_1b = np.where( (opos_dist>0) & (opos_dist <= 3.0) )
        
        angle_array = calc_angles(opos[crit_1a],h1pos[crit_1a],opos[crit_1b],box=self._u.dimensions)
        angle_array = np.rad2deg(angle_array)
        
        crit2 = np.where(angle_array > 150.0)
        ox_idx = crit_1a[crit2]
        list_1 = [wrap_opos[i] for i in ox_idx]

        
        angle_array = calc_angles(opos[crit_1a],h2pos[crit_1a],opos[crit_1b],box=self._u.dimensions)
        angle_array = np.rad2deg(angle_array)

        crit2 = np.where(angle_array > 150.0)
        ox_idx = crit_1a[crit2]
        list_2 = [wrap_opos[i] for i in ox_idx]
        
        O_hbond_list = list_1 + list_2

        dens = Density(self._u)
        dist = dens.proximity(wc,O_hbond_list,'mag')
        
        return dist
    

def hbondPlot(hist):
    fig, ax = plt.subplots()
    ax.plot(hist[1][:-1],hist[0],'.-')
    ax.set_xlabel('Distance / $\mathrm{\AA}$')
    ax.set_ylabel('HBond count')
    ax.set_xlim(-15,0)
    #ax.set_ylim(0,4)
    plt.savefig('./outputs/hbond_profile.pdf',dpi=400,bbox_inches='tight',facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()

            
        