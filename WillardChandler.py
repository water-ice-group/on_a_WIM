# script for calculating the Willard-Chandler interface
# https://pubs.acs.org/doi/pdf/10.1021/jp909219k


import numpy as np
from interface import WC_Interface
from utilities import AtomPos
from density import Density
from density import dens_plot
from orientation import Orientation
from orientation import oriPlot
from hbondz import Hbondz
from hbondz import hbondPlot

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm


class WillardChandler:
    
    '''Module for generating a Willard-Chandler interface and using this
    interface to calculate properties such as density and orientation.'''

    def __init__(self, universe, endstep=None):    
        self._u = universe
        self._end = endstep
        

    def generate(self,grid=400):
        '''Generate the WC interface.'''
        
        print()
        print('---------------------')
        print(' Loading trajectory  ')
        print('---------------------')
        print()

        pos = AtomPos(self._u,self._end)
        self._opos,self._h1pos,self._h2pos,self._cpos,self._ocpos = pos.positions()
        
        inter = WC_Interface(self._u,grid)
        opos_traj = self._opos
        num_cores = multiprocessing.cpu_count()
        
        print()
        print(f'Number of cores: {num_cores}')
        print()

        grid = inter.grid_spacing()

        print('Generating frames ...')
        result = Parallel(n_jobs=num_cores)(delayed(inter.criteria)(i,grid) for i in tqdm(opos_traj))
        self._WC = result
        self.inter = inter

        print('Done')
        print()

        return self._WC

    
    # save coordinates for visualisation
    def visualise(self):  
        self.inter.save_coords(self._WC)
        
        
        
    ##########################################################################
    ############################# Measurements ###############################
    ##########################################################################
        
    # Density
    def Density_run(self,atom_type,bins=200,lower=-20,upper=10):
        '''Obtain the density of species relative to the WC interface.'''

        self._dens_lower = lower
        self._dens_upper = upper

        dens = Density(self._u)
    
        if atom_type == 'OW':
            traj = self._opos
        elif atom_type == 'C':
            traj = self._cpos

        print()
        print(f'Obtaining {atom_type} density.')

        num_cores = multiprocessing.cpu_count()
        print('Calculating density profile ...')
        result = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(self._WC[i],traj[i]) for i in tqdm(range(len(traj))))
        self._dens_result = result
        print('Generating histogram(s)')

        hist_input = np.concatenate(result).ravel()
        if atom_type == 'OW':
            density,x_range = np.histogram(hist_input,bins=bins,range=[lower,upper])
            self._waterno = density
            water_dens = 18.01528
            N_A = 6.022*10**23
            xy = self._u.dimensions[0]
            hist_range = upper - lower
            result_hist = [(i*water_dens)/( (N_A) * (xy*xy*(hist_range/bins) * 10**(-30)) * (2*len(traj)) * 10**6) for i in density]
            
        elif atom_type == 'C':
            density,x_range = np.histogram(hist_input,bins=bins,range=[lower,upper],density=True)
            result_hist = density
        
        save_dat = np.array([x_range[:-1],result_hist])
        save_dat = save_dat.transpose()
        np.savetxt('./outputs/' + atom_type + '_dens.dat',save_dat)
        print('Done')
        print()
        return (result_hist,x_range)

    
    def Density_plot(self,data_Oxygen,data_Carbon=None):
        dens_plot(data_Oxygen,data_Carbon,self._dens_lower,self._dens_upper)

    # ------------------------------------------------------------------------

    # Orientation
    def Orientation_run(self,bins=200,lower=-20,upper=0):
        '''Obtain the orientation of the species relative to the WC interface.'''
        
        cosTheta_list = []
        dist_list = []
        ori = Orientation(self._u)        
        
        print()
        print(f'Obtaining orientations.')

        num_cores = multiprocessing.cpu_count()
        print('Calculating orientation profile ...')
        result = Parallel(n_jobs=num_cores)(delayed(ori._getCosTheta)(self._opos[i],self._h1pos[i],self._h2pos[i],self._WC[i]) for i in tqdm(range(len(self._opos))))
        
        dist = [i[0] for i in result]
        theta = [i[1] for i in result]
        
        print('Generating histogram(s)')

        dist_array = np.concatenate(dist).ravel()
        Theta_array = np.concatenate(theta).ravel()
        
        result = ori._getHistogram(dist_array,
                                   Theta_array,
                                   bins=bins,hist_range=[lower,upper])
        

        np.savetxt('./outputs/orientation.dat',result)
        self._ori = result
        print('Done.')
        print()
        return result
    
    def Orientation_plot(self):
        oriPlot(self._ori)

    # ------------------------------------------------------------------------

    # Hydrogen bond counting
    def Hbonds_run(self,bins=75,lower=-15,upper=0):
        '''Obtain the HBond profile mapping the average count with distance 
        from the interface.'''
        
        self._hbond_lower = lower
        self._hbond_upper = upper

        hbonds = []
        counter = Hbondz(self._u)
        

        print()
        print(f'Obtaining Hbonds.')

        num_cores = multiprocessing.cpu_count()
        print('Calculating H-Bond profile ...')
        result = Parallel(n_jobs=num_cores)(delayed(counter.count)(self._opos[i],self._h1pos[i],self._h2pos[i],self._WC[i],lower,upper,bins) for i in tqdm(range(len(self._opos))))
        print('Generating histogram(s)')

        hist_list = [0] * len(result[0])
        for i in range(len(result[0])):
            for j in range(len(result)):
                hist_list[i] += result[j][i]

        hist_list = np.array(hist_list)
        
        hist_adj = hist_list/len(self._WC)
        print('Done.')
        print()

        xrange = np.linspace(lower,upper,bins)

        output = np.array([ [xrange[i],hist_adj[i]] for i in range(len(xrange)-1)])
        np.savetxt('./outputs/hbonds.dat',output)
        self._hbonds = (hist_adj,xrange)
        return (hist_adj,xrange)
    
    def HBondz_plot(self):
        hbondPlot(self._hbonds,self._hbond_lower,self._hbond_upper)
        






           

           

        

    
