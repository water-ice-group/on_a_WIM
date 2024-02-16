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
from itim import ITIM

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm


class WillardChandler:
    
    '''Module for generating a Willard-Chandler interface and using this
    interface to calculate properties such as density and orientation.'''

    def __init__(self, universe, lower_z, upper_z, startstep=None,endstep=None):    
        self._u = universe
        self._start = startstep
        self._end = endstep
        self._lz = lower_z
        self._uz = upper_z

    def generate(self,grid=400,new_inter=True):
        '''Generate the WC interface.'''
        
        print()
        print('---------------------')
        print(' Loading trajectory  ')
        print('---------------------')
        print()

        # create position object and extract positions (unwrapped)
        pos = AtomPos(self._u,self._start,self._end)
        self._opos,self._h1pos,self._h2pos,self._cpos,self._ocpos1,self._ocpos2,self._boxdim = pos.prepare()
        opos_traj = self._opos # wrapped oxygen coordinates to form the main coords to generatin the interface. 

        # create interface object
        inter = WC_Interface(self._u,grid,self._lz,self._uz)

        if new_inter==True: # generate new interfacial surface

            # no. of cores
            num_cores = int(multiprocessing.cpu_count()/2)
            print()
            print(f'Number of cores: {num_cores}')
            print()

            # run the parallelised jobs
            print('Generating frames ...')
            grid = inter.grid_spacing()
            result = Parallel(n_jobs=num_cores)(delayed(inter.criteria)(opos_traj[i],grid,self._boxdim[i]) for i in tqdm(range(len(opos_traj))))
            self._WC = result

            print('Done')
            print()
        
        elif new_inter==False: # load existing surface
            result = self.load(inter)
            self._WC = result

        self.inter = inter
        return self._WC
    
    # save coordinates for visualisation
    def save(self):  
        self.inter.save_coords(self._WC)

    def load(self,inter):
        wc_univ = inter.load_coords()
        loaded_coords = []
        sel = wc_univ.select_atoms('all')
        for ts in wc_univ.trajectory:
            pos = sel.positions
            loaded_coords.append(pos)

        return loaded_coords
        
        
        
    ##########################################################################
    ############################# Measurements ###############################
    ##########################################################################
        
    # Density
    def Density_run(self,atom_type,bins=400,lower=-10,upper=10):


        
        """Computes the density of molecules relative to the water-carbon interface.

        Args:
            atom_type (str): Type of molecule ('OW' for water oxygen or 'C' for carbon).
            bins (int): Number of bins for histogram.
            lower (float): Lower bound for histogram range.
            upper (float): Upper bound for histogram range.
        
        Returns:
            tuple: Tuple containing density histogram and corresponding bin edges.

        Raises:
            ValueError: If the specified atom type is not supported."""
        


        dens = Density(self._u)
        self._dens_lower = lower
        self._dens_upper = upper

    
        if atom_type == 'OW':
            traj = self._opos
        elif atom_type == 'C':
            traj = self._cpos

        print()
        print(f'Obtaining {atom_type} density.')

        num_cores = multiprocessing.cpu_count()
        print('Calculating density profile ...')
        result = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(self._WC[i],traj[i],boxdim=self._boxdim[i],upper=self._uz) for i in tqdm(range(len(traj)))) # parse through frames
        self._dens_result = result
        print('Generating histogram(s)')

        hist_input = np.concatenate(result).ravel()
        
        density,x_range = np.histogram(hist_input,bins=bins,range=[lower,upper])
        N_A = 6.022*10**23
        xy = self._u.dimensions[0]
        hist_range = upper - lower
        if atom_type == 'OW':
            mol_dens = 18.01528
            result_hist = [(i*mol_dens)/( (N_A) * (xy*xy*(hist_range/bins) * 10**(-30)) * (len(traj)) * 10**6) for i in density]
        elif atom_type == 'C':
            mol_dens = 44.0095 
            result_hist = [(i*mol_dens)/( (N_A) * (xy*xy*(hist_range/bins) * 10**(-30)) * (len(traj)) * 10**6) for i in density] 

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
    def Orientation_run(self,atomtype='water',histtype='time',bins=400,lower=-10,upper=10,vect='z'):


        """Computes orientations of near-interface molecules based on specified atom type and histogram type.

        Args:
            atomtype (str): Type of atom ('water' or 'carbon').
            histtype (str): Type of histogram ('time' or 'heatmap').
            lower (float): Lower bound for histogram range.
            upper (float): Upper bound for histogram range.
            bins (int): Number of bins for histogram.
            vect (str): Vector with which to compute orientation ('z' axis or 'WC' vector).
        
        Returns:
            ndarray or tuple: Depending on the histtype, returns either the orientation histogram (time) or tuple containing X, Y, and the heatmap histogram (heatmap).

        Raises:
            ValueError: If the specified atom type is not supported."""
        

        ori = Orientation(self._u)  
        self._ori_lower = lower
        self._ori_upper = upper      
        
        print()
        print(f'Obtaining orientations.')

        num_cores = multiprocessing.cpu_count()
        print('Calculating orientation profile ...')

        if atomtype == 'water':
            result = Parallel(n_jobs=num_cores)(delayed(ori._getCosTheta)(self._opos[i],self._h1pos[i],self._h2pos[i],self._WC[i],self._uz,self._boxdim[i],vect) for i in tqdm(range(len(self._opos))))
            if vect == 'WC':
                lower = lower
                upper = 0
        elif atomtype == 'carbon':
            result = Parallel(n_jobs=num_cores)(delayed(ori._getCosTheta_Carbon)(self._cpos[i],self._ocpos1[i],self._ocpos2[i],self._WC[i],self._uz,self._boxdim[i],vect) for i in tqdm(range(len(self._cpos))))
            if vect == 'WC':
                lower = 0
                upper = upper
        else:
            print('Specify atom type.')
    
        dist = [i[0] for i in result]
        theta = [i[1] for i in result]
        
        print('Generating histogram(s)')

        print(len(dist))
        print(len(theta))
        dist_array = np.concatenate(dist).ravel()
        Theta_array = np.concatenate(theta).ravel()

        print(len(dist_array))
        print(len(Theta_array))
        



        if histtype=='time':
            result = ori._getHistogram(dist_array,
                                    Theta_array,
                                    bins=bins,hist_range=[lower,upper])
            np.savetxt(f'./outputs/orientation_{atomtype}.dat',result)
            self._ori = result
            print('Done.')
            print()
            return result

        elif histtype=='heatmap':
            hist,x_edges,y_edges = ori._getHeatMap(dist_array,
                                     Theta_array,
                                     bins=bins,hist_range=[lower,upper])
            H = hist.T

            X, Y = np.meshgrid(x_edges[:-1] + 0.5 * (x_edges[1] - x_edges[0]), 
                   y_edges[:-1] + 0.5 * (y_edges[1] - y_edges[0]))

            np.savetxt(f'./outputs/heatmap_X_{atomtype}.dat',X)
            np.savetxt(f'./outputs/heatmap_Y_{atomtype}.dat',Y)
            np.savetxt(f'./outputs/heatmap_hist_{atomtype}.dat',H)
            return (X,Y,H)



    
    
    def Orientation_plot(self,data_Oxygen,data_Carbon=None):
        oriPlot(data_Oxygen,data_Carbon,self._ori_lower,self._ori_upper)

    # ------------------------------------------------------------------------

    # Hydrogen bond counting
    def Hbonds_run(self,bins=75,lower=-15,upper=0):


        """Computes hydrogen bond profile mapping the average count with distance from the interface.

        Args:
            bins (int): Number of bins for histogram.
            lower (float): Lower bound for histogram range.
            upper (float): Upper bound for histogram range.
        
        Returns:
            tuple: Tuple containing histograms of donor and acceptor hydrogen bonds along with their corresponding bin edges.

        Raises:
            None"""
        

        counter = Hbondz(self._u,self._uz)
        self._hbond_lower = lower
        self._hbond_upper = upper

        print()
        print(f'Obtaining Hbonds.')
        hist_don,don_range,hist_acc,acc_range = counter.hbond_analysis(self._WC,lower,upper,self._start,self._end,self._boxdim,bins)

        self._don = hist_don
        self._donx = don_range
        self._acc = hist_acc
        self._accx = acc_range

        return ((hist_don,don_range),(hist_acc,acc_range))
    
    def HBondz_plot(self):
        hbondPlot(self._don,self._donx,self._acc,self._accx,self._hbond_lower,self._hbond_upper)
        

    # cluster analysis 

    def Clusters(self):

        itim = ITIM(self._u,self._start,self._end)
        inter_o,inter_h1,inter_h2 = itim.prepare()

        num_cores = multiprocessing.cpu_count()
        result = Parallel(n_jobs=num_cores)(delayed(itim.cluster_analysis)(inter_o[i],inter_h1[i],inter_h2[i],self._cpos[i],self._ocpos1,self._ocpos2,self._boxdim[i]) for i in tqdm(range(len(inter_o))))

        result,edges = np.histogram(result,bins=250)
        edges = 0.5 * (edges[1:] + edges[:-1])
        
        hist = np.array([edges,result])
        hist = hist.transpose()
        return (hist)
