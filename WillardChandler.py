# script for calculating the Willard-Chandler interface
# https://pubs.acs.org/doi/pdf/10.1021/jp909219k


# Standard library imports
import multiprocessing

# Third-party imports
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import stats

# Local application imports
from interface import WC_Interface
from utilities import AtomPos
from density import Density, dens_plot
from orientation import Orientation, oriPlot
from hbondz import Hbondz, hbondPlot
from rdf import RDF




class WillardChandler:
    
    """Module for calculating the WC interface and computing properties relative to interface.

    Args:
        universe (obj):         Load MDAnalysis universe for system. 
        lower_z (float):        Upper bound for histogram range.
        upper_z (float):        Upper bound for histogram range.
        startstep (int):        Number of bins for histogram.
        endstep (int):          Lower bound for histogram range.
    
    Returns:
        tuple: Tuple containing density histogram and corresponding bin edges.

    Raises:
        ValueError: If the specified atom type is not supported."""

    def __init__(self, universe, lower_z, upper_z, startstep=None,endstep=None):    
        self._u = universe
        self._start = startstep
        self._end = endstep
        self._lz = lower_z
        self._uz = upper_z



    ##########################################################################
    ################################ Surface #################################
    ##########################################################################




    def generate(self,grid=400,new_inter=True,org=True):

        '''Generate the WC interface.'''
        
        print()
        print('---------------------')
        print(' Loading trajectory  ')
        print('---------------------')
        print()

        self._grid = grid

        if org==True:       # organise waters by closest hydrogens. Returns list organised by molecule. 
            pos = AtomPos(self._u,self._start,self._end)
            self._opos,self._h1pos,self._h2pos,self._cpos,self._ocpos1,self._ocpos2,self._boxdim = pos.prepare()
            opos_traj = self._opos
        elif org==False:    # no organisation of waters. Extract list of oxygens and hydrogens. Needed for hydronium/hydroxide systems. 
            pos = AtomPos(self._u,self._start,self._end)
            self._opos,self._hpos,self._h3opos,self._boxdim = pos.prepare_unorg()
            opos_traj = self._opos


        inter = WC_Interface(self._u,grid,self._lz,self._uz)
        

        if new_inter==True: # generate new interfacial surface
            num_cores = multiprocessing.cpu_count()//2
            print()
            print(f'Number of cores: {num_cores}')
            print()
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

    def save(self):      # save coordinates for visualisation
        self.inter.save_coords(self._WC)

    def load(self,inter): # load existing coordinates for surface
        wc_univ = inter.load_coords()
        loaded_coords = []
        sel = wc_univ.select_atoms('all')
        for ts in wc_univ.trajectory:
            pos = sel.positions
            loaded_coords.append(pos)
        return loaded_coords
        
    def surface_stats(self,bins=100): # states on the deformation of the interface

        inter = WC_Interface(self._u,self._grid,self._lz,self._uz)

        num_cores = multiprocessing.cpu_count()
        print('Getting surface stats ...')
        result = Parallel(n_jobs=num_cores)(delayed(inter.dist_surf_deform)(self._WC[i]) for i in tqdm(range(len(self._WC)))) # parse through frames
        print('Generating histogram(s)')
        hist_input = np.concatenate(result).ravel()
        density,x_range = np.histogram(hist_input,bins=bins,density=True)

        save_dat = np.array([x_range[:-1],density])
        save_dat = save_dat.transpose()
        np.savetxt('outputs/surface_stats.dat',save_dat)
        print('Done')
        print()
        return (density,x_range[:-1])

        











    ##########################################################################
    ################################ Density #################################
    ##########################################################################
        
    # Density
    def Density_run(self,atom_type,bins=400,lower=-10,upper=10):


        """Computes the density of molecules relative to the water-carbon interface.

        Args:
            atom_type (str):        Type of molecule ('OW' for water oxygen or 'C' for carbon).
            bins (int):             Number of bins for histogram.
            lower (float):          Lower bound for histogram range.
            upper (float):          Upper bound for histogram range.
        
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
        elif atom_type == 'H3O':
            traj = self._h3opos

        print()
        print(f'Obtaining {atom_type} density.')
        num_cores = multiprocessing.cpu_count()
        print('Calculating density profile ...')
        result = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(self._WC[i],traj[i],boxdim=self._boxdim[i],upper=self._uz,cutoff=False) for i in tqdm(range(len(traj)))) # parse through frames
        self._dens_result = result
        print('Generating histogram(s)')

        hist_input = np.concatenate(result).ravel()
        
        density,bin_range = np.histogram(hist_input,bins=bins,range=[lower,upper])

        x_range = [(bin_range[i]+bin_range[i+1])/2 for i in range(len(bin_range)-1)]

        N_A = 6.022*10**23
        xy = self._u.dimensions[0]
        hist_range = upper - lower
        if atom_type == 'OW':
            mol_dens = 18.01528
        elif atom_type == 'C':
            mol_dens = 44.0095 # need to alter this 
        elif atom_type == 'H3O':
            mol_dens = 19.023
        result_hist = [(i*mol_dens)/( 2 * (N_A) * (xy*xy*(hist_range/bins) * 10**(-30)) * (len(traj)) * 10**6) for i in density]



        save_dat = np.array([x_range,result_hist])
        save_dat = save_dat.transpose()
        np.savetxt('./outputs/' + atom_type + '_dens.dat',save_dat)
        print('Done')
        print()
        return (result_hist,x_range)
    

    def nrg_from_dens(self,species='C'): # extract free energy from density profile
        if species == 'C':
            fin = np.loadtxt('./outputs/C_dens.dat')
        elif species == 'H3O':
            fin = np.loadtxt('./outputs/H3O_dens.dat')
        dist = fin[:,0]
        dens = fin[:,1]
        R = 8.3145
        k = 1.380649e-23 
        T = 300
        const = np.sum(dens)
        nrg = [-0.000239006*R*T*np.log(i/const) for i in dens]
        min_val = min(nrg)
        output = [i-min_val for i in nrg]

        save_dat = np.array([dist,output])
        save_dat = save_dat.transpose()
        np.savetxt('./outputs/free_energy.dat',save_dat)

        return (dist,output)

    
    def Density_plot(self,data_Oxygen,data_Carbon=None):
        dens_plot(data_Oxygen,data_Carbon,self._dens_lower,self._dens_upper)
















    ##########################################################################
    ############################## Orientation ###############################
    ##########################################################################
        

    # orientation
        
    def Orientation_run(self,atomtype='water',histtype='time',bins=400,lower=-10,upper=10,vect='WC',prop='dipole'):


        """Computes orientations of near-interface molecules based on specified atom type and histogram type.

        Args:
            atomtype (str):         Type of atom ('water' or 'carbon').
            histtype (str):         Type of histogram ('time' or 'heatmap').
            lower (float):          Lower bound for histogram range.
            upper (float):          Upper bound for histogram range.
            bins (int):             Number of bins for histogram.
            vect (str):             Vector with which to compute orientation ('z' axis or 'WC' vector).
        
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
            if vect == 'WC':
                result = Parallel(n_jobs=num_cores)(delayed(ori._getCosTheta)(self._opos[i],self._h1pos[i],self._h2pos[i],self._WC[i],self._boxdim[i]) for i in tqdm(range(len(self._opos))))
                lower = lower
                upper = upper
            elif vect == 'z':
                result = Parallel(n_jobs=num_cores)(delayed(ori._getCosTheta_z)(self._opos[i],self._h1pos[i],self._h2pos[i],self._boxdim[i]) for i in tqdm(range(len(self._opos))))

        elif atomtype == 'carbon':
            result = Parallel(n_jobs=num_cores)(delayed(ori._getCosTheta_Carbon)(self._cpos[i],self._ocpos1[i],self._ocpos2[i],self._WC[i],self._boxdim[i],prop) for i in tqdm(range(len(self._cpos))))
            if vect == 'WC':
                lower = 0
                upper = upper
        else:
            print('Specify atom type.')
        dist = [i[0] for i in result]
        theta = [i[1] for i in result]
        
        print('Generating histogram(s)')
        dist_array = np.concatenate(dist).ravel()
        Theta_array = np.concatenate(theta).ravel()
        print(dist[0])
        print(theta[0])
        
        if histtype=='time':
            print(len(dist_array))
            print(len(Theta_array))
            result = ori._getHistogram(dist_array,
                                    Theta_array,
                                    bins=bins,hist_range=[lower,upper])
            x_out = result[:,0]
            result_hist = result[:,1]
            
            save_dat = np.array([x_out,result_hist])
            save_dat = save_dat.transpose()
            np.savetxt(f'./outputs/orientation_{atomtype}.dat',save_dat)
            print('Done.')
            print()
            return save_dat

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
            print('Done.')
            print()
            return (X,Y,H)


    def Orientation_plot(self,data_Oxygen,data_Carbon=None):
        oriPlot(data_Oxygen,data_Carbon,self._ori_lower,self._ori_upper)















    ##########################################################################
    ################################ Hbonding ################################
    ##########################################################################


    # Hydrogen bond counting
    def Hbonds_run(self,bins=100,lower=-15,upper=0):
        

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
        

















    ##########################################################################
    ########################## Solvation character ###########################
    ##########################################################################

    '''Analyse the local solvation environements of the various carbon species
    under both interfacial and bulk conditions.'''

    def surf_RDF(self,bins=75,hist_range=[2,8]):

        rdf = RDF(self._u)

        print()
        print('Calculating RDFs ...')
        num_cores = multiprocessing.cpu_count()
        result = Parallel(n_jobs=num_cores)(delayed(rdf.get_rdf)(self._cpos[i],self._opos[i],self._WC[i],self._boxdim[i],dr=0.08,crit_dens=0.032) for i in tqdm(range(len(self._cpos))))

        dist = [i[0] for i in result]
        out = [i[1] for i in result]
        dist_array = np.concatenate(dist).ravel()
        rdf_array = np.concatenate(out).ravel()

        print('Generating histogram(s)')
        means, edges, binnumber = stats.binned_statistic(dist_array[:].flatten(),
                                                         rdf_array[:].flatten(),
                                                         statistic='mean', bins=bins,
                                                         range=hist_range)
        
        x_out = 0.5 * (edges[1:] + edges[:-1])
        result_hist = means
        
        save_dat = np.array([x_out,result_hist])
        save_dat = save_dat.transpose()
        np.savetxt(f'./outputs/surf_RDF.dat',save_dat)
        print('Done.')
        print()
        
        return save_dat
        




    

