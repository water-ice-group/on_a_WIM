# script for calculating the Willard-Chandler interface
# https://pubs.acs.org/doi/pdf/10.1021/jp909219k


# Standard library imports
import multiprocessing

# Third-party imports
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Local application imports
from interface import WC_Interface
from utilities import AtomPos
from density import Density, dens_plot
from orientation import Orientation, oriPlot
from hbondz import Hbondz, hbondPlot
from itim import monolayer,monolayer_properties




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
        elif org==False:    # no organisation of waters. Extract list of oxygens and hydrogens. # Needed for hydronium/hydroxide systems. 
            pos = AtomPos(self._u,self._start,self._end)
            self._opos,self._hpos,self._boxdim = pos.prepare_unorg()
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
            result_hist = [(i*mol_dens)/( (N_A) * (xy*xy*(hist_range/bins) * 10**(-30)) * (len(traj)) * 10**6) for i in density]
        elif atom_type == 'C':
            mol_dens = 44.0095 
            result_hist = [(i*mol_dens)/( (N_A) * (xy*xy*(hist_range/bins) * 10**(-30)) * (len(traj)) * 10**6) for i in density] 


        save_dat = np.array([x_range,result_hist])
        save_dat = save_dat.transpose()
        np.savetxt('./outputs/' + atom_type + '_dens.dat',save_dat)
        print('Done')
        print()
        return (result_hist,x_range)
    

    def nrg_from_dens(self): # extract free energy from density profile
        fin = np.loadtxt('./outputs/C_dens.dat')
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
            result = Parallel(n_jobs=num_cores)(delayed(ori._getCosTheta)(self._opos[i],self._h1pos[i],self._h2pos[i],self._WC[i],self._boxdim[i],vect) for i in tqdm(range(len(self._opos))))
            if vect == 'WC':
                lower = lower
                upper = 0
        elif atomtype == 'carbon':
            result = Parallel(n_jobs=num_cores)(delayed(ori._getCosTheta_Carbon)(self._cpos[i],self._ocpos1[i],self._ocpos2[i],self._WC[i],self._boxdim[i],vect,prop) for i in tqdm(range(len(self._cpos))))
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
        
        if histtype=='time':
            result = ori._getHistogram(dist_array,
                                    Theta_array,
                                    bins=bins,hist_range=[lower,upper])
            x_out = result[:,0]
            result_hist = result[:,1]

            # remove points within surface cutoff error
            # -------------------------------------------------------------
            no_surf_points = 20
            wc_width = 20/no_surf_points # min distance between points on isosurface
            dump = []
            for i in range(len(x_out)):
                if (x_out[i] > -wc_width/2) and (x_out[i] < wc_width/2):
                    dump.append(i)
            x_out = np.delete(x_out, dump)
            result_hist = np.delete(result_hist, dump)
            # -------------------------------------------------------------
            
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
    ################################ ITIM analysis ###########################
    ##########################################################################

    '''Code for analysing the CO2 monolayer within a certain cutoff. 
    Perform this analysis last. Can induce some errors in the parallelisation
    code of other functions when used.'''

    def surf_co2(self,property='rdf',min_cutoff=0,max_cutoff=4,bins=100,norm=True):

        '''Calculate properties of CO2 molecules at the interface.

        min_cutoff: minimum cutoff for locatinng CO2. 
        max_cutoff: maximum cutoff for locating CO2.'''

        itim = monolayer(self._u,self._start,self._end)
        cluster_prop = monolayer_properties(self._u)

        num_cores = int(multiprocessing.cpu_count())
        if property=='rdf':
            result = Parallel(n_jobs=num_cores,backend='threading')(delayed(cluster_prop.surf_co2)(self._WC[i],self._cpos[i],self._boxdim[i],min_cutoff,max_cutoff) for i in tqdm(range(len(self._cpos))))
            hist_input = np.concatenate(result).ravel()
            density,x_range = np.histogram(hist_input,bins=bins,
                            density=norm,range=(1,10))
            
            # Convert to lateral distribution function
            output = [density[i]/(2*np.pi*x_range[i]) for i in range(len(density))] 

        elif property=='CO_angle':
            result = Parallel(n_jobs=num_cores,backend='threading')(delayed(cluster_prop.co2_bond_angles_surf)(self._WC[i],self._cpos[i],self._ocpos1[i],self._ocpos2[i],self._boxdim[i],min_cutoff,max_cutoff) for i in tqdm(range(len(self._cpos))))
            hist_input = np.concatenate(result).ravel()
            density,x_range = np.histogram(hist_input,bins=bins,
                            density=norm,range=(5,175))
            
            # Divide by isotropic distribution 
            output = [density[i]/( 0.5*np.sin((x_range[i]*(np.pi / 180))) ) for i in range(len(x_range[:-1]))]
                
        save_dat = np.array([x_range[:-1],output])
        save_dat = save_dat.transpose()
        np.savetxt(f'./outputs/surf_co2_{property}_{max_cutoff}.dat',save_dat)
        return (output,x_range[:-1])


    

