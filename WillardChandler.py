# script for calculating the Willard-Chandler interface
# https://pubs.aip.org/aip/jcp/article/161/8/084711/3309975


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
from hbondz import Hbondz, CN, hbondPlot
from rdf import RDF


N_A = 6.022*10**23
R = 8.3145
k = 1.380649e-23 
T = 300

MOLECULAR_MASSES = {
    'OW': 18.01528,
    'H3O': 19.023,
    'CO2': 44.0095,
    'BiC': 61.016,
    'CA': 62.024,
    'TS': 62.024,
}

class WillardChandler:
    
    """Module for calculating the WC interface and computing properties using the interface as the zero. 
    Currently only supports H2O and CO2, but can be adapted for other carbon species.

    Args:
        universe (obj):         Load MDAnalysis universe for system. 
        startstep (int):        Number of bins for histogram.
        endstep (int):          Lower bound for histogram range.
        lower_z (float):        Upper bound for histogram range.
        upper_z (float):        Upper bound for histogram range.
    """

    def __init__(self, universe, lower_z, upper_z, startstep=None,endstep=None):    
        self._u = universe
        self._start = startstep
        self._end = endstep
        self._lz = lower_z
        self._uz = upper_z
        self.n_cores = multiprocessing.cpu_count() //2   # Initialize here 




    ##########################################################################
    ################################ Surface #################################
    ##########################################################################


    def generate(self,grid=400,new_inter=True,project='standard'):

        '''Generate the WC interface.
        
            Args:
            grid (int): Grid resolution for interface calculation.
            new_inter (bool): If True, generate new interface; if False, load existing.
            project (str): Project type - 'standard', 'metaD', or 'h3o+'.
            
        Returns:
            list: WC interface coordinates for each frame.
            '''
        
        print()
        print('---------------------')
        print(' Loading trajectory  ')
        print('---------------------')
        print()

        self._grid = grid
        pos = AtomPos(self._u,self._start,self._end)

        if project=='standard':       # organise waters by closest hydrogens. Returns list organised by molecule. 
            self._opos,self._h1pos,self._h2pos,self._cpos,self._ocpos1,self._ocpos2,self._boxdim = pos.prepare()
            opos_traj = self._opos
        elif project=='metaD':    # special case where oxygens can exchange between water and carbon molecule. No specific Ox labels.  
            self._opos,self._hpos,self._cpos,self._ocpos,self._boxdim = pos.prepare_undefined()
            opos_traj = self._opos
        elif project=='h3o+':    # no organisation of waters. Extract list of oxygens and hydrogens. Needed for hydronium/hydroxide systems. 
            self._opos,self._hpos,self._h3opos,self._boxdim = pos.prepare_unorg()
            opos_traj = self._opos

#

        inter = WC_Interface(self._u,grid,self._lz,self._uz)
        if new_inter: # create new surface by parsing through frames. If False, load existing surface from file.
            self.n_cores = multiprocessing.cpu_count()//2
            self._WC = self._generate_new_interface(inter, opos_traj)
        
        else: # load existing surface
            self._WC = self._load_interface(inter)

        self.inter = inter
        return self._WC


    def _generate_new_interface(self, inter, opos_traj):
        """Generate new interfacial surface."""
        print(f'Number of cores: {self.n_cores}\n')
        print('Generating frames ...')
        
        grid = inter.grid_spacing()
        result = Parallel(n_jobs=self.n_cores)(
            delayed(inter.criteria)(opos_traj[i], grid, self._boxdim[i])
            for i in tqdm(range(len(opos_traj)))
        )
        
        print('Done\n')
        return result

    def _load_interface(self, inter):
        """Load existing interface coordinates."""
        wc_univ = inter.load_coords()
        loaded_coords = []
        sel = wc_univ.select_atoms('all')
        
        for _ in wc_univ.trajectory:
            loaded_coords.append(sel.positions.copy())
            
        return loaded_coords


    def save(self):      
        """Save interface coordinates for visualisation."""
        self.inter.save_coords(self._WC)
        


    def surface_stats(self,bins=100): 
        """Calculate statistics on the deformation of the interface.
        
        Args:
            bins (int): Number of histogram bins.
            
        Returns:
            tuple: (density, x_range) histogram data.
        """
        inter = WC_Interface(self._u,self._grid,self._lz,self._uz)

        print('Getting surface stats ...')
        result = Parallel(n_jobs=self.n_cores)(
            delayed(inter.dist_surf_deform)(self._WC[i])
            for i in tqdm(range(len(self._WC)))
            ) 
        
        print('Generating histogram(s)')
        hist_input = np.concatenate(result).ravel()
        density,x_range = np.histogram(hist_input,bins=bins,density=True)

        save_dat = np.column_stack([x_range[:-1], density])
        np.savetxt('outputs/surface_stats.dat', save_dat)

        print('Done\n')
        return density,x_range[:-1]

        











    ##########################################################################
    ################################ Density #################################
    ##########################################################################
        
    # Density
    def Density_run(self,atom_type,bins=400,
                    lower=-10,upper=10,
                    select_frames=None,
                    carbon_spec=None,
                    save_raw=None):


        """Computes density profiles relative to the water-carbon interface.

        Args:
            atom_type (str): Type of molecule ('OW', 'C', or 'H3O').
            bins (int): Number of bins for histogram.
            lower (float): Lower bound for histogram range.
            upper (float): Upper bound for histogram range.
            select_frames (list, optional): Specific frames to analyse.
            carbon_spec (str, optional): Carbon species type ('CO2', 'BiC', 'CA', 'TS').
            save_raw (str, optional): Filename suffix for raw data output.

        Returns:
            tuple: (density_histogram, bin_centers)

        Raises:
            ValueError: If the specified atom type is not supported.
        """
        
        dens = Density(self._u)
        self._dens_lower = lower
        self._dens_upper = upper

        if atom_type == 'OW':
            traj = self._opos
        elif atom_type == 'C':
            traj = self._cpos
        elif atom_type == 'H3O':
            traj = self._h3opos
        else:
            raise ValueError(f"Unsupported atom type: {atom_type}")
        frames = select_frames if select_frames is not None else range(len(traj))


        print()
        print(f'Obtaining {atom_type} density.')
        print('Calculating density profile ...')
        result = Parallel(n_jobs=self.n_cores
                          )(delayed(dens.proximity)(
                              self._WC[i],traj[i],boxdim=self._boxdim[i],upper=self._uz,cutoff=False) for i in tqdm(frames)) # parse through frames
        self._dens_result = result


        print('Generating histogram(s)')
        hist_input = np.concatenate(result).ravel()

        # output total distance if looking at different types of carbon species
        if save_raw != None:
            np.savetxt(f'./outputs/dens_raw_{save_raw}.dat',hist_input)

        density,bin_range = np.histogram(hist_input,bins=bins,range=[lower,upper])
        x_range = [(bin_range[i]+bin_range[i+1])/2 for i in range(len(bin_range)-1)]

        xy = self._u.dimensions[0]
        hist_range = upper - lower
        mol_mass = self._get_molecular_mass(atom_type, carbon_spec)
        result_hist = [(i*mol_mass)/( 2 * (N_A) * (xy*xy*(hist_range/bins) * 10**(-30) # calculate normalised density in g/cm^3. 
                                                   ) * (len(frames)) * 10**6) for i in density]

        save_dat = np.column_stack([x_range, result_hist])
        np.savetxt(f'./outputs/{atom_type}_dens.dat', save_dat)

        print('Done\n')
        return result_hist,x_range

    def _get_molecular_mass(self, atom_type, carbon_spec=None):
        """Get molecular mass for density normalisation."""
        if atom_type == 'C':
            spec = carbon_spec if carbon_spec is not None else 'CO2'
            return MOLECULAR_MASSES.get(spec, MOLECULAR_MASSES['CO2'])
        return MOLECULAR_MASSES.get(atom_type, 18.01528)

    def nrg_from_dens(self,species='C'): 
        """Extract free energy from density profile.
        
        Args:
            species (str): Species type ('C' or 'H3O').
            
        Returns:
            tuple: (distance, free_energy) arrays.
        """
        filename = f'./outputs/{species}_dens.dat'
        fin = np.loadtxt(filename)
        dist, dens = fin[:, 0], fin[:, 1]
        const = np.sum(dens)

        nrg = [-0.000239006*R*T*np.log(i/const) for i in dens]

        # shift min to zero
        min_val = min(nrg)
        output = [i-min_val for i in nrg]

        save_dat = np.column_stack([dist, output])
        np.savetxt('./outputs/free_energy.dat', save_dat)

        return dist, output

    
    def Density_plot(self,data_Oxygen,data_Carbon=None):
        """Plot density profiles."""
        dens_plot(data_Oxygen,data_Carbon,self._dens_lower,self._dens_upper)
















    ##########################################################################
    ############################## Orientation ###############################
    ##########################################################################
        

    # orientation
        
    def Orientation_run(self,atomtype='water',histtype='time',bins=400,lower=-10,upper=10,vect='WC',prop='dipole'):
        """Compute orientations of near-interface molecules.

        Args:
            atomtype (str): Type of atom ('water' or 'carbon').
            histtype (str): Type of histogram ('time' or 'heatmap').
            lower (float): Lower bound for histogram range.
            upper (float): Upper bound for histogram range.
            bins (int): Number of bins for histogram.
            vect (str): Reference vector ('WC' interface or 'z' axis).
            prop (str): Property to compute for carbon orientation.

        Returns:
            ndarray or tuple: For 'time', returns histogram data.
                             For 'heatmap', returns (X, Y, H) meshgrid and histogram.

        Raises:
            ValueError: If the specified atom type is not supported.
        """

        ori = Orientation(self._u)  
        self._ori_lower = lower
        self._ori_upper = upper      
        
        print()
        print(f'Obtaining orientations.')
        print('Calculating orientation profile ...')

        dist, theta = self._calculate_orientations(ori, atomtype, vect, prop)

        # Adjust bounds for carbon
        if atomtype == 'carbon' and vect == 'WC':
            lower = 0
        
        print('Generating histogram(s)')
        dist_array = np.concatenate(dist).ravel()
        Theta_array = np.concatenate(theta).ravel()
        
        if histtype=='time':
            result = ori._getHistogram(dist_array,
                                    Theta_array,
                                    bins=bins,hist_range=[lower,upper])
            save_dat = np.column_stack([result[:, 0], result[:, 1]])
            np.savetxt(f'./outputs/orientation_{atomtype}.dat', save_dat)
            
            print('Done.\n')
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
            return X, Y, H
        
    def _calculate_orientations(self, ori, atomtype, vect, prop):
        """Calculate orientations based on atom type and vector."""
        
        if atomtype == 'water':
            if vect == 'WC':
                result = Parallel(n_jobs=self.n_cores)(
                    delayed(ori._getCosTheta)(
                        self._opos[i], self._h1pos[i], self._h2pos[i],
                        self._WC[i], self._boxdim[i]
                    )
                    for i in tqdm(range(len(self._opos)))
                )
            elif vect == 'z': # need to debug this case - not sure if it's working correctly.
                result = Parallel(n_jobs=self.n_cores)(
                    delayed(ori._getCosTheta_z)(
                        self._opos[i], self._h1pos[i], self._h2pos[i], self._boxdim[i]
                    )
                    for i in tqdm(range(len(self._opos)))
                )
            else:
                raise ValueError(f"Unknown vector type: {vect}")
        
        elif atomtype == 'carbon':
            result = Parallel(n_jobs=self.n_cores)(
                delayed(ori._getCosTheta_Carbon)(
                    self._cpos[i], self._ocpos1[i], self._ocpos2[i],
                    self._WC[i], self._boxdim[i], prop
                )
                for i in tqdm(range(len(self._cpos)))
            )
        
        else:
            raise ValueError(f"Unknown atom type: {atomtype}")
            
        dist = [r[0] for r in result]
        theta = [r[1] for r in result]
        return dist, theta

    def Orientation_plot(self,data_Oxygen,data_Carbon=None):
        """Plot orientation profiles."""
        oriPlot(data_Oxygen,data_Carbon,self._ori_lower,self._ori_upper)















    ##########################################################################
    ######################## Hbonding and Coordination #######################
    ##########################################################################


    def Hbonds_run_water(self,bins=100,lower=-8,upper=2):
        """Analyse hydrogen bonding for water molecules.
        
        Args:
            bins (int): Number of histogram bins.
            lower (float): Lower bound for histogram range.
            upper (float): Upper bound for histogram range.
            
        Returns:
            tuple: ((donor_hist, donor_range), (acceptor_hist, acceptor_range))
        """
        counter = Hbondz(self._u,self._uz)
        self._hbond_lower = lower
        self._hbond_upper = upper

        print()
        print(f'Obtaining Hbonds.')
        hist_don,don_range,hist_acc,acc_range = counter.hbond_analysis_water(
            self._WC,lower,upper,self._start,self._end,self._boxdim,bins)
    
        self._don,self._donx = hist_don, don_range
        self._acc,self._accx = hist_acc, acc_range

        return (hist_don, don_range, hist_acc, acc_range)


    def Hbonds_run_carbon(self,bins=100,lower=-8,upper=2,org=False,frame_select=None,filename=None,save_raw=None):
        """Analyse hydrogen bonding for carbon species.
        
        Args:
            bins (int): Number of histogram bins.
            lower (float): Lower bound for histogram range.
            upper (float): Upper bound for histogram range.
            org (bool): Whether molecules are organised.
            frame_select (list, optional): Specific frames to analyse.
            filename (str, optional): Output filename suffix.
            save_raw (str, optional): Filename for raw data output.
            
        Returns:
            tuple: ((donor_hist, donor_range), (acceptor_hist, acceptor_range))
        """
        counter = Hbondz(self._u,self._uz)
        self._hbond_lower = lower
        self._hbond_upper = upper

        hist_don,don_range,hist_acc,acc_range = counter.hbond_analysis_carbon(self._WC,
        self._cpos,lower,upper,self._start,self._end,self._boxdim,bins,org=org,frame_select=frame_select,
        filename=filename,save_raw=save_raw)
        
        self._don, self._donx = hist_don, don_range
        self._acc, self._accx = hist_acc, acc_range

        return (hist_don,don_range), (hist_acc,acc_range)
    

    
    def coordination_number(self,groupA='C',groupB='OW',r_0=3.5,filename=None,frame_select=None,
                            bins=100,lower=-8,upper=4,save_raw=None):
        """Calculate coordination number profile.
        
        Args:
            groupA (str): First group type ('C' or 'OC').
            groupB (str): Second group type (currently only 'OW' supported).
            r_0 (float): Cutoff radius for coordination.
            filename (str, optional): Output filename suffix.
            frame_select (list, optional): Specific frames to analyse.
            bins (int): Number of histogram bins.
            lower (float): Lower bound for histogram range.
            upper (float): Upper bound for histogram range.
            save_raw (str, optional): Filename for raw data output.
            
        Returns:
            ndarray: Coordination number histogram (edges, means).
        """
        cn_counter = CN(self._u)
        dens = Density(self._u)

        traj_A = self._cpos if groupA == 'C' else self._ocpos
        traj_B = self._opos
        frames = frame_select if frame_select is not None else range(len(traj_A))

        # Calculate distances to interface
        print('Calculating distances to interface ...')
        dist_result = Parallel(n_jobs=self.n_cores)(  # Changed from multiprocessing.cpu_count()
            delayed(dens.proximity)(
                self._WC[i], traj_A[i], boxdim=self._boxdim[i], upper=self._uz, cutoff=False
            )
            for i in tqdm(frames)
        )
        self._dens_result = dist_result
        distance_inp = np.concatenate(dist_result).ravel()


        # Calculate coordination numbers
        print('\nObtaining coordination number.')
        print('Calculating coordination number ...')
            
        cn_result = Parallel(n_jobs=self.n_cores)(  # Changed from multiprocessing.cpu_count()
            delayed(cn_counter.coordination_number)(
                traj_A[i], traj_B[i], self._boxdim[i], r_0
            )
            for i in tqdm(frames)
        )

        print('Generating histogram(s)')
        hist_input = cn_result
        
        if save_raw != None:
            save_file = np.array([distance_inp,hist_input]).T
            np.savetxt(f'./outputs/cn_raw_{save_raw}.dat',save_file)


        means, edges, _ = stats.binned_statistic(
            distance_inp.flatten(),  # Removed [:]
            hist_input,              # Removed [:]
            statistic='mean', bins=bins,
            range=[lower,upper]
        )
        edges = 0.5 * (edges[1:] + edges[:-1])
        hist = np.column_stack([edges, means])

        output_name = f'./outputs/coordination_number{"_" + filename if filename else ""}.dat'
        np.savetxt(output_name, hist)

        return hist


    def HBondz_plot(self):
        """Plot hydrogen bonding profiles."""
        hbondPlot(self._don,self._donx,self._acc,self._accx,self._hbond_lower,self._hbond_upper)
        

















    ##########################################################################
    ########################## Solvation character ###########################
    ##########################################################################

    '''Analyse the local solvation environements of the various carbon species
    under both interfacial and bulk conditions.'''

    def surf_RDF(self, bins=75, depth=None, hist_range=None):
        """Calculate surface-resolved radial distribution function.
        
        Args:
            bins (int): Number of histogram bins.
            depth (list): Depth range [min, max] for interface proximity.
            hist_range (list): Histogram range [min, max] for RDF.
            
        Returns:
            ndarray: RDF histogram (distance, g(r)).
        """
        if depth is None:
            depth = [-8, 4]
        if hist_range is None:
            hist_range = [2, 8]
            
        rdf = RDF(self._u)

        print('\nCalculating RDFs ...')
        result = Parallel(n_jobs=self.n_cores)(  # Changed from multiprocessing.cpu_count()
            delayed(rdf.get_rdf)(
                self._cpos[i], self._opos[i], self._WC[i], self._boxdim[i],
                depth, dr=0.08, crit_dens=0.032
            )
            for i in tqdm(range(len(self._cpos)))
        )

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
        
        save_dat = np.column_stack([x_out, means])
        np.savetxt('./outputs/surf_RDF.dat', save_dat)
        
        print('Done.\n')
        return save_dat
