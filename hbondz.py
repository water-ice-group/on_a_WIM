import numpy as np
from density import Density
import matplotlib.pyplot as plt
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from scipy import stats
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm



class Hbondz:
    '''calculate the number of H bonds in system as a function of 
    z-coordinate.'''
    
    def __init__(self, universe, uz):
        self._u = universe
        self._uz = uz
        self._data = None
        self._ttot = None




    #################################################################################
    ############################# MDAnalysis Module #################################
    #################################################################################


    def hbond_count(self,start,stop,mol_type,frame_select=None):
        """Run hydrogen bond analysis on the trajectory.
        
        Args:
            start (int): Starting frame index.
            stop (int): Ending frame index.
            mol_type (str): Molecule type - 'water', 'carbon', or 'carbon_unorg'.
            frame_select (list, optional): Specific frames to analyse.
            
        Returns:
            ndarray or tuple: Hydrogen bond results. For 'water', returns single array.
                             For 'carbon'/'carbon_unorg', returns (donor_hbonds, acceptor_hbonds).
                             
        Raises:
            ValueError: If unknown mol_type.
        """

        hbond_params = {
            'd_a_cutoff': 3.5,
            'd_h_cutoff': 1.2,
            'd_h_a_angle_cutoff': 140,
            'update_selections': True
        }

        if mol_type == 'water':

            hbonds = HydrogenBondAnalysis(universe=self._u,
                                donors_sel='name OW',
                                hydrogens_sel='name H',
                                acceptors_sel='name OW',
                                **hbond_params)
            hbonds.run(start=start,stop=stop)

            return hbonds.results.hbonds
        
        elif mol_type == 'carbon':

            # need to perform two different analyses for the different OW-OC combinations. 
            hbonds_1 = HydrogenBondAnalysis(universe=self._u, # donor 
                                donors_sel='name OC',
                                hydrogens_sel='name H',
                                acceptors_sel='name OW',
                                **hbond_params)
            hbonds_1.run(start=start,stop=stop)
            hbonds_2 = HydrogenBondAnalysis(universe=self._u, # acceptor
                                donors_sel='name OW',
                                hydrogens_sel='name H',
                                acceptors_sel='name OC',
                                **hbond_params)
            hbonds_2.run(start=start,stop=stop)

            return (hbonds_1.results.hbonds,hbonds_2.results.hbonds)
        
        elif mol_type == 'carbon_unorg':

            # need to perform two different analyses for the different OW-OC combinations. 
            hbonds_1 = HydrogenBondAnalysis(universe=self._u, # donor 
                                donors_sel='name O and around 1.8 name C',
                                hydrogens_sel='name H',
                                acceptors_sel='name O and not around 1.8 name C',
                                d_a_cutoff=3.5,
                                d_h_cutoff=1.2,
                                d_h_a_angle_cutoff=150,
                                update_selections=True)
            hbonds_1.run()
            hbonds_1.results.hbonds = hbonds_1.results.hbonds[np.isin(hbonds_1.results.hbonds[:, 0], frame_select)]

            hbonds_2 = HydrogenBondAnalysis(universe=self._u, # acceptor
                                donors_sel='name O and not around 1.8 name C',
                                hydrogens_sel='name H',
                                acceptors_sel='name O and around 1.8 name C',
                                d_a_cutoff=3.5,
                                d_h_cutoff=1.2,
                                d_h_a_angle_cutoff=150,
                                update_selections=True)
            hbonds_2.run()
            hbonds_2.results.hbonds = hbonds_2.results.hbonds[np.isin(hbonds_2.results.hbonds[:, 0], frame_select)]
            
            return (hbonds_1.results.hbonds, hbonds_2.results.hbonds)  # Move this INSIDE the elif block

        else:
            raise ValueError(f"Unknown mol_type: {mol_type}. Use 'water', 'carbon', or 'carbon_unorg'.")


            





    #################################################################################
    ################################# Water Hbonds ##################################
    #################################################################################

    def _water_parse(self,don,acc,time):
        """Parse hydrogen bonding results for a single timeframe.
        
        Args:
            don (ndarray): Donor atom indices.
            acc (ndarray): Acceptor atom indices.
            time (int): Frame index.
            
        Returns:
            tuple: (time, donor_positions, donor_counts, acceptor_positions, 
                   acceptor_counts, null_donor_positions, null_donor_counts,
                   null_acceptor_positions, null_acceptor_counts)
        """
        
        self._u.trajectory[time] 
        # Get donor and acceptor positions
        donor = self._u.atoms[don].positions # locate donor positions
        acceptor = self._u.atoms[acc].positions # locate acceptor positions

        # Find non-donors (waters not donating)
        no_don = np.setdiff1d(np.arange(len(self._u.atoms)), don) 
        ag = self._u.atoms[no_don] 
        nul_don = ag.select_atoms('name OW').positions 
        # Find non-acceptors (waters not accepting)
        no_acc = np.setdiff1d(np.arange(len(self._u.atoms)), acc)
        ag = self._u.atoms[no_acc]
        nul_acc = ag.select_atoms('name OW').positions

        # Count unique positions
        don_pos,don_counts = np.unique(donor,axis=0,return_counts=True)
        acc_pos,acc_counts = np.unique(acceptor,axis=0,return_counts=True)
        nuldon_counts = np.zeros(len(nul_don))
        nulacc_counts = np.zeros(len(nul_acc))

        return (time,don_pos,don_counts,acc_pos,acc_counts,nul_don,nuldon_counts,nul_acc,nulacc_counts)




    def hbond_analysis_water(self,wc,lower,upper,start,stop,boxdim,bins=250):
        """Run hydrogen bond analysis for water molecules.
        
        Args:
            wc (list): WC interface coordinates for each frame.
            lower (float): Lower bound for histogram range.
            upper (float): Upper bound for histogram range.
            start (int): Starting frame index.
            stop (int): Ending frame index.
            boxdim (list): Box dimensions for each frame.
            bins (int): Number of histogram bins.
            
        Returns:
            tuple: (mean_donors, edge_donors, mean_acceptors, edge_acceptors)
        """

        if start is None:  # Changed from == None
            start = 0
        if stop is None:  # Changed from == None
            stop = int(len(self._u.trajectory))

        result_count = self.hbond_count(start,stop,'water')
        tot_steps = int(stop - start)

        # sort results by time
        # data = dict()
        # for i in range(int(start),int(stop+1)):
        #     data[int(i)] = [[],[]]
        # self._data = data

        output_arr = np.array(result_count)
        time = output_arr[:,0]
        don_id = output_arr[:,1].astype(int)
        acc_id = output_arr[:,3].astype(int)

        # extract uniques time frames and count them
        unique_time, counts = np.unique(time, return_counts=True)
        cumulative_counts = np.cumsum(counts)
        cumulative_counts = np.insert(cumulative_counts, 0, 0)

        don_sort = [don_id[cumulative_counts[i]:cumulative_counts[i+1]] for i in range(tot_steps)]  # Remove np.array()
        acc_sort = [acc_id[cumulative_counts[i]:cumulative_counts[i+1]] for i in range(tot_steps)]  # Remove np.array()

        print('Collecting H Bond data.')
        num_cores = multiprocessing.cpu_count()
        result = Parallel(n_jobs=num_cores)(delayed(self._water_parse)(
            don_sort[i],acc_sort[i],i) for i in tqdm(range(tot_steps))) 

        # run proximity calcs
        print('Running proximity calculations.')
        dens = Density(self._u)
        result_don = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[int(unique_time[i])],np.array(result[i][1]),boxdim[int(unique_time[i])],upper=self._uz) for i in tqdm(range(len(unique_time)-1)))
        result_nul_don = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[int(unique_time[i])],np.array(result[i][5]),boxdim[int(unique_time[i])],upper=self._uz) for i in tqdm(range(len(unique_time)-1)))
        result_acc = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[int(unique_time[i])],np.array(result[i][3]),boxdim[int(unique_time[i])],upper=self._uz) for i in tqdm(range(len(unique_time)-1)))
        result_nul_acc = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[int(unique_time[i])],np.array(result[i][7]),boxdim[int(unique_time[i])],upper=self._uz) for i in tqdm(range(len(unique_time)-1)))


        # concatenate results
        dist_don = np.concatenate(result_don).ravel()
        count_don = np.concatenate([result[i][2] for i in range(len(unique_time)-1)]).ravel()
        dist_nul_don = np.concatenate(result_nul_don).ravel()
        count_nul_don = np.concatenate([result[i][6] for i in range(len(unique_time)-1)]).ravel()
        dist_don_tot = np.concatenate((dist_don, dist_nul_don)).ravel()
        count_don_tot = np.concatenate((count_don, count_nul_don)).ravel()
        
        dist_acc = np.concatenate(result_acc).ravel()
        count_acc = np.concatenate([result[i][4] for i in range(len(unique_time)-1)]).ravel()
        dist_nul_acc = np.concatenate(result_nul_acc).ravel()
        count_nul_acc = np.concatenate([result[i][8] for i in range(len(unique_time)-1)]).ravel()
        dist_acc_tot = np.concatenate((dist_acc, dist_nul_acc)).ravel()
        count_acc_tot = np.concatenate((count_acc, count_nul_acc)).ravel()
        

        # Bin statistics
        print('Binning.')
        mean_don, edge_don, _ = stats.binned_statistic(
            dist_don_tot, count_don_tot,
            statistic='mean', bins=bins, range=[lower, upper]
        )
        mean_acc, edge_acc, _ = stats.binned_statistic(
            dist_acc_tot, count_acc_tot,
            statistic='mean', bins=bins, range=[lower, upper]
        )

        edge_don = 0.5 * (edge_don[1:] + edge_don[:-1])
        edge_acc = 0.5 * (edge_acc[1:] + edge_acc[:-1])

        # Save results
        np.savetxt('./outputs/water_donor.dat', np.column_stack([edge_don, mean_don]))
        np.savetxt('./outputs/water_acceptor.dat', np.column_stack([edge_acc, mean_acc]))

        return (mean_don, edge_don, mean_acc, edge_acc)




    #################################################################################
    ################################# Carbon Hbonds #################################
    #################################################################################

    def _carbon_parse(self,id,unique_time,timeframe):

        """Parse carbon hydrogen bond data for a single timeframe.
        
        Args:
            atom_ids (list): List of atom ID arrays for each unique time.
            unique_time (ndarray): Array of unique timeframes with hbonds.
            timeframe (int): Current frame index.
            
        Returns:
            tuple: (timeframe, positions, counts, null_positions, null_counts)
        """

        self._u.trajectory[timeframe] 

        if timeframe in unique_time: # if we register a hbond 

            index = np.where(unique_time == timeframe)[0][0]
            atom_ids = id[index]

            ag = self._u.atoms[atom_ids]
            sel = ag.select_atoms('name OC').positions
            pos,counts = np.unique(sel,axis=0,return_counts=True)
            
            nul_pos = []
            nuldon_counts = []

        
        else: # if no hbond is registered in frame
            pos = []
            counts = []
            ag = self._u.atoms
            nul_pos = ag.select_atoms('name OC').positions
            nuldon_counts = np.zeros(len(nul_pos))

        return (timeframe,pos,counts,nul_pos,nuldon_counts) 
    

    

    def _organize_data(self,hbond_data,htype,frame_select):
        """Organise hydrogen bond data for carbon analysis.
        
        Args:
            hbond_data (ndarray): Raw hydrogen bond data.
            htype (str): Type - 'donor' or 'acceptor'.
            frame_select (list, optional): Specific frames to analyse.
            
        Returns:
            list: Bond counts per frame.
        """

        # organise donor data
        arr = np.array(hbond_data)
        t = arr[:,0]

        if htype == 'donor':
            atom_id = arr[:,1].astype(int)
        elif htype == 'acceptor':
            atom_id = arr[:,3].astype(int)
        else:
            raise ValueError(f"Unknown htype: {htype}. Use 'donor' or 'acceptor'.")

        frames = frame_select if frame_select is not None else range(self._ttot)
        
        bond_hits = []
        for i in frames:
            count = np.sum(t.astype(int) == int(i))
            bond_hits.append(count)

        return bond_hits


    def hbond_analysis_carbon(self,wc,cpos,lower,upper,start,stop,boxdim,bins,org,frame_select=None,filename=None,save_raw=None):
        """Run hydrogen bond analysis for carbon species.
        
        Args:
            wc (list): WC interface coordinates for each frame.
            cpos (list): Carbon atom positions for each frame.
            lower (float): Lower bound for histogram range.
            upper (float): Upper bound for histogram range.
            start (int): Starting frame index.
            stop (int): Ending frame index.
            boxdim (list): Box dimensions for each frame.
            bins (int): Number of histogram bins.
            org (bool): Whether molecules are organised.
            frame_select (list, optional): Specific frames to analyse.
            filename (str, optional): Output filename prefix.
            save_raw (str, optional): Filename suffix for raw data.
            
        Returns:
            tuple: (mean_donors, edge_donors, mean_acceptors, edge_acceptors)
        """

        if start is None:  # Changed from == None
            start = 0
        if stop is None:  # Changed from == None
            stop = int(len(self._u.trajectory))

        tot_steps = int(stop - start)
        self._ttot = tot_steps - 1

        # Perform hydrogen bond count
        mol_type = 'carbon' if org else 'carbon_unorg'
        hbonds_don, hbonds_acc = self.hbond_count(start, stop, mol_type, frame_select)

        
        # organise the data
        # extract hits for donors and acceptors, as well as nul counts (no hbonds formed)
        don_counts = self._organize_data(hbonds_don,'donor',frame_select)
        acc_counts = self._organize_data(hbonds_acc,'acceptor',frame_select)

        # run proximity calcs for carbon atoms 
        dens = Density(self._u)
        num_cores = multiprocessing.cpu_count()

        print('Running proximity calculations.')  # Add for consistency
        frames = frame_select if frame_select is not None else range(self._ttot)
        result_dist = Parallel(n_jobs=num_cores)(
            delayed(dens.proximity)(wc[i], cpos[i], boxdim[i], upper=self._uz) 
            for i in tqdm(frames)
        )
        carbon_pos = np.concatenate(result_dist).ravel()


        # Save raw data if requested
        if save_raw is not None:
            np.savetxt(f'./outputs/don_raw_{save_raw}.dat', 
                      np.column_stack([carbon_pos, don_counts]))
            np.savetxt(f'./outputs/acc_raw_{save_raw}.dat', 
                      np.column_stack([carbon_pos, acc_counts]))


        # Bin statistics
        print('Binning.')
        mean_don, edge_don, _ = stats.binned_statistic(
            carbon_pos, don_counts,
            statistic='mean', bins=bins, range=[lower, upper]
        )
        mean_acc, edge_acc, _ = stats.binned_statistic(
            carbon_pos, acc_counts,
            statistic='mean', bins=bins, range=[lower, upper]
        )

        edge_don = 0.5 * (edge_don[1:] + edge_don[:-1])
        edge_acc = 0.5 * (edge_acc[1:] + edge_acc[:-1])

        # Save results
        prefix = filename if filename is not None else 'carbon'
        np.savetxt(f'./outputs/{prefix}_donor.dat', np.column_stack([edge_don, mean_don]))
        np.savetxt(f'./outputs/{prefix}_acceptor.dat', np.column_stack([edge_acc, mean_acc]))

        return (mean_don, edge_don, mean_acc, edge_acc)


class CN:
    """Calculate coordination numbers between atom groups.
    
    Args:
        universe: MDAnalysis universe object.
    """


    def __init__(self,universe):
        self._u = universe

    def rational_switch(self,r_list,r_0,nn,mm):
        """Rational switch function for coordination number.
        
        Args:
            r_list (ndarray): Distance array.
            r_0 (float): Cutoff distance.
            nn (int): Numerator exponent.
            mm (int): Denominator exponent.
            
        Returns:
            ndarray: Switch function values.
        """
        
        func = (1 - (r_list/r_0)**nn)/(1 - (r_list/r_0)**mm)
        return func
    
    def simple_switch(self,r_list,r_0):
        """Simple step function for coordination number.
        
        Args:
            r_list (ndarray): Distance array.
            r_0 (float): Cutoff distance.
            
        Returns:
            ndarray: 1 where r < r_0, else 0.
        """
        func = np.where(r_list < r_0, 1, 0)
        return func
    
    def coordination_number(self,pos_A,pos_B,boxdim,r_0=2.0,nn=12,mm=24):
        """Calculate the coordination number between two groups of atoms.
        
        Args:
            pos_A (ndarray): Positions of group A atoms.
            pos_B (ndarray): Positions of group B atoms.
            boxdim (ndarray): Box dimensions.
            r_0 (float): Cutoff distance.
            nn (int): Numerator exponent for rational switch.
            mm (int): Denominator exponent for rational switch.
            
        Returns:
            float: Coordination number.
        """
        r_array = distance_array(pos_A,pos_B,box=boxdim)
        r_list = np.concatenate(r_array).ravel()

        coord = self.rational_switch(r_list,r_0,nn,mm)
        #coord = self.simple_switch(r_list,r_0)
        sum_coord = np.sum(coord)
        
        return sum_coord



def hbondPlot(don,donx,acc,accx,lower,upper):
    """Plot hydrogen bond profiles.
    
    Args:
        don (ndarray): Donor counts.
        donx (ndarray): Donor distances.
        acc (ndarray): Acceptor counts.
        accx (ndarray): Acceptor distances.
        lower (float): Lower x-axis limit.
        upper (float): Upper x-axis limit.
    """
    tot_bond = don + acc
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(donx, don, '.-', label='Donors')
    ax.plot(accx, acc, '.-', label='Acceptors')
    ax.plot(donx, tot_bond, '.-', label='Total',color='black')
    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_ylabel('HBond count')
    ax.set_xlim(lower, upper)
    ax.set_ylim(0, 4)
    ax.legend()
    
    plt.savefig('./outputs/hbond_profile.pdf', dpi=400, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()


