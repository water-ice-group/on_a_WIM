# define functions for calculating the number of HBonds per distance. 


import numpy as np
from density import Density
import math as m
import matplotlib.pyplot as plt
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import calc_angles
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from scipy import stats
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm



class Hbondz:
    '''calculate the number of H bonds in system as a function of 
    z-coordinate.'''
    
    def __init__(self, universe, uz, **kwargs):

        self._u = universe
        self._uz = uz



        
    # --------------------------------------------------------------------------------
    # Redacted feature
    # --------------------------------------------------------------------------------


    # def count(self, opos, h1pos, h2pos, wc, boxdim,lower,upper,bins):
    #     '''For each timeframe, determine the HBond count.'''
        
    #     # need to determine whether this is an instance of double counting or not. 
    #     opos_dist = distance_array(opos,opos,box=boxdim) 
    #     crit_1a,crit_1b = np.where( (opos_dist>0) & (opos_dist <= 3.5) )
        
    #     angle_array = calc_angles(opos[crit_1a],h1pos[crit_1a],opos[crit_1b],box=boxdim)
    #     angle_array = np.rad2deg(angle_array)
        
    #     crit2 = np.where(angle_array > 140.0)
    #     ox_idx = crit_1a[crit2]
    #     acc_idx = crit_1b[crit2]
    #     olist_1 = [opos[i] for i in ox_idx]
    #     olist_2 = [opos[i] for i in acc_idx]

        
    #     angle_array = calc_angles(opos[crit_1a],h2pos[crit_1a],opos[crit_1b],box=boxdim)
    #     angle_array = np.rad2deg(angle_array)

    #     crit2 = np.where(angle_array > 140.0)
    #     ox_idx = crit_1a[crit2]
    #     acc_idx = crit_1b[crit2]
    #     olist_3 = [opos[i] for i in ox_idx]
    #     olist_4 = [opos[i] for i in acc_idx]

    #     O_hbond_list = olist_1 + olist_2 + olist_3 + olist_4
    #     donors = olist_1 + olist_3
    #     acceptors = olist_2 + olist_4

    #     dens = Density(self._u)
    #     dist_tot = dens.proximity(wc,np.array(O_hbond_list),boxdim=boxdim,result='mag',cutoff=False)
    #     dist_don = dens.proximity(wc,np.array(donors),boxdim=boxdim,result='mag',cutoff=False)
    #     dist_acc = dens.proximity(wc,np.array(acceptors),boxdim=boxdim,result='mag',cutoff=False)
    #     dist_norm = dens.proximity(wc,np.array(opos),boxdim=boxdim,result='mag',cutoff=False)

    #     return (dist_tot,dist_don,dist_acc,dist_norm)


    #################################################################################
    ############################# MDAnalysis Module #################################
    #################################################################################


    def hbond_count(self,start,stop,mol_type):

        '''Run hydrogen bond analysis on the trajectory.
        Define what molecule, water or cabon-species, to track
        the hbond count for.'''


        # perform MDAnalysis hydrogen bond count

        if mol_type == 'water':

            hbonds = HydrogenBondAnalysis(universe=self._u,
                                donors_sel='name OW',
                                hydrogens_sel='name H',
                                acceptors_sel='name OW',
                                d_a_cutoff=3.5,
                                d_h_cutoff=1.2,
                                d_h_a_angle_cutoff=140,
                                update_selections=True)
            hbonds.run(start=start,stop=stop)

            return hbonds.results.hbonds
        
        elif mol_type == 'carbon':

            # need to perform two different analyses for the different OW-OC combinations. 

            hbonds_1 = HydrogenBondAnalysis(universe=self._u, # donor 
                                donors_sel='name OC',
                                hydrogens_sel='name H',
                                acceptors_sel='name OW',
                                d_a_cutoff=3.5,
                                d_h_cutoff=1.2,
                                d_h_a_angle_cutoff=140,
                                update_selections=True)
            hbonds_1.run(start=start,stop=stop)
            hbonds_2 = HydrogenBondAnalysis(universe=self._u, # acceptor
                                donors_sel='name OW',
                                hydrogens_sel='name H',
                                acceptors_sel='name OC',
                                d_a_cutoff=3.5,
                                d_h_cutoff=1.2,
                                d_h_a_angle_cutoff=140,
                                update_selections=True)
            hbonds_2.run(start=start,stop=stop)

            return (hbonds_1.results.hbonds,hbonds_2.results.hbonds)




    #################################################################################
    ################################# Water Hbonds ##################################
    #################################################################################

    def water_parse(self,don,acc,time):

        '''Function to parse through the HBonding results.
        Filter the time and the donor and acceptor positions.
        Input consists of indivudual time frames.'''
        
        self._u.trajectory[time] # need to check this updates the box dimensions. 

        donor = self._u.atoms[don].positions # locate donor positions
        acceptor = self._u.atoms[acc].positions # locate acceptor positions

        no_don = np.setdiff1d(np.arange(len(self._u.atoms)), don) # indexes of non-donors
        ag = self._u.atoms[no_don] # select non-donors
        nul_don = ag.select_atoms('name OW').positions # locate non-donor positions 
        no_acc = np.setdiff1d(np.arange(len(self._u.atoms)), acc)
        ag = self._u.atoms[no_acc]
        nul_acc = ag.select_atoms('name OW').positions

        don_pos,don_counts = np.unique(donor,axis=0,return_counts=True)
        acc_pos,acc_counts = np.unique(acceptor,axis=0,return_counts=True)
        nuldon_counts = np.zeros(len(nul_don))
        nulacc_counts = np.zeros(len(nul_acc))

        return (time,don_pos,don_counts,acc_pos,acc_counts,nul_don,nuldon_counts,nul_acc,nulacc_counts)




    def hbond_analysis_water(self,wc,lower,upper,start,stop,boxdim,bins=250):

        '''Run hydrogen bond analysis on the trajectory.'''

        if start == None:
            start = 0
        if stop == None:
            stop = int(len(self._u.trajectory))

        result_count = self.hbond_count(start,stop,'water')


        # create dictionary to store results
        tot_steps = int(stop - start)
        data = dict()
        for i in range(int(start),int(stop+1)):
            data[int(i)] = [[],[]]
        self._data = data

        # sort the results by unique times. 
        output_arr = np.array(result_count)
        time = output_arr[:,0]
        don_id = output_arr[:,1].astype(int)
        acc_id = output_arr[:,3].astype(int)

        # extract uniques time frames and count them
        unique_time, counts = np.unique(time, return_counts=True)
        cumulative_counts = np.cumsum(counts)
        cumulative_counts = np.insert(cumulative_counts, 0, 0)

        don_sort = np.array([don_id[cumulative_counts[i]:cumulative_counts[i+1]] for i in range(tot_steps)])
        acc_sort = np.array([acc_id[cumulative_counts[i]:cumulative_counts[i+1]] for i in range(tot_steps)])

        print('Collecting H Bond data.')
        num_cores = multiprocessing.cpu_count()
        result = Parallel(n_jobs=num_cores)(delayed(self.water_parse)(don_sort[i],acc_sort[i],i) for i in tqdm(range(tot_steps))) 

        # run proximity calcs
        print('Running proximity calculations.')
        dens = Density(self._u)
        num_cores = multiprocessing.cpu_count()
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
        

        # analyse stats
        print('Binning.')
        mean_don,edge_don,binnumber = stats.binned_statistic(dist_don_tot,
                                                       count_don_tot,
                                                       statistic='mean',
                                                       bins=bins,
                                                       range=[lower,upper])
        mean_acc,edge_acc,binnumber = stats.binned_statistic(dist_acc_tot,
                                                         count_acc_tot,
                                                         statistic='mean',
                                                         bins=bins,
                                                         range=[lower,upper])

        edge_don = 0.5*(edge_don[1:]+edge_don[:-1])
        edge_acc = 0.5*(edge_acc[1:]+edge_acc[:-1])

        don_dat = np.array([edge_don,mean_don])
        don_dat = don_dat.transpose()
        acc_dat = np.array([edge_acc,mean_acc])
        acc_dat = acc_dat.transpose()
        np.savetxt(f'./outputs/water_donor.dat',don_dat)
        np.savetxt(f'./outputs/water_acceptor.dat',acc_dat)

        return (mean_don,edge_don,mean_acc,edge_acc)




    #################################################################################
    ################################# Carbon Hbonds #################################
    #################################################################################

    def carbon_parse(self,id,unique_time,timeframe):

        '''At each time frame, acquire the positions of the 
        relveant atoms.'''

        self._u.trajectory[timeframe] 

        if timeframe in unique_time: # if we register a hbond 

            index = np.where(unique_time == timeframe)[0][0]

            atom_ids = id[index]

            ag = self._u.atoms[atom_ids]
            sel = ag.select_atoms('name OC').positions
            pos,counts = np.unique(sel,axis=0,return_counts=True)

            # excl = np.setdiff1d(np.arange(len(self._u.atoms)), atom_ids) # indexes of non-donors
            # ag = self._u.atoms[excl] # select non-donors
            # nul_pos = ag.select_atoms('name OC').positions # locate non-donor positions 

            # if len(nul_pos) > 0: 
            #     nuldon_counts = np.zeros(len(nul_pos))
            # else: # if all OC atoms are involved in hbonds
            
            nul_pos = []
            nuldon_counts = []

        
        else: # if no hbond is registered in frame
            pos = []
            counts = []
            ag = self._u.atoms
            nul_pos = ag.select_atoms('name OC').positions
            nuldon_counts = np.zeros(len(nul_pos))

        return (timeframe,pos,counts,nul_pos,nuldon_counts) 
    

    

    def organize_data(self,hbond_data,htype):

        '''Organise the data for the carbon hbond analysis.'''

        # organise donor data
        arr = np.array(hbond_data)
        t = arr[:,0]
        if htype == 'donor':
            atom_id = arr[:,1].astype(int)
        elif htype == 'acceptor':
            atom_id = arr[:,3].astype(int)

        bond_hits = []
        for i in range(self._ttot):
            count = 0
            for j in t:
                if int(j) == int(i):
                    count += 1
            bond_hits.append(count)

        return (bond_hits)



    def hbond_analysis_carbon(self,wc,cpos,lower,upper,start,stop,boxdim,bins=250):

        '''Run hydrogen bond analysis on the trajectory.'''

        if start == None:
            start = 0
        if stop == None:
            stop = int(len(self._u.trajectory))
        tot_steps = int(stop - start)
        self._ttot = tot_steps - 1

        # perform MDAnalysis hydrogen bond count
        hbonds_don,hbonds_acc = self.hbond_count(start,stop,'carbon')
        

        # organise the data
        # extract hits for donors and acceptors, as well as nul counts (no hbonds formed)
        don_counts = self.organize_data(hbonds_don,'donor')
        acc_counts = self.organize_data(hbonds_acc,'acceptor')

        # run proximity calcs for carbon atoms 
        dens = Density(self._u)
        num_cores = multiprocessing.cpu_count()
        print(len(wc))
        print(len(cpos))
        print(len(boxdim))
        result_dist = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[i],cpos[i],boxdim[i],upper=self._uz) for i in tqdm(range(self._ttot)))
        carbon_pos = np.concatenate(result_dist).ravel()

        # analyse stats
        print('Binning.')
        mean_don,edge_don,binnumber = stats.binned_statistic(carbon_pos,
                                                       don_counts,
                                                       statistic='mean',
                                                       bins=bins,
                                                       range=[lower,upper])
        mean_acc,edge_acc,binnumber = stats.binned_statistic(carbon_pos,
                                                         acc_counts,
                                                         statistic='mean',
                                                         bins=bins,
                                                         range=[lower,upper])

        edge_don = 0.5*(edge_don[1:]+edge_don[:-1])
        edge_acc = 0.5*(edge_acc[1:]+edge_acc[:-1])

        don_dat = np.array([edge_don,mean_don])
        don_dat = don_dat.transpose()
        acc_dat = np.array([edge_acc,mean_acc])
        acc_dat = acc_dat.transpose()
        np.savetxt(f'./outputs/carbon_donor.dat',don_dat)
        np.savetxt(f'./outputs/carbon_acceptor.dat',acc_dat)

        return (mean_don,edge_don,mean_acc,edge_acc)


class CN:

    def __init__(self,universe):
        self._u = universe

    def rational_switch(self,r_list,r_0=2.0,nn=6,mm=12):
        '''Rational switch function for the coordination number.'''
        
        func = (1 - (r_list/r_0)**nn)/(1 - (r_list/r_0)**mm)
        return func
    
    def coordination_number(self,pos_A,pos_B,boxdim):
        '''Calculate the coordination number between two groups of atoms.'''
        
        r_array = distance_array(pos_A,pos_B,box=boxdim)
        r_list = np.concatenate(r_array).ravel()

        coord = self.rational_switch(r_list,r_0=3.44,nn=6,mm=12)
        sum_coord = np.sum(coord)
        
        return sum_coord



def hbondPlot(don,donx,acc,accx,lower,upper):
    fig, ax = plt.subplots()
    ax.plot(donx,don,'.-',label='Donors')
    ax.plot(accx,acc,'.-',label='Acceptors')
    ax.set_xlabel('Distance / $\mathrm{\AA}$')
    ax.set_ylabel('HBond count')
    ax.set_xlim(lower,upper)
    ax.set_ylim(0,4)
    ax.legend()
    plt.savefig('./outputs/hbond_profile.pdf',dpi=400,bbox_inches='tight',facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()

            
