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



        
        
    def count(self, opos, h1pos, h2pos, wc, boxdim,lower,upper,bins):
        '''For each timeframe, determine the HBond count.'''
        
        # need to determine whether this is an instance of double counting or not. 
        opos_dist = distance_array(opos,opos,box=boxdim) 
        crit_1a,crit_1b = np.where( (opos_dist>0) & (opos_dist <= 3.5) )
        
        angle_array = calc_angles(opos[crit_1a],h1pos[crit_1a],opos[crit_1b],box=boxdim)
        angle_array = np.rad2deg(angle_array)
        
        crit2 = np.where(angle_array > 140.0)
        ox_idx = crit_1a[crit2]
        acc_idx = crit_1b[crit2]
        olist_1 = [opos[i] for i in ox_idx]
        olist_2 = [opos[i] for i in acc_idx]

        
        angle_array = calc_angles(opos[crit_1a],h2pos[crit_1a],opos[crit_1b],box=boxdim)
        angle_array = np.rad2deg(angle_array)

        crit2 = np.where(angle_array > 140.0)
        ox_idx = crit_1a[crit2]
        acc_idx = crit_1b[crit2]
        olist_3 = [opos[i] for i in ox_idx]
        olist_4 = [opos[i] for i in acc_idx]

        O_hbond_list = olist_1 + olist_2 + olist_3 + olist_4
        donors = olist_1 + olist_3
        acceptors = olist_2 + olist_4

        dens = Density(self._u)
        dist_tot = dens.proximity(wc,np.array(O_hbond_list),boxdim=boxdim,result='mag',cutoff=False)
        dist_don = dens.proximity(wc,np.array(donors),boxdim=boxdim,result='mag',cutoff=False)
        dist_acc = dens.proximity(wc,np.array(acceptors),boxdim=boxdim,result='mag',cutoff=False)
        dist_norm = dens.proximity(wc,np.array(opos),boxdim=boxdim,result='mag',cutoff=False)

        return (dist_tot,dist_don,dist_acc,dist_norm)


    #################################################################################
    ############################# MDAnalysis Module #################################
    #################################################################################


    def parse_frames(self,don,acc,time):
        '''Function to parse through the HBonding results.
        Filter the time and the donor and acceptor positions.
        Input consists of indivudual time frames.'''
        #print(don[0],acc[0],time)

        self._u.trajectory[time.astype(int)] # need to check this updates the box dimensions. 
        donor = self._u.atoms[don].positions
        acceptor = self._u.atoms[acc].positions

        no_don = np.setdiff1d(np.arange(len(self._u.atoms)), don)
        ag = self._u.atoms[no_don]
        nul_don = ag.select_atoms('name OW').positions

        no_acc = np.setdiff1d(np.arange(len(self._u.atoms)), acc)
        ag = self._u.atoms[no_acc]
        nul_acc = ag.select_atoms('name OW').positions

        print(f'Don {len(donor)}')
        print(f'Acc {len(acceptor)}')
        print(f'Nul Don {len(nul_don)}')    
        print(f'Nul Acc {len(nul_acc)}')

        don_pos,don_counts = np.unique(donor,axis=0,return_counts=True)
        acc_pos,acc_counts = np.unique(acceptor,axis=0,return_counts=True)
        nuldon_counts = np.zeros(len(nul_don))
        nulacc_counts = np.zeros(len(nul_acc))


        return (time,don_pos,don_counts,acc_pos,acc_counts,nul_don,nuldon_counts,nul_acc,nulacc_counts)



    def hbond_analysis(self,wc,lower,upper,start,stop,boxdim,bins=250):
        hbonds = HydrogenBondAnalysis(universe=self._u,
                                      donors_sel='name OW',
                                      hydrogens_sel='name H',
                                      acceptors_sel='name OW',
                                      d_a_cutoff=3.5,
                                      d_h_cutoff=1.3,
                                      d_h_a_angle_cutoff=140,
                                      update_selections=True)
        if start == None:
            start = 0
        if stop == None:
            stop = int(len(self._u.trajectory)-1)
        hbonds.run(start=start,stop=stop)

        # create dictionary to store results
        tot_steps = int(stop - start)
        data = dict()
        for i in range(int(start),int(stop+1)):
            data[int(i)] = [[],[]]
        self._data = data

        # sort the results by unique times. 
        output_arr = np.array(hbonds.results.hbonds)
        time = output_arr[:,0]
        don_id = output_arr[:,1].astype(int)
        acc_id = output_arr[:,3].astype(int)

        unique_time, counts = np.unique(time, return_counts=True) # extract uniques time frames and count them.
        cumulative_counts = np.cumsum(counts)
        cumulative_counts = np.insert(cumulative_counts, 0, 0)
        
        don_sort = [don_id[cumulative_counts[i]:cumulative_counts[i+1]] for i in range(tot_steps)]
        acc_sort = [acc_id[cumulative_counts[i]:cumulative_counts[i+1]] for i in range(tot_steps)]

        # parse through frames to accquire donors and acceptors. 
        print('Collecting H Bond data.')
        num_cores = multiprocessing.cpu_count()
        result = Parallel(n_jobs=num_cores)(delayed(self.parse_frames)(don_sort[i],acc_sort[i],unique_time[i]) for i in tqdm(range(tot_steps))) 

        print('Running proximity calculations.')
        dens = Density(self._u)
        num_cores = multiprocessing.cpu_count()
        result_don = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[i],np.array(result[i][1]),boxdim[i],upper=self._uz) for i in tqdm(range(tot_steps)))
        result_nul_don = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[i],np.array(result[i][5]),boxdim[i],upper=self._uz) for i in tqdm(range(tot_steps)))
        result_acc = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[i],np.array(result[i][3]),boxdim[i],upper=self._uz) for i in tqdm(range(tot_steps)))
        result_nul_acc = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[i],np.array(result[i][7]),boxdim[i],upper=self._uz) for i in tqdm(range(tot_steps)))


        dist_don = np.concatenate(result_don).ravel()
        count_don = np.concatenate([result[i][2] for i in range(tot_steps)]).ravel()
        dist_nul_don = np.concatenate(result_nul_don).ravel()
        count_nul_don = np.concatenate([result[i][6] for i in range(tot_steps)]).ravel()
        dist_don_tot = np.concatenate((dist_don, dist_nul_don)).ravel()
        count_don_tot = np.concatenate((count_don, count_nul_don)).ravel()
        

        dist_acc = np.concatenate(result_acc).ravel()
        count_acc = np.concatenate([result[i][4] for i in range(tot_steps)]).ravel()
        dist_nul_acc = np.concatenate(result_nul_acc).ravel()
        count_nul_acc = np.concatenate([result[i][8] for i in range(tot_steps)]).ravel()
        dist_acc_tot = np.concatenate((dist_acc, dist_nul_acc)).ravel()
        count_acc_tot = np.concatenate((count_acc, count_nul_acc)).ravel()
        

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
        np.savetxt('./outputs/donor.dat',don_dat)
        np.savetxt('./outputs/acceptor.dat',acc_dat)

        return (mean_don,edge_don,mean_acc,edge_acc)



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

            
        
