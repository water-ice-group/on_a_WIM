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
    
    def __init__(self, universe, **kwargs):

        self._u = universe



        
        
    def count(self, opos, h1pos, h2pos, wc, boxdim,lower,upper,bins):
        '''For each timeframe, determine the HBond count.'''
        
        # need to determine whether this is an instance of double counting or not. 
        opos_dist = distance_array(opos,opos,box=boxdim) 
        crit_1a,crit_1b = np.where( (opos_dist>0) & (opos_dist <= 3.5) )
        
        angle_array = calc_angles(opos[crit_1a],h1pos[crit_1a],opos[crit_1b],box=boxdim)
        angle_array = np.rad2deg(angle_array)
        
        crit2 = np.where(angle_array > 150.0)
        ox_idx = crit_1a[crit2]
        acc_idx = crit_1b[crit2]
        list_1 = [opos[i] for i in ox_idx]
        list_2 = [opos[i] for i in acc_idx]

        
        angle_array = calc_angles(opos[crit_1a],h2pos[crit_1a],opos[crit_1b],box=boxdim)
        angle_array = np.rad2deg(angle_array)

        crit2 = np.where(angle_array > 150.0)
        ox_idx = crit_1a[crit2]
        acc_idx = crit_1b[crit2]
        list_3 = [opos[i] for i in ox_idx]
        list_4 = [opos[i] for i in acc_idx]

        O_hbond_list = list_1 + list_2 + list_3 + list_4
        donors = list_1 + list_3
        acceptors = list_2 + list_4

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

        self._u.trajectory[time.astype(int)]
        donor = self._u.atoms[don].positions
        acceptor = self._u.atoms[acc].positions
        # donor = [self._u.atoms[int(i)].position for i in don]
        # acceptor = [self._u.atoms[int(i)].position for i in acc]

        return (time,donor,acceptor)


        # while counter < time_dat[1]:
        #     self._u.trajectory[hbond_result[counter].astype(int)] # set the time frame
        #     time = hbond_result[0].astype(int) # append the time
        #     donor = self._u.atoms[hbond_result[1].astype(int)].position # append donor positions
        #     acceptor = self._u.atoms[hbond_result[3].astype(int)].position # append acceptor positions
        # return (time,donor,acceptor)





    def hbond_analysis(self,wc,lower,upper,start,stop,boxdim,bins=250):
        hbonds = HydrogenBondAnalysis(universe=self._u,
                                      donors_sel='name OW OC',
                                      hydrogens_sel='name H',
                                      acceptors_sel='name OW',
                                      d_a_cutoff=3.5,
                                      d_h_a_angle_cutoff=140)
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

        # hbonds will return results of the following form
        # [frame, donor_ID, H_ID, acceptor_ID, bond_distance, angle]

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

        # print(result[0][1][0])
        # print(result[0][2][0])

        # time = []
        # for i in range(len(result)):
        #     time.append(result[i][0])
        #     data[int(result[i][0])][0].append(result[i][1])
        #     data[int(result[i][0])][1].append(result[i][2])

        print('Running proximity calculations.')
        dens = Density(self._u)
        num_cores = multiprocessing.cpu_count()
        result_don = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[i],np.array(result[i][1]),boxdim[i]) for i in tqdm(range(tot_steps))) 
        result_acc = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[i],np.array(result[i][2]),boxdim[i]) for i in tqdm(range(tot_steps))) 

        dist_don = np.concatenate(result_don).ravel()
        dist_acc = np.concatenate(result_acc).ravel()

        print('Acquiring background density.')
        ox_pos = []
        sel = self._u.select_atoms('name OW OC')
        for ts in self._u.trajectory[:stop]:
            ox_pos.append(sel.positions)
        bkg_dens = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[i],ox_pos[i],boxdim[i]) for i in tqdm(range(tot_steps)))
        bkg_dist = np.concatenate(bkg_dens).ravel()

        print('Binning.')
        hist_don,don_range = np.histogram(dist_don,bins=bins,range=[lower,upper])
        hist_acc,acc_range = np.histogram(dist_acc,bins=bins,range=[lower,upper])
        hist_bkg,bkg_range = np.histogram(dist_bkg,bins=bins,range=[lower,upper])
        

        out_don = []
        out_acc = []
        for i in range(len(hist_don)):
            print(hist_bkg[i])
            if hist_bkg[i] < 2:
                out_don.append(0)
                out_acc.append(0)
            else:
                out_don.append(hist_don[i]/hist_bkg[i])
                out_acc.append(hist_acc[i]/hist_bkg[i])
        don_range = 0.5*(don_range[1:]+don_range[:-1])
        acc_range = 0.5*(acc_range[1:]+acc_range[:-1])

        don_dat = np.array([don_range,out_don])
        don_dat = don_dat.transpose()
        acc_dat = np.array([acc_range,out_acc])
        acc_dat = acc_dat.transpose()
        np.savetxt('./outputs/donor.dat',don_dat)
        np.savetxt('./outputs/acceptor.dat',acc_dat)

        return (hist_don,don_range,hist_acc,acc_range)



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

            
        
