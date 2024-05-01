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


    def parse_frames(self,hbond_result):
        '''Function to parse through the HBonding results.
        Filter the time and the donor and acceptor positions.
        Input consists of indivudual time frames.'''

        self._u.trajectory[hbond_result[0].astype(int)] # set the time frame
        time = hbond_result[0].astype(int) # append the time
        donor = self._u.atoms[hbond_result[1].astype(int)].position # append donor positions
        acceptor = self._u.atoms[hbond_result[3].astype(int)].position # append acceptor positions
        return (time,donor,acceptor)


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

        # hbonds will return results of the following form
        # [frame, donor_ID, H_ID, acceptor_ID, bond_distance, angle]


        output_arr = np.array(hbonds.results.hbonds)
        time = output_arr[:,0]
        don_id = output_arr[:,1]
        acc_id = output_arr[:,3]
        counts = hbonds.times

        print(start)
        print(stop)

        # create dictionary to store results
        steps = [i[0] for i in hbonds.results.hbonds] # will feature multiple occurences of the same step
        tot_steps = int(stop - start)
        data = dict()

        range_t = range(int(start),int(stop))
        for i in range_t:
            data[int(i)] = [[],[],[]]
        self._data = data

        # parse through frames to accquire donors and acceptors. 
        print('Collecting H Bond data.')
        num_cores = multiprocessing.cpu_count()
        result = Parallel(n_jobs=num_cores)(delayed(self.parse_frames)(hbonds.results.hbonds[i]) for i in tqdm(range(len(hbonds.results.hbonds)))) 
        
        time = []
        bkg  = []
        for i in range(len(result)):
            time.append(result[i][0])
            data[int(result[i][0])][0].append(result[i][1])
            data[int(result[i][0])][1].append(result[i][2])
            data[int(result[i][0])][2].append(result[i][1].tolist())
            data[int(result[i][0])][2].append(result[i][2].tolist())
        
        # extract the unique values from the background
        bkg = []
        for i in range_t:
            dat = np.array(data[i][2])
            result = np.unique(dat,axis=0)
            bkg.append(result)


        print('Running proximity calculations.')
        dens = Density(self._u)
        num_cores = multiprocessing.cpu_count()
        result_don = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[i],np.array(data[i][0]),boxdim[i],upper=self._uz) for i in tqdm(range_t)) 
        result_acc = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[i],np.array(data[i][1]),boxdim[i],upper=self._uz) for i in tqdm(range_t))
        result_bkg = Parallel(n_jobs=num_cores)(delayed(dens.proximity)(wc[i],bkg[i],boxdim[i],upper=self._uz) for i in tqdm(range_t))

        dist_don = np.concatenate(result_don).ravel()
        dist_acc = np.concatenate(result_acc).ravel()
        dist_bkg = np.concatenate(result_bkg).ravel()

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

            
        
