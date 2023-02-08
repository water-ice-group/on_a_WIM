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




class WillardChandler:
    
    '''Module for generating a Willard-Chandler interface and using this
    interface to calculate properties such as density and orientation.'''

    def __init__(self, universe, endstep=None):    
        self._u = universe
        self._end = endstep
        


    # generate the initial WC interface
    def generate(self,grid=400):
        '''Generate the WC interface.'''
        
        pos = AtomPos(self._u,self._end)
        self._unopos,self._unh1pos,self._unh2pos,self._opos,self._h1pos,self._h2pos,self._cpos = pos.prepare()
        
        inter = WC_Interface(self._u,grid)
        opos_traj = self._opos
        
        WC_inter = []
        iter_run = 0 
        for i in opos_traj:
            result = inter.criteria(i)
            WC_inter.append(result)
            iter_run += 1
            if (iter_run) in range(0,len(self._u.trajectory),50):
                print(f'Completed {iter_run}/{len(opos_traj)} frames')
        self._WC = WC_inter
        self.inter = inter
        
        return WC_inter

    
    # save coordinates for visualisation
    def visualise(self):  
        self.inter.save_coords(self._WC)
        
        
        
    ##########################################################################
    ############################# Measurements ###############################
    ##########################################################################
        
    # Density
    def Density_run(self,atom_type,bins=200,lower=-20,upper=10):
        '''Obtain the density of species relative to the WC interface.'''
        
        dens = Density(self._u)
    
        if atom_type == 'OW':
            traj = self._opos
        elif atom_type == 'C':
            traj = self._cpos
 
        print(atom_type)
        
        density = []
        for i in range(len(traj)):
            proxim = dens.proximity(self._WC[i],traj[i])
            density += proxim
            if i in range(0,len(traj),50):
                print(f'Completed {i}/{len(traj)} frames.')
        
        if atom_type == 'OW':
            density,x_range = np.histogram(density,bins=bins,range=[lower,upper])
            self._waterno = density
            water_dens = 18.01528
            N_A = 6.022*10**23
            xy = self._u.dimensions[0]
            hist_range = upper - lower
            result = [(i*water_dens)/( (N_A) * (xy*xy*(hist_range/bins) * 10**(-30)) * (2*len(traj)) * 10**6) for i in density]
            
        elif atom_type == 'C':
            density,x_range = np.histogram(density,bins=bins,range=[lower,upper],density=True)
            result = density
        
        save_dat = np.array([x_range[:-1],result])
        save_dat = save_dat.transpose()
        np.savetxt('./outputs/' + atom_type + '_dens.dat',save_dat)
        return (result,x_range)
    
    def Density_plot(self,data_Oxygen,data_Carbon=None):
        dens_plot(data_Oxygen,data_Carbon)

    # ------------------------------------------------------------------------

    # Orientation
    def Orientation_run(self,bins=200,lower=-20,upper=0):
        '''Obtain the orientation of the species relative to the WC interface.'''
        
        cosTheta_list = []
        dist_list = []
        ori = Orientation(self._u)        
        
        counter = 0 
        for i in range(len(self._unopos)):
            cosTheta = ori._getCosTheta(self._unopos[i],self._unh1pos[i],self._unh2pos[i],self._WC[i],self._opos[i])
            dist_list.append(cosTheta[0])
            cosTheta_list.append(cosTheta[1])
            counter += 1
            if counter in range(0,len(self._unopos)+1,50):
                print(f'Frame {counter} / {len(self._opos)}.')

        dist_array = np.concatenate(dist_list).ravel()
        Theta_array = np.concatenate(cosTheta_list).ravel()
        
        result = ori._getHistogram(dist_array,
                                   Theta_array,
                                   bins=bins,hist_range=[lower,upper])

        np.savetxt('./outputs/orientation.dat',result)
        self._ori = result
        return result
    
    def Orientation_plot(self):
        oriPlot(self._ori)

    # ------------------------------------------------------------------------

    # Hydrogen bond counting
    def Hbonds_run(self,bins=200,lower=-15,upper=0):
        '''Obtain the HBond profile mapping the average count with distance 
        from the interface.'''
        
        hbonds = []
        counter = Hbondz(self._u)
        
        for i in range(len(self._opos)):
            result = counter.count(self._unopos[i],self._unh1pos[i],self._unh2pos[i],self._WC[i],self._opos[i])
            hbonds += result
            print(i)
            
        hbonds = np.array(hbonds).ravel()
        
        hist,xrange = np.histogram(hbonds,bins=bins,range=[lower,upper])
        
        N_A = 6.022*10**23
        xy = self._u.dimensions[0]
        hist_range = upper - lower
        #result = [i/( (N_A) * (xy*xy*(hist_range/bins) * 10**(-30)) * (2*len(traj)) * 10**6) for i in hbond_count]
        
        
        result = [hist[i]/self._waterno[i] for i in range(len(hist)) if (self._waterno[i] != 0)]
        xrange = [xrange[i] for i in range(len(hist)) if (self._waterno[i] != 0)]
        
        return (result,xrange)
        






           

           

        

    
