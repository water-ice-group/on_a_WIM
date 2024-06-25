import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.analysis.distances import self_distance_array
from MDAnalysis.lib import distances
from MDAnalysis.analysis.rdf import InterRDF
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pytim
from pytim import observables
from scipy import stats
from density import Density
from ase import io
import numpy as np

class monolayer:

    def __init__(self, universe, startstep=None,endstep=None):    
        self._u = universe
        self._start = startstep if startstep is not None else (0)
        self._end = endstep if endstep is not None else (len(self._u.trajectory)-1)


    def locate(self,atomtype='OW'):

        u_alt = self._u
        no_layers=4
        sel = u_alt.select_atoms(f'name {atomtype}')
        inter = pytim.ITIM(u_alt,group=sel,molecular=False,max_layers=no_layers,normal='z',alpha=2)

        return (u_alt,inter)


    ############################################################################################
    #########################  Identify surface H2O ############################################
    ############################################################################################

    def surf_positions(self):

        '''Obtain the positions of the first-layer water, as determined using ITIM method.'''

        u_alt,*_ = self.locate(atomtype='OW')
        
        opos_traj = []
        h1_traj = []
        h2_traj = []
        for ts in u_alt.trajectory[self._start:self._end]:
            atom_1 = u_alt.atoms[u_alt.atoms.layers==1]
            atom_2 = u_alt.select_atoms(f'name H')

            oh_dist = distance_array(atom_1.positions, # distance array loaded from module
                        atom_2.positions, 
                        box=u_alt.dimensions)
            idx = np.argpartition(oh_dist, 3, axis=-1)
            opos = atom_1.positions
            h1pos = atom_2[idx[:, 0]].positions
            h2pos = atom_2[idx[:, 1]].positions
            
            h1pos = h1pos[:len(opos)]
            h2pos = h2pos[:len(opos)]

            opos_traj.append(opos)
            h1_traj.append(h1pos)
            h2_traj.append(h2pos)

        return (opos_traj,h1_traj,h2_traj)
    

    def surf_positions_single_interface(self,boxdim):

        '''Determine the interfacial water molecules for a single interface.
        (Required for NPT analysis)'''

        opos,h1_traj,h2_traj = self.surf_positions()
        

        updated_opos = []
        updated_h1pos = []
        updated_h2pos = []

        for i in range(len(opos)):
            frame_opos = []
            frame_h1pos = []
            frame_h2pos = []

            opos_wrap = distances.apply_PBC(opos[i],boxdim[i])
            h1pos_wrap = distances.apply_PBC(h1_traj[i],boxdim[i])
            h2pos_wrap = distances.apply_PBC(h2_traj[i],boxdim[i])

            for j in range(len(opos_wrap)):
                if opos_wrap[j][2] < 30:
                    frame_opos.append(opos_wrap[j])
                    frame_h1pos.append(h1pos_wrap[j])
                    frame_h2pos.append(h2pos_wrap[j])
            updated_opos.append(np.array(frame_opos))
            updated_h1pos.append(np.array(frame_h1pos))
            updated_h2pos.append(np.array(frame_h2pos))
        
        return (updated_opos,updated_h1pos,updated_h2pos)

# ----------------------------------------------------------------------------------------------------------
    
    # save coordinates of the interfacial water molecules
    def save_coords(self):
    
        u_alt,interface = self.locate()
        with mda.Writer("outputs/interfacial_h2o.pdb") as W:
            for ts in u_alt.trajectory:
                ag = u_alt.atoms[u_alt.atoms.layers==1]
                W.write(ag)

    



    ############################################################################################
    ###################################  Properties  ###########################################
    ############################################################################################


class monolayer_properties:

    def __init__(self, universe):    
        self._u = universe

    def get_dipoles(self,ox,h1,h2,boxdim):

        '''Input single frame. Returns the dipole vector calculated for given water molecules.'''

        vect1 = distances.minimize_vectors(h1-ox,box=boxdim)
        vect2 = distances.minimize_vectors(h2-ox,box=boxdim)
        dipVector = (vect1 + vect2) * 0.5

        return dipVector
    
    
    def get_OH_vects(self,ox,h1,h2,boxdim):

        '''Input single frame. Returns the dipole vector calculated for given water molecules.'''

        vect1 = distances.minimize_vectors(h1-ox,box=boxdim)
        vect2 = distances.minimize_vectors(h2-ox,box=boxdim)

        return (vect1,vect2)
    

    
    def get_closest_vect(self,atomtype_1,atomtype_2,boxdim,locr=False):

        '''Input single frame. Determine vector connecting water to closest carbon.'''

        dist_mat = distance_array(atomtype_1,atomtype_2, box=boxdim)
        proxim = np.min(dist_mat,axis=1)
        loc = [(np.where(dist_mat[i] == proxim[i])[0][0]) for i in range(len(proxim))]

        vect_list = []
        center = boxdim[:3]/2
        for i in range(len(atomtype_1)):
            vect = distances.minimize_vectors(atomtype_2[loc[i]]-atomtype_1[i],box=boxdim)
            #vect = init-center
            vect_list.append(vect)

        if locr == False:
            return (proxim,vect_list)
        elif locr == True:
            return (proxim,loc)


    def calc_angles(self,vect1,vect2):

        # require the negative of vect2 to get vector from water to interface
        cosTheta = [np.dot(vect1[i],-vect2[i])/((np.linalg.norm(vect1[i]))*np.linalg.norm(-vect2[i])) for i in range(len(vect1))]
        theta = np.rad2deg(np.arccos(cosTheta))

        return theta
    
    def calc_angle_normal(self,vect1):
        vect2    = [0,0,1]
        cosTheta = [np.dot(vect1[i],vect2)/((np.linalg.norm(vect1[i]))*np.linalg.norm(vect2)) for i in range(len(vect1))]
        return cosTheta





    #####################################################################################################


    ''''Analyse the surface CO2.'''
    def surf_co2(self,wc,cpos,boxdim,lower,upper):
    
        dens = Density(self._u)
        result = dens.proximity(wc,cpos,boxdim)
        result = np.array(result)
        indices = np.where((result >= lower) & (result <= upper))[0]
        co2_surf = cpos[indices]
        
        try:
            co2_dist = self_distance_array(co2_surf, box=boxdim)
            return co2_dist
        except:
            result = [0, 0]
            return result
            # do nothing if no co2 at the surface.


    def co2_bond_angles_surf(self,wc,cpos,ocpos1,ocpos2,boxdim,lower,upper):

        dens = Density(self._u)
        result = dens.proximity(wc,cpos,boxdim)
        result = np.array(result)
        indices = np.where((result >= lower) & (result <= upper))[0]

        co2_surf = cpos[indices]
        oc1_surf = ocpos1[indices]
        oc2_surf = ocpos2[indices]

        vect_1 = distances.minimize_vectors(oc1_surf - co2_surf,box=boxdim)
        vect_2 = distances.minimize_vectors(oc2_surf - co2_surf,box=boxdim)

        dist,surf_vect = dens.proximity(wc,co2_surf,boxdim,result='both',cutoff=False)

        theta_1 = self.calc_angles(vect_1,surf_vect)
        theta_2 = self.calc_angles(vect_2,surf_vect)

        output = np.concatenate((theta_1,theta_2))

        return output




   








    

    



