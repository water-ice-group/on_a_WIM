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
    ###################################  RDFs  #################################################
    ############################################################################################
    #                          Currently under maintainance

    # def cluster_RDF_basic(self,atomtype_1,atomtype_2,layer,bins,range):
    #     '''Cluster analysis for a single frame.'''


    #     u,*_ = self.locate(atomtype=atomtype_1)
    #     mol_1 = u.atoms[self._u.atoms.layers==layer]
    #     mol_2 = u.select_atoms(f'name {atomtype_2}')

    #     if atomtype_1==atomtype_2:
    #         rdf = InterRDF(mol_1,mol_2,
    #                        #exclusion_block=(0,0),
    #                        nbins=bins,range=range)
    #         rdf.run()

    #         return (rdf.bins,rdf.rdf)
        
    #     else:
    #         rdf = InterRDF(mol_1,mol_2,nbins=bins,range=range)
    #         rdf.run()

    #         return (rdf.bins,rdf.rdf)


    # def cluster_RDF_pytim(self,atomtype_1,atomtype_2,layer,bins):
        
    #     u,inter = self.locate(atomtype=atomtype_1)
    #     mol_1 = u.atoms[self._u.atoms.layers==layer]
    #     mol_2 = u.select_atoms(f'name {atomtype_2}')

    #     nres = observables.Number()
    #     rdf = observables.RDF(u,max_distance=10,observable=nres,observable2=nres,nbins=bins)

    #     for ts in u.trajectory[:]:
    #         rdf.sample(mol_1,mol_2)
    #     rdf.count[0] = 0

    #     return(rdf.bins,rdf.rdf)
        

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
    


    ############################################################################################
    ###################################  Properties  ###########################################
    ############################################################################################


class monolayer_properties:

    def __init__(self, universe):    
        self._u = universe

    def get_dipoles(self,ox,h1,h2,boxdim):

        '''Input single frame. Returns the dipole vector calculated for given water molecules.'''

        center = boxdim[:3]/2
        vect1 = distances.apply_PBC(h1-ox+center,box=boxdim)
        vect2 = distances.apply_PBC(h2-ox+center,box=boxdim)
        dipVector = (vect1 + vect2) * 0.5 - center

        return dipVector
    
    def get_closest_vect(self,atomtype_1,atomtype_2,boxdim,locr=False):

        '''Input single frame. Determine vector connecting water to closest carbon.'''

        dist_mat = distance_array(atomtype_1,atomtype_2, box=boxdim)
        proxim = np.min(dist_mat,axis=1)
        loc = [(np.where(dist_mat[i] == proxim[i])[0][0]) for i in range(len(proxim))]

        vect_list = []
        for i in range(len(atomtype_1)):
            vect = atomtype_2[loc[i]] - atomtype_1[i]
            vect_list.append(vect)

        if locr == False:
            return (proxim,vect_list)
        elif locr == True:
            return (proxim,loc)


    def calc_angles(self,vect1,vect2):

        cosTheta = [np.dot(vect1[i],vect2[i])/((np.linalg.norm(vect1[i]))*np.linalg.norm(vect2[i])) for i in range(len(vect1))]
        theta = np.rad2deg(np.arccos(cosTheta))
        return theta


    #####################################################################################################

    def calc_h2o_dipole_angle(self,ox,h1,h2,wc,boxdim):

        '''Calculate the angle between the water's dipole and the 
        vector connecting the molecule to the instantaneous
        interface. Select layer using the monolayer class.'''

        dipVector = self.get_dipoles(ox,h1,h2,boxdim)
        dist,surf_vect = Density(self._u).proximity(wc,ox,boxdim,result='both',cutoff=False)

        theta = self.calc_angles(dipVector,surf_vect)

        return (dist,theta)



    def calc_dip_C_angle(self,ox,h1,h2,cpos,boxdim):

        '''Calculate the angle between the water's dipole and the 
        closest carbon.'''

        dipVector = self.get_dipoles(ox,h1,h2,boxdim)
        interm_dist,interm_vect = self.get_closest_vect(ox,cpos,boxdim)

        theta = self.calc_angles(dipVector,interm_vect)

        return (interm_dist,theta)

    
    def calc_OW_C_RDF(self,inter_ox,cpos,boxdim):

        '''Calculate the angle between the water's dipole and the 
        closest carbon.'''

        dist_mat = distance_array(inter_ox,cpos,box=boxdim)
        output = np.concatenate(dist_mat).ravel()
        
        return output
    


    def OW_OC_dist(self,ox,ocpos1,ocpos2,boxdim):

        '''Combine the co2 oxygens at each frame. Plug into function to extract distances.
        OW-OC distance important in gauging the number of hydrogen bonds at an interface.'''

        ocpos_comb = []
        for i in range(len(ocpos1)):
            ocpos_comb.append(ocpos1[i])
            ocpos_comb.append(ocpos2[i])

        ocpos_comb = np.array(ocpos_comb)

        interm_dist,interm_vect = self.get_closest_vect(ox,ocpos_comb,boxdim)

        return interm_dist

    def OW_OC_angle(self,ox,ocpos1,ocpos2,wc,boxdim):

        '''Combine the co2 oxygens at each frame. Plug into function to extract distances.
        OW-OC distance important in gauging the number of hydrogen bonds at an interface.'''

        ocpos_comb = []
        for i in range(len(ocpos1)):
            ocpos_comb.append(ocpos1[i])
            ocpos_comb.append(ocpos2[i])

        ocpos_comb = np.array(ocpos_comb)

        interm_dist,interm_vect = self.get_closest_vect(ox,ocpos_comb,boxdim)
        dist,surf_vect = Density(self._u).proximity(wc,ox,boxdim,result='both',cutoff=False)

        theta = self.calc_angles(interm_vect,surf_vect)


        return theta


    #####################################################################################################
    
    def hbond_properties(self,ox,h1,h2,ocpos1,ocpos2,boxdim):

        ocpos_comb = []
        for i in range(len(ocpos1)):
            ocpos_comb.append(ocpos1[i])
            ocpos_comb.append(ocpos2[i])
        ocpos_comb = np.array(ocpos_comb)

        interm_dist,loc = self.get_closest_vect(ox,ocpos_comb,boxdim,locr=True) # first value returned gives distances. 
        #print(loc)
        acc = [ocpos_comb[i] for i in loc]


        h1_dist = distances.apply_PBC(h1-acc,boxdim)
        h2_dist = distances.apply_PBC(h2-acc,boxdim)


        angles = []
        for i in range(len(ox)):

            oxpos = ox[i]
            hpos = [h1[i],h2[i]]
            hpos = np.array(hpos)

            dist_mat = distance_array(ocpos_comb[loc[i]],hpos)
            proxim = np.min(dist_mat,axis=1)            
            loc_h = [(np.where(dist_mat[i] == proxim[i])[0][0]) for i in range(len(proxim))]

            cent_atom = hpos[loc_h]

            vect1 =  cent_atom - oxpos 
            vect2 = ocpos_comb[loc[i]] - oxpos
            vect1 = vect1[0]
            vect2 = vect2[0]

            cosTheta = np.dot(vect1,vect2)/((np.linalg.norm(vect1))*np.linalg.norm(vect2))
            theta = np.rad2deg(np.arccos(cosTheta))

            angles.append(theta)

        return (interm_dist,angles)









    #####################################################################################################
    
    '''Isolating interfacial CO2 more difficult. Following section looks to calculate RDFs.'''

    def extract_atoms(self,distance_matrix, threshold):
        within_threshold_mask = distance_matrix <= threshold
        within_threshold_rows = np.any(within_threshold_mask, axis=1)
        atoms_within_distance = np.where(within_threshold_rows)[0]
        
        return atoms_within_distance


    def co2_surf_dist(self,wc_inter,cpos,boxdim,cutoff):

        '''Identify CO2s residing at the water surface using proximity to the instantaneous interface.'''

        dist_mat = distance_array(cpos,wc_inter,box=boxdim)
        loc = self.extract_atoms(dist_mat,cutoff)

        co2_surf = [cpos[i] for i in loc]
        co2_surf = np.array(co2_surf)
        #print(len(co2_surf))
        #for i in range(len(co2_surf)):
        #    print(co2_surf[i][2])
        try:
            co2_dist = self_distance_array(co2_surf,box=boxdim)
            return co2_dist
        except:
            result = [0,0]
            return result
            # do nothin







    

    



