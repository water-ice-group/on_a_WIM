import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib import distances
from MDAnalysis.analysis.rdf import InterRDF
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pytim
from pytim import observables
from scipy import stats

class monolayer:

    def __init__(self, universe, startstep=None,endstep=None):    
        self._u = universe
        self._start = startstep if startstep is not None else (0)
        self._end = endstep if endstep is not None else (len(self._u.trajectory)-1)


    def locate(self,atomtype='OW'):

        u = self._u
        no_layers=4
        sel = u.select_atoms(f'name {atomtype}')
        inter = pytim.ITIM(u,group=sel,molecular=False,max_layers=no_layers,normal='z',alpha=2)

        return (u,inter)


    ############################################################################################
    ###################################  RDFs  #################################################
    ############################################################################################


    def cluster_RDF_basic(self,atomtype_1,atomtype_2,layer,bins,range):
        '''Cluster analysis for a single frame.'''


        u,*_ = self.locate(atomtype=atomtype_1)
        mol_1 = u.atoms[self._u.atoms.layers==layer]
        mol_2 = u.select_atoms(f'name {atomtype_2}')

        if atomtype_1==atomtype_2:
            rdf = InterRDF(mol_1,mol_2,
                           #exclusion_block=(0,0),
                           nbins=bins,range=range)
            rdf.run()

            return (rdf.bins,rdf.rdf)
        
        else:
            rdf = InterRDF(mol_1,mol_2,nbins=bins,range=range)
            rdf.run()

            return (rdf.bins,rdf.rdf)


    def cluster_RDF_pytim(self,atomtype_1,atomtype_2,layer,bins):
        
        u,inter = self.locate(atomtype=atomtype_1)
        mol_1 = u.atoms[self._u.atoms.layers==layer]
        mol_2 = u.select_atoms(f'name {atomtype_2}')

        nres = observables.Number()
        rdf = observables.RDF(u,max_distance=10,observable=nres,observable2=nres,nbins=bins)

        for ts in u.trajectory[:]:
            rdf.sample(mol_1,mol_2)
        rdf.count[0] = 0

        return(rdf.bins,rdf.rdf)
        

# -------------------------------------------------------------------------------------------------------
    

    # def get_volume_normalization(slit_width, pair_distances, ostar_heights, c_vdw=1.7):
    #     r = pair_distances.flatten()
    #     h_top = ostar_heights.flatten() - c_vdw
    #     h_bot = slit_width.flatten() - 2*c_vdw - h_top

    #     cos_theta_top = np.where(h_top < r, h_top/r, 1)
    #     cos_theta_bot = np.where(h_bot < r, h_bot/r, 1)

    #     hist_vol = (4*np.pi*r**2 - 2*np.pi*r**2*(1-cos_theta_top) - 2*np.pi*r**2*(1-cos_theta_bot))   

    #     return hist_vol

    # def get_rdf(run, slit_width, pair_distances, ostar_heights, c_vdw=1.7, dr=0.08):
    #     area = run.dimensions[0]*run.dimensions[1]
    #     bins = np.arange(0.3,np.sqrt(area),dr)
    #     weights = get_volume_normalization(slit_width, pair_distances, ostar_heights, c_vdw)
    #     dens, edges = np.histogram(pair_distances.flatten(), weights=1/weights, bins=bins, density=False)
    #     edges = edges[:-1]
    #     bulk_dens = len(pair_distances.flatten())/(area*(np.mean(slit_width)-2*c_vdw))
    #     rdf = dens/dr/bulk_dens

    #     return rdf, edges

    # def get_ion_pair_stats(path, run, run_start=0, skip=1, ostaro_dist=3.5):
    #     ostar_atoms = run.select_atoms("name Ostar")
    #     o_atoms = run.select_atoms("name O")
    #     pair_distances = np.zeros((len(run.trajectory[start_frame:end_frame][::frame_frequency]), len(ostar_atoms),len(o_atoms)),dtype=float)
    #     ostar_heights = np.zeros((len(run.trajectory[start_frame:end_frame][::frame_frequency]), len(ostar_atoms),len(o_atoms)),dtype=float)
    #     bot_c_pos = np.min(run.atoms.positions[:,2])
    #     top_c_pos = np.max(run.atoms.positions[:,2])

    #     for i, frames in enumerate(tqdm((run.trajectory[start_frame:end_frame])[::frame_frequency])):
    #         ostar_atoms = run.select_atoms("name Ostar")
    #         o_atoms = run.select_atoms("name O")
            
    #         slit_width = np.abs(top_c_pos - bot_c_pos)
    #         reshaped_slit_width = np.full((len(ostar_atoms), len(o_atoms)), slit_width)


    #         dists = np.zeros((3, len(ostar_atoms), len(o_atoms)))
    #         for j in range(3):
    #             dists[j,:,:] = cdist((ostar_atoms.positions[:,j]%run.dimensions[j]).reshape(-1,1), (o_atoms.positions[:,j]%run.dimensions[j]).reshape(-1,1))
    #             dists[j,:,:] = np.where(dists[j,:,:] > (run.dimensions[j]/2)[..., None], dists[j,:,:] - run.dimensions[j][..., None], dists[j,:,:])

    #         for j in range(len(ostar_atoms)):
    #             for k in range(len(o_atoms)):
    #                 pair_distances[i,j,k] = np.linalg.norm(dists[:,j,k])
    #                 ostar_heights[i,j,k] = np.min([ostar_atoms.positions[j,2] - bot_c_pos, top_c_pos - ostar_atoms.positions[j,2]])
    #     np.save(path + 'pair_distances.npy', pair_distances)
    #     np.save(path + 'ostar_heights.npy', ostar_heights)




    ############################################################################################
    ###################################  Orientations  #########################################
    ############################################################################################
        

    def surf_positions(self):

        '''Obtain the positions of the first-layer water, as determined using ITIM method.'''

        u,inter = self.locate(atomtype='OW')
        
        opos_traj = []
        h1_traj = []
        h2_traj = []
        cpos_traj = []
        for ts in u.trajectory[self._start:self._end]:
            atom_1 = u.atoms[self._u.atoms.layers==1]
            atom_2 = u.select_atoms(f'name H')

            oh_dist = distance_array(atom_1.positions, # distance array loaded from module
                        atom_2.positions, 
                        box=u.dimensions)
            idx = np.argpartition(oh_dist, 3, axis=-1)
            opos = atom_1.positions
            h1pos = atom_2[idx[:, 0]].positions
            h2pos = atom_2[idx[:, 1]].positions
            
            h1pos = h1pos[:len(opos)]
            h2pos = h2pos[:len(opos)]

            opos_traj.append(opos)
            h1_traj.append(h1pos)
            h2_traj.append(h2pos)

            cpos = u.select_atoms('name C').positions
            cpos_traj.append(cpos)

        return (opos_traj,h1_traj,h2_traj)
    

class monolayer_angles:

    def get_dipoles(self,ox,h1,h2,boxdim):

        '''Input single frame. Returns the dipole vector calculated for given water molecules.'''

        center = boxdim[:3]/2
        vect1 = distances.apply_PBC(h1-ox+center,box=boxdim)
        vect2 = distances.apply_PBC(h2-ox+center,box=boxdim)
        dipVector = (vect1 + vect2) * 0.5 - center

        return dipVector
    
    def get_Ow_C_vect(self,ox,c,boxdim):

        '''Input single frame. Determine vector connecting water to closest carbon.'''

        dist_mat = distance_array(ox,c, box=boxdim)
        proxim = np.min(dist_mat,axis=1)
        loc = [(np.where(dist_mat[i] == proxim[i])[0][0]) for i in range(len(proxim))]

        vect_list = []
        for i in range(len(ox)):
            vect = c[loc[i]] - ox[i]
            vect_list.append(vect)

        return (proxim,vect_list)

    def calc_angles(self,ox,h1,h2,cpos,boxdim):

        dipVector = self.get_dipoles(ox,h1,h2,boxdim)
        interm_dist,interm_vect = self.get_Ow_C_vect(ox,cpos,boxdim)

        cosTheta = [np.dot(dipVector[i],interm_vect[i])/((np.linalg.norm(interm_vect[i]))*np.linalg.norm(dipVector[i])) for i in range(len(dipVector))]

        theta = np.rad2deg(np.arccos(cosTheta))

        return (interm_dist,theta)
    





    

    



