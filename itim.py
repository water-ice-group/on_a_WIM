import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib import distances
from MDAnalysis.analysis.rdf import InterRDF
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pytim

class monolayer:

    def __init__(self, universe, startstep=None,endstep=None):    
        self._u = universe
        self._start = startstep
        self._end = endstep




    def locate(self,atomtype='OW'):

        u = self._u
        no_layers=4
        sel = u.select_atoms(f'name {atomtype}')
        inter = pytim.ITIM(u,group=sel,molecular=False,max_layers=no_layers,normal='z',alpha=2)

        return u


    # def prepare(self):

    #     u = self.locate(atomtype='OW')

    #     o_traj = u.atoms[self._u.atoms.layers==1]
    #     #h1_traj = []
    #     #h2_traj = []

    #     #for ts in u.trajectory[self._start:self._end]:
    #         #o_atoms = u.atoms[self._u.atoms.layers==1]
    #         #h_atoms = u.select_atoms('name H')

    #         # oh_dist = distance_array(o_atoms,h_atoms,u.dimensions)
    #         # idx = np.argpartition(oh_dist, 3, axis=-1)
    #         # o_traj.append(o_atoms)
    #         # h1_traj.append(self._u.select_atoms('name' + ' H')[idx[:, 0]].positions)
    #         # h2_traj.append(self._u.select_atoms('name' + ' H')[idx[:, 1]].positions)

    #     return (o_traj)


    def cluster_RDF(self,atomtype_1,atomtype_2,layer,bins,range):
        '''Cluster analysis for a single frame.'''


        u = self.locate(atomtype=atomtype_1)
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



    # def cluster_RDF(self,inter_o,inter_h1,inter_h2,cpos,oc1_pos,oc2_pos,boxdim):
    #     '''Cluster analysis for a single frame.'''

    #     rdf = InterRDF(inter_o,cpos)
    #     rdf.run()

    #     #dist_mat = distance_array(inter_o,cpos,box=boxdim)
    #     #proxim = np.min(dist_mat,axis=1)
    #     #loc = [(np.where(dist_mat[i] == proxim[i])[0][0]) for i in range(len(proxim))] # list of indices for closest carbon

    #     # have inter_o detailing the interfacial atoms located in the first layer
    #     # want to define a radial distance between interface and the closest atoms
    #     # want to find the angles between interfacial carbon and the closest oxygens

    #     c_pos = [cpos[i] for i in loc]
    #     oc_pos1 = [oc1_pos[i] for i in loc]
    #     oc_pos2 = [oc2_pos[i] for i in loc]
    
    #     #center = boxdim[:3]/2
    #     coc_vect = distances.apply_PBC(oc_pos1-c_pos, boxdim)
    #     cow_vect = distances.apply_PBC(inter_o-c_pos, boxdim)

    #     cosTheta = abs(np.dot(coc_vect,cow_vect) / np.multiply(np.linalg.norm(coc_vect),np.linalg.norm(cow_vect)))

    #     return cosTheta





        




