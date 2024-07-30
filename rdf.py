import numpy as np
from density import Density
from MDAnalysis.analysis.distances import distance_array

class RDF:

    def __init__(self,universe):

        self._u = universe

    '''Pass in data on a per frame basis.
    Provide carbon positions, oxygen positions, 
    WC interface positions, and the box dimensions.'''
    

    def get_volume_normalization_approx(self,pair_distances,prox):

        r = np.array(pair_distances)
        p = prox

        if p < 0: # molecule immersed in the fluid
            cos_theta = np.where(np.abs(p) < r, np.abs(p)/r, 1)
            hist_vol = 4*np.pi*r**2 - 2*np.pi*r**2*(1-cos_theta)
        else: # p >= 0 : molecule on top of water.
            cos_theta = np.where(np.abs(p) < r, np.abs(p)/r, 0.999)            
            hist_vol = 2*np.pi*r**2*(1-cos_theta)

        return hist_vol
        

        

    def get_rdf_approx(self,pos_a,pos_b,WC,boxdim,depth,dr=0.08,crit_dens=0.032):

        # obtain pos_a-pos_b distances
        # ensure only those relevant to the histogram are considered

        dens = Density(self._u)
        prox = dens.proximity(WC,pos_a,boxdim,result='mag')[0] # determine which side of interface
        WC_dist = distance_array(pos_a,WC,box=boxdim).flatten()

        if (prox > depth[0]) and (prox < depth[1]): # molecule within 4A of the interface

            dist_mat = distance_array(pos_a, pos_b, box=boxdim)
            dists = dist_mat.flatten()

            # obtain the bins
            area = boxdim[0]*boxdim[1]
            bins = np.arange(0.1,np.sqrt(area),dr)

            # obtain the volume normalization
            # weights = self.get_volume_normalization(dists,prox,WC_dist)
            weights = self.get_volume_normalization_approx(dists,prox)

            # calculate the histogram
            dens, edges = np.histogram(dists, weights=1/weights, bins=bins, density=False)
            edges = edges[:-1]
            
            rdf = dens/dr/crit_dens

            return (edges,rdf)
        
        else:
            return ([0,0],[0,0])
        



#   ------------------------------------------------------------------------------------------------------------



    def get_volume_normalization(self,pair_distances,prox,proxies):

        
        r = np.array(pair_distances) # distance between carbon and water
        p = prox # proximity of carbon to interface
        pro = proxies # proximity of water to interface

        if p < 0: # molecule immersed in the fluid
            cos_theta = np.where(np.abs(p) < r, np.abs(p)/r, 1)
            hist_vol = 4*np.pi*r**2 - 2*np.pi*r**2*(1-cos_theta)
        else: # p >= 0 : molecule on top of water.
            cos_theta = np.where(np.abs(p) < r, np.abs(p)/r, 0.999)            
            hist_vol = np.where(pro < 0, 2*np.pi*r**2*(1-cos_theta), 4*np.pi*r**2 - 2*np.pi*r**2*(1-cos_theta))

        return hist_vol

    def get_rdf(self,pos_a,pos_b,WC,boxdim,depth,dr=0.08,crit_dens=0.032):

        # obtain pos_a-pos_b distances
        # ensure only those relevant to the histogram are considered

        dens = Density(self._u)
        prox_c = dens.proximity(WC,pos_a,boxdim,result='mag')[0] # determine which side of interface
        # determine which side of interface

        if (prox_c > depth[0]) and (prox_c < depth[1]): # molecule within 4A of the interface

            dist_mat = distance_array(pos_a, pos_b, box=boxdim)
            dists = dist_mat.flatten()
            prox_h2o = np.array(dens.proximity(WC,pos_b,boxdim,result='mag'))
            proxies = prox_h2o.flatten()

            # obtain the bins
            area = boxdim[0]*boxdim[1]
            bins = np.arange(0.1,np.sqrt(area),dr)

            # obtain the volume normalization
            # weights = self.get_volume_normalization(dists,prox,WC_dist)
            weights = self.get_volume_normalization(dists,prox_c,proxies)

            # calculate the histogram
            dens, edges = np.histogram(dists, weights=1/weights, bins=bins, density=False)
            edges = edges[:-1]
            
            rdf = dens/dr/crit_dens

            return (edges,rdf)
        
        else:
            return ([0,0],[0,0])








