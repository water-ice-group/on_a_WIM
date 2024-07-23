import numpy as np
from density import Density
from MDAnalysis.analysis.distances import distance_array

class RDF:

    def __init__(self,universe):

        self._u = universe

    '''Pass in data on a per frame basis.
    Provide carbon positions, oxygen positions, 
    WC interface positions, and the box dimensions.'''

    def get_volume_normalization(self,pair_distances,proximities):

        r = pair_distances.flatten()
        try:
            p = proximities.flatten()
        except:
            p = proximities[0]

        cos_theta = np.where(p < r, p/r, 1)

        if p < 0: # molecule immersed in the fluid
            hist_vol = 4*np.pi*r**2 - 2*np.pi*r**2*(1-cos_theta)
        else: # p >= 0 : molecule on top of water. 
            hist_vol = 2*np.pi*r**2*(1-cos_theta)

        return hist_vol


    def get_rdf(self,pos_a,pos_b,WC,boxdim,dr=0.08,crit_dens=0.032):

        # obtain pos_a-pos_b distances
        dist_mat = distance_array(pos_a, pos_b, box=boxdim)

        # calculate proximity of pos_a to the WC interface
        dens = Density(self._u)
        WC_dist = dens.proximity(WC,pos_a,boxdim,result='mag')

        # obtain the bins
        area = boxdim[0]*boxdim[1]
        bins = np.arange(0.1,np.sqrt(area),dr)

        # obtain the volume normalization
        weights = self.get_volume_normalization(dist_mat,WC_dist)

        # calculate the histogram
        dens, edges = np.histogram(dist_mat.flatten(), weights=1/weights, bins=bins, density=False)
        edges = edges[:-1]
        
        rdf = dens/dr/crit_dens

        return edges,rdf






