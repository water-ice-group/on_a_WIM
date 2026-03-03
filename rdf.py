import numpy as np
from density import Density
from MDAnalysis.analysis.distances import distance_array

class RDF:
    """Calculate radial distribution functions near interfaces.
    
    Computes RDFs with volume corrections accounting for the presence
    of the interface (partial spherical shells).
    
    Args:
        universe: MDAnalysis universe object.
    """
    def __init__(self,universe):

        self._u = universe

    

    def get_volume_normalization_approx(self,pair_distances,prox):
        """Calculate approximate volume normalisation for RDF.
        
        Accounts for partial spherical shells when molecule is near interface.
        Uses only the proximity of the central molecule.
        
        Args:
            pair_distances (ndarray): Distances between atom pairs.
            prox (float): Proximity of central molecule to interface.
                         Negative = immersed in fluid, positive = above interface.
            
        Returns:
            ndarray: Volume normalisation factors for each distance.
        """
        r = np.array(pair_distances)

        if prox < 0: # molecule immersed in the fluid
            cos_theta = np.where(np.abs(prox) < r, np.abs(prox)/r, 1)
            hist_vol = 4*np.pi*r**2 - 2*np.pi*r**2*(1-cos_theta)
        else: # p >= 0 : molecule on top of water.
            cos_theta = np.where(np.abs(prox) < r, np.abs(prox)/r, 0.999)            
            hist_vol = 2*np.pi*r**2*(1-cos_theta)

        return hist_vol
        

    def get_volume_normalization(self,pair_distances,prox,proxies):
        """Calculate volume normalisation for RDF with full correction.
        
        Accounts for partial spherical shells when molecule is near interface.
        Uses proximity of both the central molecule and surrounding molecules.
        
        Args:
            pair_distances (ndarray): Distances between atom pairs.
            prox (float): Proximity of central molecule to interface.
            proxies (ndarray): Proximities of surrounding molecules to interface.
            
        Returns:
            ndarray: Volume normalisation factors for each distance.
        """
        
        r = np.array(pair_distances)

        if prox < 0:  # Molecule immersed in the fluid
            cos_theta = np.where(np.abs(prox) < r, np.abs(prox) / r, 1)
            hist_vol = 4 * np.pi * r**2 - 2 * np.pi * r**2 * (1 - cos_theta)
        else:  # prox >= 0: Molecule on top of water
            cos_theta = np.where(np.abs(prox) < r, np.abs(prox) / r, 0.99)
            # Use full sphere if surrounding molecule is below interface
            hist_vol = np.where(
                proxies < 0,
                2 * np.pi * r**2 * (1 - cos_theta),
                4 * np.pi * r**2 - 2 * np.pi * r**2 * (1 - cos_theta)
            )

        return hist_vol




    def get_rdf_approx(self,pos_a,pos_b,WC,boxdim,depth,dr=0.08,crit_dens=0.032):
        """Calculate RDF using approximate volume normalisation.
        
        Uses only the proximity of the central molecule for volume correction.
        Pass in data on a per-frame basis.
        
        Args:
            pos_a (ndarray): Positions of central atoms (e.g., carbon).
            pos_b (ndarray): Positions of surrounding atoms (e.g., oxygen).
            WC (ndarray): WC interface coordinates.
            boxdim (ndarray): Box dimensions for PBC.
            depth (list): [min, max] depth range for interface proximity.
            dr (float): Bin width for histogram.
            crit_dens (float): Critical density for normalisation.
            
        Returns:
            tuple: (bin_edges, rdf_values) or ([0, 0], [0, 0]) if outside depth range.
        """
        dens = Density(self._u)
        prox = dens.proximity(WC,pos_a,boxdim,result='mag')[0] # determine which side of interface

        # Check if molecule is within specified depth range
        if not (depth[0] < prox < depth[1]):
            return ([0, 0], [0, 0])

        dist_mat = distance_array(pos_a, pos_b, box=boxdim)
        dists = dist_mat.flatten()

        # obtain the bins
        area = boxdim[0]*boxdim[1]
        bins = np.arange(0.1,np.sqrt(area),dr)

        # obtain the volume normalization
        weights = self.get_volume_normalization_approx(dists,prox)

        # calculate volume-normalised histogram
        hist, edges = np.histogram(dists, weights=1/weights, bins=bins, density=False)  # Changed dens to hist
        edges = edges[:-1]
        
        rdf = hist/dr/crit_dens  # Changed dens to hist

        return (edges,rdf)

    def get_rdf(self,pos_a,pos_b,WC,boxdim,depth,dr=0.08,crit_dens=0.032):
        """Calculate RDF using full volume normalisation.
        
        Uses proximity of both central and surrounding molecules for 
        volume correction. Pass in data on a per-frame basis.
        
        Args:
            pos_a (ndarray): Positions of central atoms (e.g., carbon).
            pos_b (ndarray): Positions of surrounding atoms (e.g., oxygen).
            WC (ndarray): WC interface coordinates.
            boxdim (ndarray): Box dimensions for PBC.
            depth (list): [min, max] depth range for interface proximity.
            dr (float): Bin width for histogram.
            crit_dens (float): Critical density for normalisation.
            
        Returns:
            tuple: (bin_edges, rdf_values) or ([0, 0], [0, 0]) if outside depth range.
        """
        dens = Density(self._u)
        prox_c = dens.proximity(WC,pos_a,boxdim,result='mag')[0] # determine which side of interface

        # Check if molecule is within specified depth range
        if not (depth[0] < prox_c < depth[1]):
            return ([0, 0], [0, 0])

        dist_mat = distance_array(pos_a, pos_b, box=boxdim)
        dists = dist_mat.flatten()

        prox_h2o = np.array(dens.proximity(WC,pos_b,boxdim,result='mag'))
        proxies = prox_h2o.flatten()

        # obtain the bins
        area = boxdim[0]*boxdim[1]
        bins = np.arange(0.1,np.sqrt(area),dr)

        # obtain the volume normalization
        weights = self.get_volume_normalization(dists,prox_c,proxies)

        # calculate the histogram
        hist, edges = np.histogram(dists, weights=1/weights, bins=bins, density=False)  # Changed dens to hist
        edges = edges[:-1]
        
        rdf = hist/dr/crit_dens  # Changed dens to hist

        return (edges,rdf)










