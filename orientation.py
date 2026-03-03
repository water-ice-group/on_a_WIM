import numpy as np
from scipy import stats
from density import Density
import matplotlib.pyplot as plt
from MDAnalysis.lib import distances


class Orientation:
    
    def __init__(self, universe):
        self._u = universe
    
    def _getCosTheta(self,ox,h1,h2,wc,boxdim):
        """Calculate cos(theta) for water dipoles relative to WC interface.
        
        Args:
            ox (ndarray): Oxygen atom positions.
            h1 (ndarray): First hydrogen positions.
            h2 (ndarray): Second hydrogen positions.
            wc (ndarray): WC interface coordinates.
            boxdim (ndarray): Box dimensions for PBC.
            
        Returns:
            tuple: (distances, cos_theta) arrays.
        """
        vect1 = distances.minimize_vectors(h1-ox,boxdim)
        vect2 = distances.minimize_vectors(h2-ox,boxdim)
        dipVector0 = distances.minimize_vectors((vect1 + vect2) * 0.5,boxdim) 

        dens = Density(self._u)
        dist,surf_vect = dens.proximity(wc,ox,boxdim,result='both',cutoff=False)

        cosTheta = [np.dot(dipVector0[i],-surf_vect[i])/((np.linalg.norm(-surf_vect[i]))*np.linalg.norm(dipVector0[i])) for i in range(len(dist))]

        return (dist,cosTheta)


    def _getCosTheta_z(self,ox,h1,h2,boxdim):
        """Calculate cos(theta) for water dipoles relative to z-axis.
        
        Args:
            ox (ndarray): Oxygen atom positions.
            h1 (ndarray): First hydrogen positions.
            h2 (ndarray): Second hydrogen positions.
            boxdim (ndarray): Box dimensions for PBC.
            
        Returns:
            tuple: (z_positions, cos_theta) arrays.
        """
        vect1 = distances.minimize_vectors(h1-ox,boxdim)
        vect2 = distances.minimize_vectors(h2-ox,boxdim)
        dipVector0 = distances.minimize_vectors((vect1 + vect2) * 0.5,boxdim) 

        dist = [distances.apply_PBC(ox[i],boxdim)[2] for i in range(len(ox))]

        norm = [0,0,1]
        cosTheta = [np.dot(dipVector0[i], norm)/(np.linalg.norm(dipVector0[i])*np.linalg.norm(norm)) for i in range(len(dist))]

        return (dist,cosTheta)



        
    
    def _getCosTheta_Carbon(self,c,oc1,oc2,wc,boxdim,vector,property='dipole'):
        """Calculate cos(theta) for CO2 molecules.
        
        Args:
            c (ndarray): Carbon atom positions.
            oc1 (ndarray): First oxygen positions.
            oc2 (ndarray): Second oxygen positions.
            wc (ndarray): WC interface coordinates.
            boxdim (ndarray): Box dimensions for PBC.
            vector (str): Reference vector - 'z' or 'WC'.
            property (str): Property to compute - 'dipole' or 'bond'.
            
        Returns:
            tuple: (distances, cos_theta) arrays.
            
        Raises:
            ValueError: If unknown vector or property type.
        """
        if property=='dipole':

            vect1 = distances.minimize_vectors(oc1-c,boxdim)
            vect2 = distances.minimize_vectors(oc2-c,boxdim)

            dipVector0 = (vect1 + vect2) * 0.5

            dens = Density(self._u)
            dist,surf_vect = dens.proximity(wc,c,boxdim,result='both',cutoff=False)

            if vector=='z':
                cosTheta = [np.dot(dipVector0[i],[0,0,1])/np.linalg.norm(dipVector0[i]) for i in range(len(dist))]
            elif vector=='WC':
                cosTheta = [np.dot(dipVector0[i],surf_vect[i])/((np.linalg.norm(surf_vect[i]))*np.linalg.norm(dipVector0[i])) for i in range(len(dist))]
            else:
                raise ValueError(f"Unknown vector type: {vector}. Use 'z' or 'WC'.")

            return (dist, cosTheta)  # Return here for dipole case

        
        elif property=='bond':
            # center = boxdim[:3]/2  # Remove - unused

            vect1 = distances.minimize_vectors(oc1-c,boxdim)
            vect2 = distances.minimize_vectors(oc2-c,boxdim)

            dens = Density(self._u)
            dist,surf_vect = dens.proximity(wc,c,boxdim,result='both',cutoff=False)

            if vector=='z':
                cosTheta_1 = [np.dot(vect1[i],[0,0,1])/np.linalg.norm(vect1[i]) for i in range(len(dist))]
                cosTheta_2 = [np.dot(vect2[i],[0,0,1])/np.linalg.norm(vect2[i]) for i in range(len(dist))]
                
            elif vector=='WC':
                cosTheta_1 = [np.dot(vect1[i],surf_vect[i])/((np.linalg.norm(surf_vect[i]))*np.linalg.norm(vect1[i])) for i in range(len(dist))]
                cosTheta_2 = [np.dot(vect2[i],surf_vect[i])/((np.linalg.norm(surf_vect[i]))*np.linalg.norm(vect2[i])) for i in range(len(dist))]
            else:
                raise ValueError(f"Unknown vector type: {vector}. Use 'z' or 'WC'.")

            dist_out = dist + dist
            cosThet_out = cosTheta_1 + cosTheta_2

            return (dist_out, cosThet_out)  # Return here for bond case
        
        else:
            raise ValueError(f"Unknown property type: {property}. Use 'dipole' or 'bond'.")





    def _getHistogram(self, dist, cosThetra, bins=200,hist_range=None):
        """Generate binned histogram of orientation vs distance.
        
        Args:
            dist (ndarray): Distance values.
            cos_theta (ndarray): Cosine theta values.
            bins (int): Number of histogram bins.
            hist_range (list): [min, max] range for binning.
            
        Returns:
            ndarray: Histogram with columns [bin_centers, mean_cos_theta].
        """
        if hist_range is None:
            hist_range = [-20, 10]

        means, edges, _ = stats.binned_statistic(dist[:].flatten(),
                                                         cosThetra.flatten(),
                                                         statistic='mean', bins=bins,
                                                         range=hist_range)
        
        counts, _, _ = stats.binned_statistic(dist[:].flatten(),
                                                         cosThetra.flatten(),
                                                         statistic='count', bins=bins,
                                                         range=hist_range)
        

        count_threshold = np.max(counts) * 0.2
        filtered_means = np.where(counts > count_threshold, means, 0)

        bin_centers = 0.5 * (edges[1:] + edges[:-1])
        hist = np.column_stack([bin_centers, filtered_means])
        
        return hist




    def _getHeatMap(self, dist, cos_theta, bins=50, hist_range=None):
        """Generate 2D histogram heatmap of orientation vs distance.
        
        Args:
            dist (ndarray): Distance values.
            cos_theta (ndarray): Cosine theta values.
            bins (int): Number of bins in each dimension.
            hist_range (list): [min, max] range for distance axis.
            
        Returns:
            tuple: (histogram, x_edges, y_edges) arrays.
        """
        if hist_range is None:
            hist_range = [-10, 10]

        hist, x_edges, y_edges = np.histogram2d(
            dist, cos_theta, 
            bins=bins, 
            density=True,
            range=[hist_range, [-1, 1]]
        )
        
        return (hist, x_edges, y_edges)
    

    




def oriPlot(data_Oxygen,data_Carbon=None,lower=-15,upper=15,smooth=2):
    """Plot orientation profiles.
    
    Args:
        data_Oxygen (ndarray): Orientation data for water with columns [dist, cos_theta].
        data_Carbon (ndarray, optional): Orientation data for carbon species.
        lower (float): Lower x-axis limit.
        upper (float): Upper x-axis limit.
        smooth (int): Smoothing factor (take every Nth point).
        
    Raises:
        ValueError: If data_Oxygen is None.
    """
    if data_Oxygen is None:
        raise ValueError("data_Oxygen is None. Ensure Orientation_run() completed successfully.")

    dist = []
    plot = []

    for i in data_Oxygen:
        dist.append(i[0])
        plot.append(i[1])
    
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(dist[::smooth],plot[::smooth],'blue')
    ax.fill_between(dist[::smooth], 0, plot[::smooth],  # Use 0 instead of zeros list
                    color='blue',
                    alpha=0.2)
    ax.set_xlabel(r'Distance / $\mathrm{\AA}$')  # Add raw string prefix
    ax.set_ylabel(r'$\langle \cos(\theta) \rangle$')  # Fixed formatting
    ax.set_xlim(lower,upper)

    if data_Carbon is not None:
        dist_C = []
        plot_C = []
        for i in data_Carbon:
            dist_C.append(i[0])
            plot_C.append(i[1])
        ax.plot(dist_C[::smooth],plot_C[::smooth],'black')
        ax.fill_between(dist_C[::smooth], 0, plot_C[::smooth],  # Use 0 instead of zeros_C list
                color='black',
                alpha=0.2)

    plt.savefig('./outputs/orientation.pdf',dpi=400,bbox_inches='tight',facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()



