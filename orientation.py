import numpy as np
from scipy import stats
from density import Density
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from MDAnalysis.transformations.wrap import wrap,unwrap 
from utilities import AtomPos


class Orientation:
    
    def __init__(self, universe, 
                 **kwargs):
        self._u = universe
    
    def _getCosTheta(self,ox,h1,h2,wc):

        vect1 = np.subtract(h1,ox)
        vect2 = np.subtract(h2,ox)
        mid = - np.add(vect1,vect2)/2
        unitvect = ( mid / np.linalg.norm(mid, axis=1)[:, None] )

        dens = Density(self._u)
        dist,surf_vect = dens.proximity(wc,ox,'both')

        cosTheta = [np.dot(unitvect[i],surf_vect[i])/dist[i] for i in range(len(dist))]
        
        return (np.array(dist),np.array(cosTheta))


    def _getHistogram(self, dist, cosThetra, bins=200,hist_range=[-20,10]):

        means, edges, binnumber = stats.binned_statistic(dist[:].flatten(),
                                                         cosThetra[:].flatten(),
                                                         statistic='mean', bins=bins,
                                                         range=hist_range)
        
        counts, edges, binnumber = stats.binned_statistic(dist[:].flatten(),
                                                         cosThetra[:].flatten(),
                                                         statistic='count', bins=bins,
                                                         range=hist_range)
        
        print(counts)

        final = []
        for i in range(len(counts)):
            if counts[i] > max(counts)*0.1:
                final.append(means[i])
            else:
                final.append(0)

        edges = 0.5 * (edges[1:] + edges[:-1])
        hist = np.array([edges, final])
        hist = hist.transpose()
        return (hist)
    


    

def oriPlot(hist,smooth=2):
    dist = []
    plot = []
    for i in hist:
        dist.append(i[0])
        plot.append(i[1])
        
        

    zeros = [0]*len(dist)
    fig, ax = plt.subplots()
    ax.plot(dist[::smooth],plot[::smooth],'blue')
    ax.fill_between(dist[::smooth],zeros[::smooth],plot[::smooth],
                    color='blue',
                    alpha=0.2)
    ax.set_xlabel('Distance / $\mathrm{\AA}$')
    ax.set_ylabel(r'P<cos($\theta$)>')
    plt.savefig('./outputs/orientation.pdf',dpi=400,bbox_inches='tight',facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
