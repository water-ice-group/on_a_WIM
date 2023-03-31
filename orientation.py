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
    
    def _getCosTheta(self,ox,h1,h2,wc,opos):

        vect1 = np.subtract(h1,ox)
        vect2 = np.subtract(h2,ox)
        mid = - np.add(vect1,vect2)/2
        unitvect = ( mid / np.linalg.norm(mid, axis=1)[:, None] )

        dens = Density(self._u)
        dist,surf_vect = dens.proximity(wc,opos,'both')

        cosTheta = [np.dot(unitvect[i],surf_vect[i])/dist[i] for i in range(len(dist))]
        
        return (np.array(dist),np.array(cosTheta))
    
    def _getCosTheta_Carbon(self,c,oc,wc,cpos):

        if len(oc) < 3:
            vect = np.subtract(oc[0],c)
        else:
            vect = np.subtract(oc,c)

        unitvect = ( vect / np.linalg.norm(vect, axis=1)[:, None] )
        dens = Density(self._u)
        dist,surf_vect = dens.proximity(wc,cpos,'both')
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
    


    

def oriPlot(data_Oxygen,data_Carbon,lower=-15,upper=15,smooth=2):
    dist = []
    dist_C = []
    plot = []
    plot_C = []

    for i in data_Oxygen:
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
    ax.set_xlim(lower,upper)

    if data_Carbon is not None:
        for i in data_Carbon:
            dist_C.append(i[0])
            plot_C.append(i[1])
        zeros_C = [0]*len(dist_C)
        ax.plot(dist_C[::smooth],plot_C[::smooth],'black')
        ax.fill_between(dist_C[::smooth],zeros_C[::smooth],plot_C[::smooth],
                color='black',
                alpha=0.2)

    plt.savefig('./outputs/orientation.pdf',dpi=400,bbox_inches='tight',facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()


    