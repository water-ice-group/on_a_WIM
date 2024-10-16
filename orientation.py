import numpy as np
from scipy import stats
from density import Density
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from MDAnalysis.transformations.wrap import wrap,unwrap
from MDAnalysis.lib import distances
from utilities import AtomPos


class Orientation:
    
    def __init__(self, universe, 
                 **kwargs):
        self._u = universe
    
    def _getCosTheta(self,ox,h1,h2,wc,boxdim,vector):

        center = boxdim[:3]/2
        # vect1 = distances.apply_PBC(h1-ox+center,box=boxdim)
        # vect2 = distances.apply_PBC(h2-ox+center,box=boxdim)

        vect1 = distances.minimize_vectors(h1-ox,boxdim)
        vect2 = distances.minimize_vectors(h2-ox,boxdim)

        dipVector0 = (vect1 + vect2) * 0.5 # - center # map the dipole

        dens = Density(self._u)
        dist,surf_vect = dens.proximity(wc,ox,boxdim,result='both',cutoff=False)

        if vector=='z':
            cosTheta = [np.dot(dipVector0[i],[0,0,1])/np.linalg.norm(dipVector0[i]) for i in range(len(dist))]
        elif vector=='WC':
            cosTheta = [np.dot(dipVector0[i],surf_vect[i])/((np.linalg.norm(surf_vect[i]))*np.linalg.norm(dipVector0[i])) for i in range(len(dist))]

        return (dist,cosTheta)

    # redacted feature
    '''elif property=='bond':
        #vect1 = distances.apply_PBC(h1-ox,box=boxdim)
        vect1 = np.subtract(h1,ox)
        unitvect = ( vect1 / np.linalg.norm(vect1, axis=1)[:, None] )

    dens = Density(self._u)
    dist,surf_vect = dens.proximity(wc,ox,boxdim,upper=upper_z,result='both')

    if vector=='WC':
        cosTheta = [np.dot(unitvect[i],surf_vect[i])/dist[i] for i in range(len(dist))]
    elif vector=='z':
        cosTheta = [np.dot(unitvect[i],[0,0,-1])/dist[i] for i in range(len(dist))]
    
    return (np.array(dist),np.array(cosTheta))'''
        
    
    def _getCosTheta_Carbon(self,c,oc1,oc2,wc,boxdim,vector,property='dipole'):

        if property=='dipole':
            center = boxdim[:3]/2
            # vect1 = distances.apply_PBC(oc1-c+center,box=boxdim)
            # vect2 = distances.apply_PBC(oc2-c+center,box=boxdim)

            vect1 = distances.minimize_vectors(oc1-c,boxdim)
            vect2 = distances.minimize_vectors(oc2-c,boxdim)

            dipVector0 = (vect1 + vect2) * 0.5 # - center # map the dipole

            dens = Density(self._u)
            dist,surf_vect = dens.proximity(wc,c,boxdim,result='both',cutoff=False)

            if vector=='z':
                cosTheta = [np.dot(dipVector0[i],[0,0,1])/np.linalg.norm(dipVector0[i]) for i in range(len(dist))]
            elif vector=='WC':
                cosTheta = [np.dot(dipVector0[i],surf_vect[i])/((np.linalg.norm(surf_vect[i]))*np.linalg.norm(dipVector0[i])) for i in range(len(dist))]

            return (dist,cosTheta)
        
        elif property=='bond':
            center = boxdim[:3]/2

            # init_1 = distances.apply_PBC(oc1-c+center,box=boxdim)
            # init_2 = distances.apply_PBC(oc2-c+center,box=boxdim)
            # vect1 = init_1 - center
            # vect2 = init_2 - center

            vect1 = distances.minimize_vectors(oc1-c,boxdim)
            vect2 = distances.minimize_vectors(oc2-c,boxdim)

            #test_1 = oc1-c
            #print(f'Normal {test_1[-5]}')
            #print(f'Wrapped {vect1[-5]}')

            dens = Density(self._u)
            dist,surf_vect = dens.proximity(wc,c,boxdim,result='both',cutoff=False)

            if vector=='z':
                cosTheta_1 = [np.dot(vect1[i],[0,0,1])/np.linalg.norm(vect1[i]) for i in range(len(dist))]
                cosTheta_2 = [np.dot(vect2[i],[0,0,1])/np.linalg.norm(vect2[i]) for i in range(len(dist))]
                
            elif vector=='WC':
                cosTheta_1 = [np.dot(vect1[i],surf_vect[i])/((np.linalg.norm(surf_vect[i]))*np.linalg.norm(vect1[i])) for i in range(len(dist))]
                cosTheta_2 = [np.dot(vect2[i],surf_vect[i])/((np.linalg.norm(surf_vect[i]))*np.linalg.norm(vect2[i])) for i in range(len(dist))]

            dist_out = dist + dist
            cosTheta_out = cosTheta_1 + cosTheta_2

            return (dist_out,cosTheta_out)

    # redacted feature
    '''elif property=='bond':
        center = boxdim[:3]/2
        vect1 = distances.apply_PBC(oc1-c+center,box=boxdim)
        vect2 = distances.apply_PBC(oc2-c+center,box=boxdim) 
        #dipVector0 = (vect1 + vect2) * 0.5 - center # map the dipole
        dipVector0 = vect1 - center # map the bond angle
        unitvect = ( dipVector0/ np.linalg.norm(vect1, axis=1)[:, None] )

        dens = Density(self._u)
        dist,surf_vect = dens.proximity(wc,c,boxdim,upper=upper_z,result='both')

        if vector=='WC':
            cosTheta = [np.dot(unitvect[i],surf_vect[i])/dist[i] for i in range(len(dist))]
        elif vector=='z':
            cosTheta = [np.dot(unitvect[i],[0,0,-1])/dist[i] for i in range(len(dist))]

        return (np.array(dist),np.array(cosTheta))'''

    def _getHistogram(self, dist, cosThetra, bins=200,hist_range=[-20,10]):

        means, edges, binnumber = stats.binned_statistic(dist[:].flatten(),
                                                         cosThetra[:].flatten(),
                                                         statistic='mean', bins=bins,
                                                         range=hist_range)
        
        counts, edges, binnumber = stats.binned_statistic(dist[:].flatten(),
                                                         cosThetra[:].flatten(),
                                                         statistic='count', bins=bins,
                                                         range=hist_range)
        

        final = []
        for i in range(len(counts)):
            if counts[i] > max(counts)*0.2:
                final.append(means[i])
            else:
                final.append(0)

        edges = 0.5 * (edges[1:] + edges[:-1])
        hist = np.array([edges, final])
        hist = hist.transpose()
        return (hist)
    

    def _getHeatMap(self,dist, cosThetra, bins=200,hist_range=[-10,10]):
        hist, x_edges, y_edges = np.histogram2d(dist,cosThetra, bins=50,density=True,
                                        range=[hist_range,[-1,1]]
                                        )
        
        return (hist,x_edges,y_edges)
    

    

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


    