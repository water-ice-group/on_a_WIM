#%%
import MDAnalysis as mda
from WillardChandler import WillardChandler
import matplotlib.pyplot as plt

pdb = 'ref_coords.pdb'
trj = 'test.dcd'

u = mda.Universe(pdb)
dimensions = u.dimensions
u = mda.Universe(pdb, trj)
u.dimensions = dimensions
u.add_TopologyAttr('charges')



# load module
# calculate 
# -------------------------------------------------

WC_inter = WillardChandler(u,endstep=500)
result = WC_inter.generate(grid=400)

# -------------------------------------------------

#%%
# density analysis

data_Oxygen = WC_inter.Density_run('OW',500,-15,10)
data_Carbon = WC_inter.Density_run('C',500,-15,10)

#%%
# density plot

WC_inter.Density_plot(data_Oxygen,data_Carbon)

#%%
# orientational analysis

result = WC_inter.Orientation_run(200,-20,-1)

#%%
# orientational plot

WC_inter.Orientation_plot()


#%%
# hbond analysis

result = WC_inter.Hbonds_run()

#%%
# plot of the hbond analysis

fig, ax = plt.subplots()
ax.plot(result[1],result[0])
ax.set_xlabel('Distance / $\mathrm{\AA}$')
ax.set_ylabel('HBond count')
ax.set_xlim(-15,0)
ax.set_ylim(0,4)
plt.savefig('hbond_profile.pdf',dpi=400,bbox_inches='tight',facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()

#%%
# save coordinates of WC for visualisation

WC_inter.visualise()
