import MDAnalysis as mda
from WillardChandler import WillardChandler

pdb = 'ref_coords.pdb'
trj = 'test.dcd'

u = mda.Universe(pdb)
dimensions = u.dimensions
u = mda.Universe(pdb, trj)
u.dimensions = dimensions
u.add_TopologyAttr('charges')


# load traj
# generate interface 
# -------------------------------------------------

WC_inter = WillardChandler(u)
result = WC_inter.generate(grid=400)

# -------------------------------------------------
#%%

# density analysis
data_Oxygen = WC_inter.Density_run('OW',500,-15,15)
data_Carbon = WC_inter.Density_run('C',500,-15,15)

# density plot
WC_inter.Density_plot(data_Oxygen,data_Carbon)


# # orientational analysis
# result = WC_inter.Orientation_run(200,-10,1)

# # orientational plot
# WC_inter.Orientation_plot()


# # hbond analysis
# result = WC_inter.Hbonds_run(lower=-15,upper=10)

# # plot of the hbond analysis
# WC_inter.HBondz_plot()


# # save coordinates of WC for visualisation
# WC_inter.visualise()
