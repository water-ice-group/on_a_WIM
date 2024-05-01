
import MDAnalysis as mda
from WillardChandler import WillardChandler

# Willard-Chandler interfacial module adapted for modelling NPT
# consider only a single interface of a centred system. 
# IMPORTANT THAT SYSTEM IS CENTERED WRT WATER BEFORE LOADING TRAJ. 

pdb = '../ref_coords.pdb'
trj = '../centered.dcd'

u = mda.Universe(pdb,trj)


# load traj
# generate interface 
# -------------------------------------------------

WC_inter = WillardChandler(u,upper_z=25)
result = WC_inter.generate(grid=400,new_inter=True)
WC_inter.save()

# -------------------------------------------------
# density analysis
# data_Oxygen = WC_inter.Density_run('OW',400,-10,10)
# data_Carbon = WC_inter.Density_run('C',400,-10,10)

# density plot
#WC_inter.Density_plot(data_Oxygen,data_Carbon)


# # orientational analysis
# ori_water = WC_inter.Orientation_run('water',200,-15,15)
# ori_carbon = WC_inter.Orientation_run('carbon',200,-15,15)

# # orientational plot
# WC_inter.Orientation_plot(ori_water,ori_carbon)


# # hbond analysis
# result = WC_inter.Hbonds_run(lower=-15,upper=15)

# # plot of the hbond analysis
# WC_inter.HBondz_plot()


# # save coordinates of WC for visualisation
# # WC_inter.visualise()
