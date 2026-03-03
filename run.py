import MDAnalysis as mda
from WillardChandler import WillardChandler

# IMPORTANT THAT SYSTEM IS CENTERED WRT WATER BEFORE LOADING TRAJ. 
# WATER OXYGENS MUST BE LABELLED 'OW' IN PDB FILE.

pdb = './ref_coords.pdb'
trj = './test.dcd'

u = mda.Universe(pdb,trj)


# load traj
# generate interface 
# -------------------------------------------------

WC_inter = WillardChandler(u,lower_z=5,upper_z=95)
result = WC_inter.generate(grid=400,new_inter=True)
WC_inter.save()

# density analysis
data_Oxygen = WC_inter.Density_run('OW',400,-10,10)
WC_inter.Density_plot(data_Oxygen)

# orientational analysis
ori_water = WC_inter.Orientation_run('water','time',60,-8,2,vect='WC')
WC_inter.Orientation_plot(ori_water)

# hbond analysis
result = WC_inter.Hbonds_run_water(bins=100,lower=-8,upper=2)
WC_inter.HBondz_plot()
