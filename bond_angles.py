import numpy as np
from scipy import stats
from density import Density
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from MDAnalysis.transformations.wrap import wrap,unwrap 
from utilities import AtomPos


class BondAngle:

    def __init__(self,universe,**kwargs):
        self._u = universe






