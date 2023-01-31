# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import scipy.constants as sc

PROTON_MASS_EV = sc.m_p *sc.c**2 /sc.e
ELECTRON_MASS_EV = sc.m_e * sc.c**2 /sc.e
MUON_MASS_EV = sc.physical_constants['muon mass'][0] * sc.c**2 /sc.e
