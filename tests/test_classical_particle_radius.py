# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import json 

import xobjects as xo
import xpart as xp

def test_classical_particle_radius_ions():
    """
    Test classical particle radius for ions 
    """
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        
        # Load SPS ion sequence with Pb82 ions  
        fname_line = ('../../xtrack/test_data/sps_ions/line.json')
        with open(fname_line, 'r') as fid:
             input_data = json.load(fid)
        
        particle_ref = xp.Particles.from_dict(input_data['particle'])
        r0_Pb82 = 4.9986172550871315e-17  # calculated theoretical classical particle radius 
        
        assert np.isclose(particle_ref.classical_particle_radius0, r0_Pb82, rtol=1e-05)