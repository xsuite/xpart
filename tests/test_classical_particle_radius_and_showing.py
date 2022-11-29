# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import io
import sys
import json 

import xobjects as xo
import xpart as xp
import xtrack as xt

test_data_folder = xt._pkg_root.joinpath('../test_data').absolute()

def test_classical_particle_radius_ions():
    """
    Test classical particle radius for ions 
    """
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        
        # Load SPS ion sequence with Pb82 ions  
        fname_line =  test_data_folder.joinpath('sps_ions/line_with_particle.json') 
        with open(fname_line, 'r') as fid:
             input_data = json.load(fid)
        
        particle_ref = xp.Particles.from_dict(input_data['particle'])
        r0_Pb82 = np.float64(4.998617e-17)  # calculated theoretical classical particle radius 
           
        assert np.isclose(particle_ref.get_classical_particle_radius0(), r0_Pb82, atol=1e-5)
        

def test_classical_particle_radius_protons():
    """
    Test classical particle radius for protons
    """
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        
        # Load SPS sequence without space charge 
        fname_line =  test_data_folder.joinpath('sps_w_spacecharge/line_no_spacecharge_and_particle.json') 
        with open(fname_line, 'r') as fid:
             input_data = json.load(fid)
        
        particle_ref = xp.Particles.from_dict(input_data['particle'])
        r0_proton = np.float64(1.534698e-18)  # calculated theoretical classical particle radius 
           
        assert np.isclose(particle_ref.get_classical_particle_radius0(), r0_proton, atol=1e-5)
        

def test_showing():
    """
    Test whether showing particle values gives reasonable output
    """
    test_data_folder = xt._pkg_root.joinpath('../test_data').absolute()

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        
        # Load SPS ion sequence with Pb82 ions  
        fname_line =  test_data_folder.joinpath('sps_ions/line_with_particle.json') 
        with open(fname_line, 'r') as fid:
             input_data = json.load(fid)
        
        particle_ref = xp.Particles.from_dict(input_data['particle'])
        particle_ref.show()
