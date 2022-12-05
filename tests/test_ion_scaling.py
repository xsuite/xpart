# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import json
import NAFFlib

import xobjects as xo
import xpart as xp
import xtrack as xt
import xfields as xf

test_data_folder = xt._pkg_root.joinpath('../test_data').absolute()

def test_ion_scaling_tracking():
    """
    Test if the Q^2/A = 1 scaling holds versus a proton reference case 
    for some linearly increasing values of Q^2 and A
    """
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        ########## Test settings ##################
        bunch_intensity = 1e11/3  # Need short bunch to avoid bucket non-linearity
        sigma_z = 22.5e-2/3
        nemitt_x = 2.5e-6
        nemitt_y = 2.5e-6
        num_particles = 1000
        num_turns = 30
        
        num_spacecharge_interactions = 540
        tol_spacecharge_position = 1e-2
        
        ########## Load the SPS proton sequence and ref particle #####################
        fname_line =  test_data_folder.joinpath('sps_w_spacecharge/line_no_spacecharge_and_particle.json')
        with open(fname_line, 'r') as fid:
             input_data = json.load(fid)
        
        # Load line for tracking and reference particle 
        line = xt.Line.from_dict(input_data['line'])
        particle_ref = xp.Particles.from_dict(input_data['particle'])
        ctx2arr = context.nparray_from_context_array  # Copies an array to the device to a numpy array.
        
        # Define longitudinal profile
        lprofile = xf.LongitudinalProfileQGaussian(
            number_of_particles=bunch_intensity,
            sigma_z=sigma_z,
            z0=0.,
            q_parameter=1.)
            
        ########### PROTON REFERENCE CASE #############
        # Install space charge 
        xf.install_spacecharge_frozen(line=line,
                           particle_ref=particle_ref,
                           longitudinal_profile=lprofile,
                           nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                           sigma_z=sigma_z,
                           num_spacecharge_interactions=num_spacecharge_interactions,
                           tol_spacecharge_position=tol_spacecharge_position)
        
        
        # Define tracker and particle beam 
        tracker = xt.Tracker(_context=context,
                            line=line)
        tracker_sc_off = tracker.filter_elements(exclude_types_starting_with='SpaceCh')
        
        particles0 = xp.generate_matched_gaussian_bunch(_context=context,
                 num_particles=num_particles, total_intensity_particles=bunch_intensity,
                 nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
                 particle_ref=particle_ref, tracker=tracker_sc_off)
        
        # Set up for the tracking 
        tracker.optimize_for_tracking()
        x_tbt = np.zeros((num_particles, num_turns), dtype=np.float64)
        y_tbt = np.zeros((num_particles, num_turns), dtype=np.float64)
        Qx0 = np.zeros(num_particles)
        Qy0 = np.zeros(num_particles)
        
        # Perform tracking
        for ii in range(num_turns):
            print(f'Turn: {ii}\n', end='\r', flush=True)
            x_tbt[:, ii] = ctx2arr(particles0.x[:num_particles]).copy()
            y_tbt[:, ii] = ctx2arr(particles0.y[:num_particles]).copy()
            tracker.track(particles0)
        
        # Find maximum tune shift
        for i_part in range(num_particles):
            Qx0[i_part] = NAFFlib.get_tune(x_tbt[i_part, :])
            Qy0[i_part] = NAFFlib.get_tune(y_tbt[i_part, :])
        
        Qx0_max = np.max(Qx0)
        Qy0_max = np.max(Qy0)
        
        ###################################################
        
        ############ ION SCALING TRACKING #################
        proton_ref_momentum =  25.92e9  
        As = np.array([4., 9., 25., 100., 144.])  # mass numbers 
        Qs = np.array([2., 3., 5., 10., 12.])     # charge states 
        Qx_max_array = []
        Qy_max_array = []
        
        for jj in range(len(As)):
            print("\nIon scaling nr {}\n".format(jj+1))
            x_tbt_ion = np.zeros((num_particles, num_turns), dtype=np.float64)
            y_tbt_ion = np.zeros((num_particles, num_turns), dtype=np.float64)
            Qx = np.zeros(num_particles)
            Qy = np.zeros(num_particles)
        
            
            # Modify reference particle according to new masses
            mass_ion = As[jj]*931494102.42 # atomic mass unit-electron volt relationship
            charge_ion = Qs[jj]
            
            line_ion = xt.Line.from_dict(input_data['line'])
            
            particle_sample = xp.Particles(
                                mass0 = mass_ion, 
                                q0= charge_ion, 
                                p0c = proton_ref_momentum*As[jj]
                                )
            
            ########### INSTALL SPACE CHARGE FOR IONS #################
            xf.install_spacecharge_frozen(line=line_ion,
                               particle_ref=particle_sample,
                               longitudinal_profile=lprofile,
                               nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                               sigma_z=sigma_z,
                               num_spacecharge_interactions=num_spacecharge_interactions,
                               tol_spacecharge_position=tol_spacecharge_position)
            
            
            ############ DEFINE TRACKER AND PARTICLE BEAM ######################
            tracker_ion = xt.Tracker(_context=context,
                                line=line_ion)
            tracker_sc_off_ion = tracker_ion.filter_elements(exclude_types_starting_with='SpaceCh')
            
            particles_ion = xp.generate_matched_gaussian_bunch(_context=context,
                     num_particles=num_particles, total_intensity_particles=bunch_intensity,
                     nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
                     particle_ref=particle_sample, tracker=tracker_sc_off_ion)
            
            
            ########### TRACK THE PARTICLES ################
            tracker_ion.optimize_for_tracking()
            
            # Perform tracking
            for ii in range(num_turns):
                print(f'Turn: {ii}\n', end='\r', flush=True)
                x_tbt_ion[:, ii] = ctx2arr(particles_ion.x[:num_particles]).copy()
                y_tbt_ion[:, ii] = ctx2arr(particles_ion.y[:num_particles]).copy()
                tracker_ion.track(particles_ion)
        
            # Find maximum tune shift
            for i_part in range(num_particles):
                Qx[i_part] = NAFFlib.get_tune(x_tbt_ion[i_part, :])
                Qy[i_part] = NAFFlib.get_tune(y_tbt_ion[i_part, :])
        
            # Remove any possible outliers due to resonances 
            Qx = Qx[(Qx < 1.5*Qx0_max) & (Qx > 0.5*np.min(Qx0))]
            Qy = Qy[(Qy < 1.5*Qy0_max) & (Qy > 0.5*np.min(Qy0))]
        
            # Check maximum tune shift
            Qx_max = np.max(Qx)
            Qy_max = np.max(Qy)
            Qx_max_array.append(Qx_max)
            Qy_max_array.append(Qy_max)
            print("Ion type {}:  dQx = {},  dQy = {}".format(jj+1, Qx_max, Qy_max))
            print("Proton:      dQx = {},  dQy = {}".format(Qx0_max, Qy0_max))
            
            assert np.isclose(Qx_max, Qx0_max, atol=1e-2)


    
    

    
