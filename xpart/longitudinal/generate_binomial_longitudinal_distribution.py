from xpart.longitudinal import generate_longitudinal_coordinates
from xpart import build_particles
import numpy as np
from ..general import _print

def binomial_longitudinal_distribution(_context=None, 
					num_particles=None,
                			nemitt_x=None, 
                			nemitt_y=None, 
                			sigma_z=None,
                			particle_ref=None, 
                			total_intensity_particles=None,
                			tracker=None,
                			line=None,
                			return_matcher=False,
                            m=None
                			):

	"""
	Function to generate a parabolic longitudinal distribution 
	"""
	
	if line is not None and tracker is not None:
		raise ValueError(
		    'line and tracker cannot be provided at the same time.')

	if tracker is not None:
		_print('Warning! '
		    "The argument tracker is deprecated. Please use line instead.")
		line = tracker.line

	if line is not None:
		assert line.tracker is not None, ("The line must have a tracker, "
		    "please call Line.build_tracker() first.")
	
	if num_particles is None:
		raise ValueError(
				'Number of particles must be provided')
	if sigma_z is None:
		raise ValueError(
				'Bunch length sigma_z must be provided')
				
	if particle_ref is None:
		raise ValueError(
				'Reference particle must be provided')
	
	# If emittances are not provided, set them to default value of one
	if nemitt_x is None:
		nemitt_x = 1.0
	
	if nemitt_y is None:
		nemitt_y = 1.0
        
	if m is None:
		m = 4.7 # typical value for ions at PS extraction

	# Generate longitudinal coordinates s
	zeta, delta, matcher = generate_longitudinal_coordinates(line=line, distribution='binomial', 
							num_particles=num_particles, 
							engine='single-rf-harmonic', sigma_z=sigma_z,
							particle_ref=particle_ref, return_matcher=True, m=m)
	
	# Initiate normalized coordinates 
	x_norm = np.random.normal(size=num_particles)
	px_norm = np.random.normal(size=num_particles)
	y_norm = np.random.normal(size=num_particles)
	py_norm = np.random.normal(size=num_particles)

	# If not provided, use number of particles as intensity 
	if total_intensity_particles is None:   
		total_intensity_particles = num_particles

	particles = build_particles(_context=None, particle_ref=particle_ref,
				zeta=zeta, delta=delta, 
				x_norm=x_norm, px_norm=px_norm,
				y_norm=y_norm, py_norm=py_norm,
				nemitt_x=nemitt_x, nemitt_y=nemitt_y,
				weight=total_intensity_particles/num_particles, line=line)

	if return_matcher:
		return particles, matcher
	else:
		return particles
