from xpart.longitudinal import generate_longitudinal_coordinates
import numpy as np
from ..general import _print

def generate_parabolic_longitudinal_coordinates(num_particles=None,
									    		nemitt_x=None, 
												nemitt_y=None, 
												sigma_z=None,
												particle_ref=None, 
												tracker=None,
												line=None,
												return_matcher=False
												):

	"""
	Function to generate a parabolic longitudinal distribution 

	Parameters:
	-----------
	num_particles : int
		number of macroparticles
	nemitt_x : float
		normalized horizontal emittance in m rad
    nemitt_y : float
		normalized vertical emittance in m rad
	sigma_z : float
		bunch length in meters
	particle_ref : xp.particle
		reference particle
	tracker : xt.tracker
	line: xt.line
	return_matcher : bool
		whether to also return xp.SingleRFHarmonicMatcher object

	Returns:
	-------- 
	zeta : np.ndarray
		longitudinal coordinates zeta for particles
	delta : np.ndarray
		relative momentum offset coordinates for particles
	matcher : xp.SingleRFHarmonicMatcher
		RF matcher object
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
	
	# Generate longitudinal coordinates s
	zeta, delta, matcher = generate_longitudinal_coordinates(line=line, distribution='parabolic', 
							num_particles=num_particles, 
							engine='single-rf-harmonic', sigma_z=sigma_z,
							particle_ref=particle_ref, return_matcher=True)
	
	if return_matcher:
		return zeta, delta, matcher
	else:
		return zeta, delta
