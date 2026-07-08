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
	Generate longitudinal coordinates with a parabolic distribution.

	This is a convenience wrapper around `generate_longitudinal_coordinates`
	using `distribution='parabolic'` and `engine='single-rf-harmonic'`.

	Parameters
	----------
	num_particles : int
		Number of particles to generate.
	nemitt_x : float, optional
		Accepted for backward compatibility; not used by this function.
	nemitt_y : float, optional
		Accepted for backward compatibility; not used by this function.
	sigma_z : float
		RMS bunch length in m.
	particle_ref : xpart.Particles
		Reference particle.
	tracker : xtrack.Tracker, optional
		Deprecated. Use `line` instead.
	line : xtrack.Line
		Line used to infer the RF and optics parameters. The line must already
		have a tracker.
	return_matcher : bool, optional
		If True, also return the `SingleRFHarmonicMatcher` object.

	Returns
	-------
	zeta : np.ndarray
		Longitudinal position in m.
	delta : np.ndarray
		Relative momentum deviation.
	matcher : xpart.longitudinal.SingleRFHarmonicMatcher
		Matcher object used for the generation. Returned only when
		`return_matcher` is True.

	Example
	-------

	.. code-block:: python

		import numpy as np
		import xpart as xp
		import xtrack as xt
		from xpart.longitudinal import generate_parabolic_longitudinal_coordinates

		np.random.seed(12345)

		circumference = 26658.883
		line = xt.Line(elements=[
			xt.LineSegmentMap(
				length=circumference,
				betx=1.0, qx=0.31,
				bety=1.0, qy=0.32,
				longitudinal_mode='linear_fixed_rf',
				voltage_rf=16e6,
				frequency_rf=400.8e6,
				phase_rf=np.pi,
				slippage_length=circumference,
				momentum_compaction_factor=3.225e-4,
			)
		])
		line.set_particle_ref('proton', p0c=7e12)
		line.build_tracker()

		zeta, delta = generate_parabolic_longitudinal_coordinates(
			num_particles=4,
			sigma_z=0.02,
			particle_ref=line.particle_ref,
			line=line)

		zeta   # [-0.029114, -0.021342, -0.016575, -0.01217]
		delta  # [-4.490284e-05, -2.297977e-05, -3.434673e-05, -9.747943e-06]
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
