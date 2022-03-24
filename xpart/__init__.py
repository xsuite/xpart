from .general import _pkg_root
from .particles import Particles, gen_local_particle_api, pmass

from .build_particles import build_particles
from .matched_gaussian import generate_matched_gaussian_bunch

from .linear_normal_form import compute_linear_normal_form

from .transverse_generators import generate_2D_polar_grid
from .transverse_generators import generate_2D_uniform_circular_sector
from .transverse_generators import generate_2D_pencil
from .transverse_generators import generate_2D_gaussian

from .longitudinal import generate_longitudinal_coordinates

from .constants import PROTON_MASS_EV, ELECTRON_MASS_EV

from .monitors import PhaseMonitor

def enable_pyheadtail_interface():
    import xpart.pyheadtail_interface.pyhtxtparticles as pp
    import xpart as xp
    xp.Particles = pp.PyHtXtParticles

def disable_pyheadtail_interface():
    import xpart as xp
    xp.Particles = xp.particles.Particles
