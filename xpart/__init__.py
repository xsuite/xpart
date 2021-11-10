from .particles import Particles, gen_local_particle_api, pmass
from .gaussian import generate_matched_gaussian_bunch
from .general import _pkg_root
from .build_particles import build_particles

def enable_pyheadtail_interface():
    import xpart.pyheadtail_interface.pyhtxtparticles as pp
    import xpart as xp
    xp.Particles = pp.PyHtXtParticles
