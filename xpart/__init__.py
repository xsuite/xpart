from .particles import Particles, gen_local_particle_api
from .gaussian import generate_matched_gaussian_bunch
from .general import _pkg_root
from .assemble_particles import assemble_particles

def enable_pyheadtail_interface():
    import xpart.pyheadtail_interface.pyhtxtparticles as pp
    import xpart as xp
    xp.Particles = pp.PyHtXtParticles
