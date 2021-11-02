from .particles import Particles, gen_local_particle_api
from .gaussian import generate_matched_gaussian_bunch
from .general import _pkg_root

def enable_pyheadtail_interface():
    import xpart.pyheadtail_interface.pyhtxtparticles as pp
    import xpart as xt
    xp.Particles = pp.PyHtXtParticles
