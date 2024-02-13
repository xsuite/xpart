# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .general import _pkg_root, _print
from .particles import Particles, ParticlesBase, reference_from_pdg_id

from .constants import PROTON_MASS_EV, ELECTRON_MASS_EV, MUON_MASS_EV, Pb208_MASS_EV
pmass = PROTON_MASS_EV  # backwards compatibility
from .pdg import get_pdg_id_from_name, get_name_from_pdg_id

from .build_particles import build_particles
from .matched_gaussian import generate_matched_gaussian_bunch

from .transverse_generators import generate_2D_polar_grid
from .transverse_generators import generate_2D_uniform_circular_sector
from .transverse_generators import generate_2D_pencil
from .transverse_generators import generate_2D_pencil_with_absolute_cut
from .transverse_generators import generate_2D_gaussian

from .longitudinal import generate_longitudinal_coordinates

from .monitors import PhaseMonitor

from ._version import __version__

def enable_pyheadtail_interface():
    import xpart.pyheadtail_interface.pyhtxtparticles as pp
    import xpart as xp
    import xtrack as xt
    xp.Particles = pp.PyHtXtParticles
    xt.Particles = pp.PyHtXtParticles


def disable_pyheadtail_interface():
    import xpart as xp
    import xtrack as xt
    xp.Particles = xp.particles.Particles
    xt.Particles = xp.particles.Particles


