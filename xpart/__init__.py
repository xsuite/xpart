# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from xtrack.particles import Particles, PROTON_MASS_EV, ELECTRON_MASS_EV, MUON_MASS_EV, Pb208_MASS_EV, reference_from_pdg_id, enable_pyheadtail_interface, disable_pyheadtail_interface
pmass = PROTON_MASS_EV  # backwards compatibility

from .pdg import get_pdg_id_from_name, get_name_from_pdg_id

from .build_particles import build_particles
from .matched_gaussian import (generate_matched_gaussian_bunch,
                               generate_matched_gaussian_multibunch_beam,
                               )

from .transverse_generators import generate_2D_polar_grid
from .transverse_generators import generate_2D_uniform_circular_sector
from .transverse_generators import generate_2D_pencil
from .transverse_generators import generate_2D_pencil_with_absolute_cut
from .transverse_generators import generate_2D_gaussian

from .longitudinal import generate_longitudinal_coordinates
from .longitudinal.generate_longitudinal import _characterize_line

from .monitors import PhaseMonitor

from ._version import __version__
