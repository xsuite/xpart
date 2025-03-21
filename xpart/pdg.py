# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from warnings import warn

# This file is deprecated and will be removed in the future.


def __getattr__(name):
    if name.startswith('__'):
        # Needed for module resolution, i.e. for the below errors to actually
        # work properly
        raise AttributeError

    warn("The functionality in xpart.pdg moved to xtrack.particles.pdg. Using it "
      + f"via xpart is still possible, but deprecated. It will be removed in the "
      + f"future.", FutureWarning)
    import xtrack.particles.pdg as pdg
    if name in pdg.__dict__:
        return pdg.__dict__[name]
    else:
        raise AttributeError(f"module 'xtrack.particles.pdg' has no " \
                           + f"attribute '{name}'")

