# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

def __getattr__(name):
    if name.startswith('__'):
        # Needed for module resolution, i.e. for the below errors to actually
        # work properly
        raise AttributeError

    print("Warning: xpart.pdg moved to xtrack.particles.pdg.")
    import xtrack.particles.pdg as pdg
    if name in pdg.__dict__:
        return pdg.__dict__[name]
    else:
        raise AttributeError(f"module 'xtrack.particles.pdg' has no " \
                           + f"attribute '{name}'")

