# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

def __getattr__(name):
    if name.startswith('__'):
        # Needed for module resolution, i.e. for the below errors to actually
        # work properly
        raise AttributeError

    if name == 'ParticlesPurelyLongitudinal':
        raise ModuleNotFoundError(
            "ParticlesPurelyLongitudinal class has been removed. Please track "
            "the usual Particles class with the config flags `FREEZE_VAR_x`, "
            "etc. set on the line instead."
        )

    raise ModuleNotFoundError(
        "It seems you are trying to import a Particles related object from "
        "xpart.particles. Please note that the Particles class has been moved "
        "to Xtrack, and can be imported as xtrack.Particles."
    )
