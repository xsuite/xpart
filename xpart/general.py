# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from pathlib import Path

_pkg_root = Path(__file__).parent.absolute()

class Print():
    suppress = False

    def __call__(self, *args, **kwargs):
        if not self.suppress:
            print(*args, **kwargs)

_print = Print()