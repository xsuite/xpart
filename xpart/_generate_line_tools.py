# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""
Generate ``xpart/line_tools.py``.

The public ``line.xpart`` methods carry docstrings adapted from the
corresponding ``xpart`` functions. This script performs that adaptation once,
so importing ``xpart.line_tools`` does not need runtime docstring generation.
"""

import inspect
import pathlib
import re
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from xpart.build_particles import build_particles
from xpart.longitudinal import (
    generate_binomial_longitudinal_coordinates,
    generate_longitudinal_coordinates,
    generate_parabolic_longitudinal_coordinates,
    generate_qgaussian_longitudinal_coordinates,
)
from xpart.matched_gaussian import (
    generate_matched_gaussian_bunch,
    generate_matched_gaussian_multibunch_beam,
)
from xpart.transverse_generators import (
    generate_2D_gaussian,
    generate_2D_pencil,
    generate_2D_pencil_with_absolute_cut,
    generate_2D_polar_grid,
    generate_2D_uniform_circular_sector,
    generate_hypersphere_2D,
    generate_hypersphere_4D,
    generate_hypersphere_6D,
)


LINE_BOUND_METHODS = [
    ('generate_matched_gaussian_bunch', generate_matched_gaussian_bunch),
    ('generate_matched_gaussian_multibunch_beam',
     generate_matched_gaussian_multibunch_beam),
    ('generate_2D_pencil_with_absolute_cut',
     generate_2D_pencil_with_absolute_cut),
    ('generate_longitudinal_coordinates', generate_longitudinal_coordinates),
    ('generate_binomial_longitudinal_coordinates',
     generate_binomial_longitudinal_coordinates),
    ('generate_parabolic_longitudinal_coordinates',
     generate_parabolic_longitudinal_coordinates),
    ('generate_qgaussian_longitudinal_coordinates',
     generate_qgaussian_longitudinal_coordinates),
]

LINE_FREE_METHODS = [
    ('generate_2D_polar_grid', generate_2D_polar_grid),
    ('generate_2D_uniform_circular_sector',
     generate_2D_uniform_circular_sector),
    ('generate_2D_pencil', generate_2D_pencil),
    ('generate_2D_gaussian', generate_2D_gaussian),
    ('generate_hypersphere_2D', generate_hypersphere_2D),
    ('generate_hypersphere_4D', generate_hypersphere_4D),
    ('generate_hypersphere_6D', generate_hypersphere_6D),
]


def _insert_line_for_container_example(doc):
    if 'line.xpart.' not in doc or 'line = xt.Line' in doc:
        return doc

    doc = re.sub(
        r'(?m)^(\s*)import numpy as np\n\s*\n\s*import xtrack as xt$',
        r'\1import numpy as np\n\1import xtrack as xt',
        doc,
        count=1)

    lines = doc.splitlines()
    out = []
    in_code_block = False
    inserted = False

    for ii, line in enumerate(lines):
        out.append(line)

        if '.. code-block:: python' in line:
            in_code_block = True
            continue

        if not in_code_block or inserted:
            continue

        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            code_indent = line[:len(line) - len(line.lstrip())]
            next_line = lines[ii + 1] if ii + 1 < len(lines) else ''
            next_stripped = next_line.strip()
            if not next_stripped.startswith(('import ', 'from ')):
                out.extend([
                    '',
                    f'{code_indent}line = xt.Line(elements=[], element_names=[])',
                ])
                inserted = True

    doc = '\n'.join(out)
    doc = re.sub(
        r'(?m)^(\s*)import numpy as np\n\n'
        r'\1line = xt\.Line\(elements=\[\], element_names=\[\]\)\n\n'
        r'\1import xtrack as xt$',
        r'\1import numpy as np\n'
        r'\1import xtrack as xt\n\n'
        r'\1line = xt.Line(elements=[], element_names=[])',
        doc,
        count=1)
    doc = re.sub(
        r'(\.\. code-block:: python\n\n)(\s*)'
        r'((?!import |from ).*line\.xpart\.)',
        r'\1\2import xtrack as xt\n\n'
        r'\2line = xt.Line(elements=[], element_names=[])\n\n'
        r'\2\3',
        doc,
        count=1)
    if 'line.xpart.' in doc and 'line = xt.Line' not in doc:
        doc = doc.replace(
            '.. code-block:: python\n\n',
            '.. code-block:: python\n\n'
            '    import xtrack as xt\n\n'
            '    line = xt.Line(elements=[], element_names=[])\n\n',
            1)
    return doc


def _adapt_docstring(function, method_name, use_default_line):
    doc = inspect.cleandoc(function.__doc__ or '')
    source_name = function.__name__
    for name in {source_name, method_name}:
        doc = doc.replace(f'xp.{name}(', f'line.xpart.{method_name}(')
        doc = re.sub(
            rf'(?<![\w.])xpart\.{re.escape(name)}\(',
            f'line.xpart.{method_name}(',
            doc)
        doc = re.sub(
            rf'(?<![\w.]){re.escape(name)}\(',
            f'line.xpart.{method_name}(',
            doc)

    doc = re.sub(
        rf'\n\s*from xpart\.longitudinal import {re.escape(source_name)}\n',
        '\n',
        doc)

    if 'line.xpart.' in doc and 'xp.' not in doc:
        doc = re.sub(r'\n\s*import xpart as xp\n', '\n', doc)

    if 'line.xpart.' in doc and 'import xtrack as xt' not in doc:
        doc = re.sub(
            r'(?m)^(\s*)import numpy as np$',
            r'\1import numpy as np\n\1import xtrack as xt',
            doc,
            count=1)

    if use_default_line:
        doc = re.sub(r',\n\s*line=line\)', ')', doc)
        doc = re.sub(r'\n\s*line=line,\n', '\n', doc)
        line_param = re.search(
            r'(line : xtrack\.Line[^\n]*\n)'
            r'((?:    .*\n)+?)(?=\S[^:\n]* : |\*\*kwargs|Returns\n-------)',
            doc)
        if line_param is not None:
            _, end = line_param.span()
            doc = (
                doc[:end]
                + '    Defaults to the line owning this ``xpart`` container.\n'
                + doc[end:])

    return _insert_line_for_container_example(doc)


def _docstring_block(doc):
    lines = doc.splitlines()
    if not lines:
        return '        """"""'
    out = ['        """']
    out.extend(f'        {line}' if line else '        ' for line in lines)
    out.append('        """')
    return '\n'.join(out)


def _method_block(method_name, function_name, doc, use_default_line):
    if use_default_line:
        return_call = (
            f'        return xp.{function_name}(\n'
            f'            *args, **self._kwargs_with_line(kwargs))')
    else:
        return_call = f'        return xp.{function_name}(*args, **kwargs)'
    return f'''
    def {method_name}(self, *args, **kwargs):
{_docstring_block(doc)}
        import xpart as xp
{return_call}
'''


def _build_particles_docstring():
    return inspect.cleandoc("""
    Build particles using this line by default.

    This is the ``line.xpart`` container form of ``xpart.build_particles``.
    Defaults to the line owning this ``xpart`` container.
    Explicit ``line`` or ``tracker`` arguments override the container. See
    ``xtrack.Line.build_particles`` for the full parameter list.
    """)


def _render():
    out = '''# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
#
# This file is generated by xpart/_generate_line_tools.py.
# Do not edit it directly.


class XpartLineAPI:
    """
    Line-bound access to Xpart particle generation helpers.

    This API is exposed as ``line.xpart``. Methods delegate to the
    corresponding ``xpart`` functions. When the underlying function accepts a
    ``line`` argument, this line is used by default unless ``line`` or
    ``tracker`` is provided explicitly.
    """

    def __init__(self, line):
        self._line = line

    @property
    def line(self):
        """Line associated with this Xpart API."""
        return self._line

    def _kwargs_with_line(self, kwargs):
        if 'line' not in kwargs and 'tracker' not in kwargs:
            kwargs['line'] = self.line
        return kwargs
'''

    out += _method_block(
        'build_particles', build_particles.__name__,
        _build_particles_docstring(), use_default_line=True)

    for method_name, function in LINE_BOUND_METHODS:
        out += _method_block(
            method_name, function.__name__,
            _adapt_docstring(function, method_name, use_default_line=True),
            use_default_line=True)

    for method_name, function in LINE_FREE_METHODS:
        out += _method_block(
            method_name, function.__name__,
            _adapt_docstring(function, method_name, use_default_line=False),
            use_default_line=False)

    return out.lstrip() + '\n'


def main():
    (REPO_ROOT / 'xpart' / 'line_tools.py').write_text(_render())


if __name__ == '__main__':
    main()
