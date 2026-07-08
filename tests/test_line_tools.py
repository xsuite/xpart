import pytest

import xpart as xp
import xtrack as xt
from xpart.line_tools import XpartLineAPI


LINE_BOUND_GENERATORS = [
    ('build_particles', 'build_particles'),
    ('generate_matched_gaussian_bunch', 'generate_matched_gaussian_bunch'),
    ('generate_matched_gaussian_multibunch_beam',
     'generate_matched_gaussian_multibunch_beam'),
    ('generate_2D_pencil_with_absolute_cut',
     'generate_2D_pencil_with_absolute_cut'),
    ('generate_longitudinal_coordinates', 'generate_longitudinal_coordinates'),
    ('generate_binomial_longitudinal_coordinates',
     'generate_binomial_longitudinal_coordinates'),
    ('generate_parabolic_longitudinal_coordinates',
     'generate_parabolic_longitudinal_coordinates'),
    ('generate_qgaussian_longitudinal_coordinates',
     'generate_qgaussian_longitudinal_coordinates'),
]

LINE_FREE_GENERATORS = [
    ('generate_2D_polar_grid', 'generate_2D_polar_grid'),
    ('generate_2D_uniform_circular_sector',
     'generate_2D_uniform_circular_sector'),
    ('generate_2D_pencil', 'generate_2D_pencil'),
    ('generate_2D_gaussian', 'generate_2D_gaussian'),
    ('generate_hypersphere_2D', 'generate_hypersphere_2D'),
    ('generate_hypersphere_4D', 'generate_hypersphere_4D'),
    ('generate_hypersphere_6D', 'generate_hypersphere_6D'),
]


@pytest.mark.parametrize('method_name,function_name', LINE_BOUND_GENERATORS)
def test_line_api_injects_line(monkeypatch, method_name, function_name):
    line = xt.Line(elements=[], element_names=[])
    api = XpartLineAPI(line)
    calls = []

    def fake_function(*args, **kwargs):
        calls.append((args, kwargs))
        return 'result'

    monkeypatch.setattr(xp, function_name, fake_function)

    assert getattr(api, method_name)('arg', option='value') == 'result'
    assert calls == [(('arg',), {'option': 'value', 'line': line})]


@pytest.mark.parametrize('method_name,function_name', LINE_BOUND_GENERATORS)
def test_line_api_keeps_explicit_tracker(monkeypatch, method_name, function_name):
    line = xt.Line(elements=[], element_names=[])
    api = XpartLineAPI(line)
    tracker = object()
    calls = []

    def fake_function(*args, **kwargs):
        calls.append((args, kwargs))
        return 'result'

    monkeypatch.setattr(xp, function_name, fake_function)

    assert getattr(api, method_name)(tracker=tracker) == 'result'
    assert calls == [((), {'tracker': tracker})]


@pytest.mark.parametrize('method_name,function_name', LINE_FREE_GENERATORS)
def test_line_api_keeps_line_free_generators_line_free(
        monkeypatch, method_name, function_name):
    line = xt.Line(elements=[], element_names=[])
    api = XpartLineAPI(line)
    calls = []

    def fake_function(*args, **kwargs):
        calls.append((args, kwargs))
        return 'result'

    monkeypatch.setattr(xp, function_name, fake_function)

    assert getattr(api, method_name)('arg', option='value') == 'result'
    assert calls == [(('arg',), {'option': 'value'})]


@pytest.mark.parametrize(
    'method_name,function_name', LINE_BOUND_GENERATORS + LINE_FREE_GENERATORS)
def test_line_api_docstrings_adapt_xpart_functions(method_name, function_name):
    doc = getattr(XpartLineAPI, method_name).__doc__
    source_doc = getattr(xp, function_name).__doc__

    assert doc is not None
    if function_name != 'build_particles':
        assert source_doc.strip().splitlines()[0] in doc
        assert f'line.xpart.{method_name}(' in doc
        assert f'xp.{function_name}(' not in doc
        assert f'from xpart.longitudinal import {function_name}' not in doc

    if (method_name, function_name) in LINE_BOUND_GENERATORS:
        assert 'Defaults to the line owning this ``xpart`` container.' in doc
        assert 'line=line' not in doc

    if (method_name, function_name) in LINE_FREE_GENERATORS:
        assert 'line = xt.Line(elements=[], element_names=[])' in doc
