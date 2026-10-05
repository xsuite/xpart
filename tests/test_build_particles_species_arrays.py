import numpy as np
import pytest

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts


@pytest.mark.parametrize('mode', ['set', 'shift'])
@pytest.mark.parametrize('use_line', [False, True])
@pytest.mark.parametrize('longitudinal', ['delta', 'ptau', 'pzeta'])
@pytest.mark.parametrize('species', [
    dict(mass_ratio=[1., 2., 1.]),
    dict(charge_ratio=np.array([1., -1., 1.])),
    dict(chi=[1., .5, 1.]),
    dict(mass_ratio=[1., 2., 1.], charge_ratio=[1., 3., 1.]),
    dict(chi=[1., .5, 1.], charge_ratio=1.),
    dict(chi=[1., .5, 1.], mass_ratio=[1.]),
    dict(chi=[1., .5, 1.], mass_ratio=[1., 2., 1.], charge_ratio=1.),
    dict(pdg_id=[2212, 'electron', 'proton']),
])
def test_physical_species_arrays(mode, use_line, longitudinal, species):
    ref = xt.Particles(p0c=1e9, mass0=2 * xt.PROTON_MASS_EV, q0=2,
                       mass_ratio=1.5, charge_ratio=1.2, x=.002, delta=.01)
    if use_line:
        line = xt.Line(elements=[xt.Drift(length=1.)], particle_ref=ref)
        line.build_tracker(compile=False)
        build = line.build_particles
    else:
        def build(**kwargs):
            return xp.build_particles(particle_ref=ref, **kwargs)

    coordinates = dict(x=[.001, .002, -.001], px=1e-5,
                       **{longitudinal: [0., .01, -.01]})
    p = build(mode=mode, _capacity=5, **coordinates, **species)
    for i in range(3):
        inputs = {name: value if np.ndim(value) == 0 else value[
                    i if len(value) > 1 else 0]
                  for name, value in (coordinates | species).items()}
        expected = build(mode=mode, **inputs)
        for name in ('x', 'px', 'delta', 'pzeta', 'rvv', 'rpp',
                     'mass_ratio', 'charge_ratio', 'chi', 'pdg_id'):
            np.testing.assert_allclose(getattr(p, name)[i],
                                       getattr(expected, name)[0],
                                       rtol=1e-13, atol=1e-14)
    assert p._capacity == 5
    assert p._num_active_particles == 3
    np.testing.assert_array_equal(p.particle_id[:3], [0, 1, 2])
    np.testing.assert_allclose(ref.mass_ratio, 1.5)
    np.testing.assert_allclose(ref.charge_ratio, 1.2)


@pytest.mark.parametrize('name, values', [
    ('mass_ratio', [1., 2.]), ('charge_ratio', [1., 2.]),
    ('chi', [1., .5]), ('pdg_id', [2212, 11]),
])
@pytest.mark.parametrize('optics', [
    dict(x_norm=[0., 1.]), dict(mode='normalized_transverse'),
    dict(x_norm=[0., 1.], W_matrix=np.eye(6)),
    dict(mode='normalized_transverse', R_matrix=np.eye(6)),
])
def test_normalized_species_arrays_rejected_before_twiss(name, values, optics, monkeypatch):
    line = xt.Line(elements=[xt.Drift(length=1.)],
                   particle_ref=xt.Particles(p0c=1e9))
    line.build_tracker(compile=False)

    def unexpected_twiss(*args, **kwargs):
        pytest.fail('Mixed species must be rejected before computing optics')

    monkeypatch.setattr(line, 'twiss', unexpected_twiss)
    with pytest.raises(ValueError, match='Normalized coordinates require a single species') as exc:
        line.build_particles(**{name: values}, **optics)
    assert 'xt.Particles.merge()' in str(exc.value)


@pytest.mark.parametrize('species, scalar', [
    (dict(mass_ratio=[1.25, 1.25]), dict(mass_ratio=1.25)),
    (dict(charge_ratio=[.8, .8]), dict(charge_ratio=.8)),
    (dict(chi=[.8, .8]), dict(chi=.8)),
    (dict(pdg_id=[2212, 'proton']), dict(pdg_id=2212)),
])
def test_normalized_identical_species_and_merge(species, scalar):
    line = xt.Line(elements=[
        xt.Drift(length=1), xt.Multipole(knl=[0, .1]),
        xt.Drift(length=1), xt.Multipole(knl=[0, -.1]),
    ] * 4, particle_ref=xt.Particles(p0c=1e9))
    coords = dict(x_norm=[1., 2.], nemitt_x=1e-6, method='4d')
    p = line.build_particles(**coords, **species)
    expected = line.build_particles(**coords, **scalar)
    for name in ('x', 'px', 'y', 'py', 'delta', 'chi', 'charge_ratio', 'pdg_id'):
        np.testing.assert_allclose(getattr(p, name), getattr(expected, name),
                                   rtol=0, atol=1e-13)

    # The documented workflow for mixed normalized beams.
    reference_species = line.build_particles(**coords)
    merged = xt.Particles.merge([p, reference_species])
    np.testing.assert_allclose(merged.x, np.r_[p.x, reference_species.x])
    np.testing.assert_allclose(merged.chi, np.r_[p.chi, reference_species.chi])


@pytest.mark.parametrize('mode', ['set', 'shift'])
def test_species_arrays_infer_particle_count(mode):
    p = xp.build_particles(particle_ref=xt.Particles(p0c=1e9), mode=mode,
                           x=.001, mass_ratio=[1., 2., 1.], charge_ratio=[1.])
    np.testing.assert_array_equal(p.x, [.001] * 3)
    np.testing.assert_array_equal(p.mass_ratio, [1., 2., 1.])


@pytest.mark.parametrize('kwargs, message', [
    (dict(x=[0., 1.]), 'invalid length'),
    (dict(charge_ratio=[1., 2.]), 'invalid length'),
    (dict(num_particles=2), 'inconsistent with array length'),
    (dict(pdg_id=2212), 'inconsistent with'),
    (dict(pdg_id=[2212, 11]), 'invalid length'),
])
def test_species_arrays_inconsistent_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        xp.build_particles(particle_ref=xt.Particles(p0c=1e9),
                           mass_ratio=[1., 2., 1.], **kwargs)


def test_pdg_arrays_and_ratios_elementwise_consistency():
    mass_ratio = np.array([1., xt.ELECTRON_MASS_EV / xt.PROTON_MASS_EV])
    charge_ratio = np.array([1., -1.])
    ratios = dict(mass_ratio=mass_ratio, charge_ratio=charge_ratio,
                  chi=charge_ratio / mass_ratio)
    ref = xt.Particles(p0c=1e9)
    p = xp.build_particles(particle_ref=ref, pdg_id=[2212, 11],
                           x=[.001, -.002], delta=[.01, -.01], **ratios)
    expected = xt.Particles(p0c=1e9, pdg_id=[2212, 11],
                            x=[.001, -.002], delta=[.01, -.01], **ratios)
    for name in ('x', 'delta', 'pzeta', 'mass_ratio', 'charge_ratio', 'chi'):
        np.testing.assert_allclose(getattr(p, name), getattr(expected, name),
                                   rtol=1e-13, atol=1e-14)
    for name, value in ratios.items():
        inconsistent = value.copy()
        inconsistent[-1] *= 2
        with pytest.raises(ValueError, match=f'{name}.*inconsistent.*pdg_id'):
            xp.build_particles(particle_ref=ref, pdg_id=[2212, 11],
                               **(ratios | {name: inconsistent}))


@pytest.mark.parametrize('name', ['chi', 'charge_ratio', 'mass_ratio', 'pdg_id'])
@pytest.mark.parametrize('value', [[], [[1., 2.]]])
def test_invalid_species_shapes(name, value):
    with pytest.raises(ValueError, match=f'``{name}`` must be a scalar'):
        xp.build_particles(particle_ref=xt.Particles(p0c=1e9), **{name: value})


@for_all_test_contexts
def test_species_arrays_context_and_buffer(test_context):
    buffer = test_context.new_buffer(capacity=32768)
    ref = xt.Particles(p0c=1e9, _context=test_context)
    p = xp.build_particles(particle_ref=ref, _buffer=buffer, _offset=128,
                           mass_ratio=test_context.nparray_to_context_array(
                               np.array([1., 2., 1.])))
    assert p._buffer is buffer
    assert p._xobject._offset == 128
    p_cpu = p.copy(_context=xo.ContextCpu())
    np.testing.assert_array_equal(p_cpu.mass_ratio, [1., 2., 1.])
