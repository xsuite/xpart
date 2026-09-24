import numpy as np
import pytest

import xobjects as xo
import xpart as xp
import xtrack as xt
from xtrack.particles.masses import C12_MASS_EV, He4_MASS_EV


@pytest.mark.parametrize('mode', ['set', 'shift'])
@pytest.mark.parametrize('use_line', [False, True])
@pytest.mark.parametrize('overrides, charge_ratio, mass_ratio', [
    ({}, 1.2, 1.5),
    ({'mass_ratio': 2.}, 1.2, 2.),
    ({'charge_ratio': 0.8}, 0.8, 1.5),
    ({'chi': 0.6}, 1.2, 2.),
    ({'mass_ratio': 2., 'charge_ratio': 0.8}, 0.8, 2.),
])
def test_build_particles_species(mode, use_line, overrides,
                                 charge_ratio, mass_ratio):
    particle_ref = xt.Particles(
        p0c=1e9, x=1e-3, delta=0.01, charge_ratio=1.2, mass_ratio=1.5)
    kwargs = dict(mode=mode, num_particles=2, x=[1e-3, 2e-3], **overrides)

    if use_line:
        line = xt.Line(elements=[xt.Drift(length=1)], particle_ref=particle_ref)
        particles = line.build_particles(**kwargs)
    else:
        particles = xp.build_particles(particle_ref=particle_ref, **kwargs)

    xo.assert_allclose(particles.charge_ratio, charge_ratio, rtol=0, atol=1e-14)
    xo.assert_allclose(particles.mass_ratio, mass_ratio, rtol=0, atol=1e-14)
    xo.assert_allclose(particles.chi, charge_ratio / mass_ratio,
                       rtol=0, atol=1e-14)
    xo.assert_allclose(particle_ref.charge_ratio, 1.2, rtol=0, atol=1e-14)
    xo.assert_allclose(particle_ref.mass_ratio, 1.5, rtol=0, atol=1e-14)


@pytest.mark.parametrize('overrides, charge_ratio, mass_ratio', [
    ({'mass_ratio': 1.25}, 1., 1.25),
    ({'charge_ratio': 0.8}, 0.8, 1.),
    ({'chi': 0.8}, 1., 1.25),
    ({'mass_ratio': 1.25, 'charge_ratio': 1.}, 1., 1.25),
])
def test_build_particles_species_with_twiss(overrides, charge_ratio, mass_ratio):
    line = xt.Line(elements=[
        xt.Drift(length=1),
        xt.Multipole(knl=[0, 0.1]),
        xt.Drift(length=1),
        xt.Multipole(knl=[0, -0.1]),
    ] * 4, particle_ref=xt.Particles(p0c=1e9))
    kwargs = dict(x_norm=[1., 2.], nemitt_x=1e-6, method='4d')
    particles = line.build_particles(**overrides, **kwargs)
    expected_ref = xt.Particles(
        p0c=1e9, delta=0, mass_ratio=mass_ratio, charge_ratio=charge_ratio)
    expected = line.build_particles(particle_ref=expected_ref, **kwargs)

    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta',
                 'mass_ratio', 'charge_ratio', 'chi'):
        xo.assert_allclose(getattr(particles, name), getattr(expected, name),
                           rtol=0, atol=1e-13)
    xo.assert_allclose(line.particle_ref.mass_ratio, 1., rtol=0, atol=1e-14)


@pytest.mark.parametrize('particle_arg', ['particle_on_co', 'co_guess'])
@pytest.mark.parametrize('overrides', [{'mass_ratio': 1.25}, {'pdg_id': 2212}])
def test_build_particles_species_with_supplied_orbit(particle_arg, overrides):
    particle = xt.Particles(p0c=1e9)
    kwargs = {particle_arg: particle, **overrides}
    if particle_arg == 'co_guess':
        kwargs['particle_ref'] = particle
    with pytest.raises(ValueError, match='Set the species ratios directly'):
        xp.build_particles(**kwargs)


@pytest.mark.parametrize('mode', ['set', 'shift'])
@pytest.mark.parametrize('use_line', [False, True])
@pytest.mark.parametrize('pdg_id, expected_id, mass, charge', [
    (11, 11, xt.ELECTRON_MASS_EV, -1),
    ('positron', -11, xt.ELECTRON_MASS_EV, 1),
    ('He4', 1000020040, He4_MASS_EV, 2),
    (np.array([1000060120]), 1000060120, C12_MASS_EV, 6),
    (np.array([1000020040]), 1000020040, He4_MASS_EV, 2),
])
def test_build_particles_pdg_id(mode, use_line, pdg_id, expected_id, mass, charge):
    particle_ref = xt.Particles(
        p0c=1e9, mass0=2 * xt.PROTON_MASS_EV, q0=2, pdg_id=2212,
        delta=0.01, mass_ratio=1.5, charge_ratio=1.2)
    kwargs = dict(pdg_id=pdg_id, mode=mode, x=[1e-3, 2e-3])
    if use_line:
        line = xt.Line(elements=[xt.Drift(length=1)], particle_ref=particle_ref)
        particles = line.build_particles(**kwargs)
    else:
        particles = xp.build_particles(particle_ref=particle_ref, **kwargs)

    xo.assert_allclose(particles.pdg_id, expected_id, rtol=0, atol=0)
    xo.assert_allclose(particles.mass_ratio, mass / particle_ref.mass0,
                       rtol=1e-14, atol=0)
    xo.assert_allclose(particles.charge_ratio, charge / particle_ref.q0,
                       rtol=0, atol=1e-14)
    xo.assert_allclose(particles.mass, mass, rtol=1e-14, atol=0)
    xo.assert_allclose(particles.charge, charge, rtol=0, atol=1e-14)
    xo.assert_allclose(particles.delta, 0.01 if mode == 'shift' else 0.,
                       rtol=0, atol=1e-14)
    xo.assert_allclose(particle_ref.pdg_id, 2212, rtol=0, atol=0)
    xo.assert_allclose(particle_ref.mass_ratio, 1.5, rtol=0, atol=1e-14)
    xo.assert_allclose(particle_ref.charge_ratio, 1.2, rtol=0, atol=1e-14)


@pytest.mark.parametrize('pdg_id', [[], [11, -11]])
def test_build_particles_pdg_id_requires_single_species(pdg_id):
    with pytest.raises(ValueError, match='must identify a single species'):
        xp.build_particles(particle_ref=xt.Particles(p0c=1e9), pdg_id=pdg_id)


@pytest.mark.parametrize('ratio_names', [
    (), ('mass_ratio',), ('charge_ratio',), ('chi',),
    ('mass_ratio', 'charge_ratio'), ('mass_ratio', 'chi'),
    ('charge_ratio', 'chi'), ('mass_ratio', 'charge_ratio', 'chi'),
])
def test_build_particles_pdg_id_consistent_ratios(ratio_names):
    particle_ref = xt.Particles(p0c=1e9, mass0=2 * xt.PROTON_MASS_EV, q0=2)
    ratios = dict(mass_ratio=0.5, charge_ratio=0.5, chi=1.)
    particles = xp.build_particles(
        particle_ref=particle_ref, pdg_id=2212,
        **{name: ratios[name] for name in ratio_names})
    for name, expected in ratios.items():
        xo.assert_allclose(getattr(particles, name), expected, rtol=0, atol=1e-14)
    xo.assert_allclose(particles.pdg_id, 2212, rtol=0, atol=0)


@pytest.mark.parametrize('overrides, inconsistent', [
    ({'mass_ratio': 1.}, 'mass_ratio'),
    ({'charge_ratio': 1.}, 'charge_ratio'),
    ({'chi': 2.}, 'chi'),
    ({'mass_ratio': 1., 'charge_ratio': 1.}, 'mass_ratio'),
])
def test_build_particles_pdg_id_inconsistent_ratios(overrides, inconsistent):
    line = xt.Line(elements=[xt.Drift(length=1)], particle_ref=xt.Particles(
        p0c=1e9, mass0=2 * xt.PROTON_MASS_EV, q0=2))
    with pytest.raises(ValueError, match=f'{inconsistent}.*inconsistent.*pdg_id'):
        line.build_particles(pdg_id=2212, **overrides)


def test_build_particles_pdg_id_with_twiss():
    line = xt.Line(elements=[
        xt.Drift(length=1),
        xt.Multipole(knl=[0, 0.1]),
        xt.Drift(length=1),
        xt.Multipole(knl=[0, -0.1]),
    ] * 4, particle_ref=xt.Particles(pdg_id_0='He4', p0c=1e9))
    kwargs = dict(x_norm=[1., 2.], nemitt_x=1e-6, method='4d')
    particles = line.build_particles(pdg_id=2212, **kwargs)
    expected_ref = xt.Particles(
        p0c=1e9, mass0=He4_MASS_EV, q0=2, delta=0, pdg_id=2212,
        mass_ratio=xt.PROTON_MASS_EV / He4_MASS_EV, charge_ratio=0.5)
    expected = line.build_particles(particle_ref=expected_ref, **kwargs)
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta',
                 'mass_ratio', 'charge_ratio', 'chi', 'pdg_id'):
        xo.assert_allclose(getattr(particles, name), getattr(expected, name),
                           rtol=0, atol=1e-13)
    xo.assert_allclose(line.particle_ref.pdg_id, 1000020040, rtol=0, atol=0)
