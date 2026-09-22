import pytest

import xobjects as xo
import xpart as xp
import xtrack as xt


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
def test_build_particles_species_with_supplied_orbit(particle_arg):
    particle = xt.Particles(p0c=1e9)
    kwargs = {particle_arg: particle, 'mass_ratio': 1.25}
    if particle_arg == 'co_guess':
        kwargs['particle_ref'] = particle
    with pytest.raises(ValueError, match='Set the species ratios directly'):
        xp.build_particles(**kwargs)
