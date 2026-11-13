import numpy as np
import pytest

import xpart as xp
import xtrack as xt


@pytest.fixture(params=['xpart', 'line', 'line.xpart'])
def builder(request):
    if request.param == 'xpart':
        return xp.build_particles
    line = xt.Line(elements=[xt.Drift(length=1)])
    line.build_tracker(compile=False)
    if request.param == 'line':
        return line.build_particles
    return line.xpart.build_particles


@pytest.mark.parametrize('emittances', [
    {'gemitt_x': 4e-6, 'gemitt_y': 9e-6, 'gemitt_zeta': 1e-4},
    {'gemitt_x': 4e-6, 'nemitt_y': 2e-6},
    {},
])
def test_build_particles_geometric_emittances(builder, emittances):
    co = xt.Particles(p0c=2e9)
    beta_gamma = co.beta0[0] * co.gamma0[0]
    diagonal = np.array([2., 0.5, 3., 1 / 3, 4., 0.25])
    coordinates = ['x', 'px', 'y', 'py', 'zeta', 'pzeta']
    normalized = {
        f'{name}_norm': np.array([0.01, -0.02]) * (ii + 1)
        for ii, name in enumerate(coordinates)
    }
    particles = builder(particle_on_co=co, W_matrix=np.diag(diagonal),
                        **normalized, **emittances)
    normalized_emittances = {}
    for ii, plane in enumerate(['x', 'y', 'zeta']):
        gemitt = emittances.get(f'gemitt_{plane}')
        if gemitt is None:
            nemitt = emittances.get(f'nemitt_{plane}')
            gemitt = 1 if nemitt is None else nemitt / beta_gamma
        normalized_emittances[f'nemitt_{plane}'] = gemitt * beta_gamma
        for jj in [2 * ii, 2 * ii + 1]:
            name = coordinates[jj]
            expected = normalized[f'{name}_norm'] * diagonal[jj] * np.sqrt(gemitt)
            np.testing.assert_allclose(getattr(particles, name), expected,
                                       rtol=1e-12, atol=1e-14)

    equivalent = builder(particle_on_co=co, W_matrix=np.diag(diagonal),
                         **normalized, **normalized_emittances)
    for name in coordinates:
        np.testing.assert_allclose(getattr(particles, name), getattr(equivalent, name),
                                   rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('plane', ['x', 'y', 'zeta'])
def test_build_particles_conflicting_emittances(builder, plane):
    with pytest.raises(ValueError, match=f'both nemitt_{plane} and gemitt_{plane}'):
        builder(particle_on_co=xt.Particles(p0c=2e9), W_matrix=np.eye(6),
                x_norm=1, **{f'nemitt_{plane}': 1e-6, f'gemitt_{plane}': 1e-6})


@pytest.mark.parametrize('plane', ['x', 'y'])
def test_build_particles_legacy_emittance_conflict(plane):
    with pytest.raises(ValueError, match=f'both nemitt_{plane} and gemitt_{plane}'):
        xp.build_particles(particle_on_co=xt.Particles(p0c=2e9),
                           W_matrix=np.eye(6), x_norm=1,
                           scale_with_transverse_norm_emitt=[1e-6, 2e-6],
                           **{f'gemitt_{plane}': 1e-6})
