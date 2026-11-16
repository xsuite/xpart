import numpy as np
import pytest

import xpart as xp
import xtrack as xt


@pytest.fixture
def line():
    line = xt.Line(elements=[xt.LineSegmentMap(
        length=100., qx=0.28, qy=0.31, betx=3., bety=5.,
        longitudinal_mode='linear_fixed_qs', qs=0.01, bets=20.)],
        element_names=['map'], particle_ref=xt.Particles(p0c=2e9))
    line.build_tracker()
    return line


@pytest.mark.parametrize('generator', ['bunch', 'multibunch', 'pencil_x', 'pencil_y'])
@pytest.mark.parametrize('mixed', [False, True])
def test_geometric_distribution(line, generator, mixed):
    bg = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
    norm = dict(nemitt_x=2e-6, nemitt_y=3e-6)
    geom = dict(gemitt_x=2e-6/bg, gemitt_y=3e-6/bg)
    if mixed:
        geom.pop('gemitt_y')
        geom['nemitt_y'] = norm['nemitt_y']
    kwargs = dict(line=line)
    if generator == 'bunch':
        fun = xp.generate_matched_gaussian_bunch
        kwargs.update(num_particles=50, sigma_z=0.1)
    elif generator == 'multibunch':
        fun = xp.generate_matched_gaussian_multibunch_beam
        kwargs.update(filling_pattern=[1, 0, 1], bunch_num_particles=50,
                      bunch_intensity_particles=1e10, sigma_z=0.1, bucket_length=10.)
    else:
        fun = xp.generate_2D_pencil_with_absolute_cut
        kwargs.update(num_particles=50, plane=generator[-1], absolute_cut=0.01,
                      dr_sigmas=0.1)
    np.random.seed(789)
    pn = fun(**kwargs, **norm)
    np.random.seed(789)
    pg = fun(**kwargs, **geom)
    if generator.startswith('pencil'):
        np.testing.assert_allclose(pn, pg, rtol=1e-12, atol=1e-15)
        assert np.min(pg[0]) >= kwargs['absolute_cut']
    else:
        for key in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'weight'):
            np.testing.assert_allclose(getattr(pn, key), getattr(pg, key), rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError, match='Cannot provide both'):
        fun(**kwargs, **norm, gemitt_x=1e-6)
