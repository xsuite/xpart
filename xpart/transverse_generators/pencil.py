import numpy as np
from .polar import generate_2D_uniform_circular_sector

def generate_2D_pencil(num_particles, pos_cut_sigmas, dr_sigmas,
                       side='+'):

    assert side == '+' or side == '-' or side == '+-'

    if side == '+-':
        n_plus = int(num_particles/2)
        n_minus = num_particles - n_plus
        x_plus, px_plus, r_plus, theta_plus = generate_2D_pencil(n_plus,
                                                 pos_cut_sigmas, dr_sigmas, side='+')
        x_minus, px_minus, r_minus, theta_minus = generate_2D_pencil(n_minus,
                                                 pos_cut_sigmas, dr_sigmas, side='-')
        x_norm = np.concatenate([x_minus, x_plus])
        px_norm = np.concatenate([px_minus, px_plus])
        r_points = np.concatenate([r_minus, r_plus])
        theta_points = np.concatenate([theta_minus, theta_plus])

        return x_norm, px_norm, r_points, theta_points

    else:

        r_min = np.abs(pos_cut_sigmas)
        r_max = r_min + dr_sigmas
        theta_max = np.arccos(pos_cut_sigmas/(np.abs(pos_cut_sigmas) + dr_sigmas))
        target_area = r_max**2/2*(2 * theta_max - np.sin(2 * theta_max))
        generated_area = (r_max**2 - r_min**2) * theta_max

        n_gen = int(num_particles*generated_area/target_area)

        x_gen, px_gen, r_points, theta_points = generate_2D_uniform_circular_sector(
                # Should avoid regen most of the times
                num_particles=int(n_gen*1.5+100),
                r_range=(r_min, r_max),
                theta_range=(-theta_max, theta_max))

        mask = x_gen > r_min
        x_norm = x_gen[mask]
        px_norm = px_gen[mask]
        r_points = r_points[mask]
        theta_points = theta_points[mask]

        assert len(x_norm) > num_particles

        x_norm = x_norm[:num_particles]
        px_norm = px_norm[:num_particles]
        r_points = r_points[:num_particles]
        theta_points = theta_points[:num_particles]

        if side == '-':
            x_norm = -x_norm

        return x_norm, px_norm, r_points, theta_points
