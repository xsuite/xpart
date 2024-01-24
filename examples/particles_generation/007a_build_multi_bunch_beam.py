import xpart as xp
import numpy as np
import xtrack as xt
import json
import matplotlib.pyplot as plt

class DummyCommunicator:
    def __init__(self, n_procs, rank):
        self.n_procs = n_procs
        self.rank = rank

    def Get_size(self):
        return self.n_procs

    def Get_rank(self):
        return self.rank


filename = xt._pkg_root.joinpath('../../../xsuite/xtrack/test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)

line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])

line.build_tracker()
circumference = line.get_length()
h_list = [35640]
bunch_spacing_in_buckets = 10
filling_scheme = np.zeros(int(np.amin(h_list)/bunch_spacing_in_buckets))
n_bunches_tot = 10
filling_scheme[0:int(n_bunches_tot/2)] = 1

filling_scheme[n_bunches_tot:int(3*n_bunches_tot/2)] = 1

n_procs = 2
rank = 0

bunch_intensity = 1e11
sigma_z = 22.5e-2/5
n_part_per_bunch = int(1e5)
nemitt_x = 2e-6
nemitt_y = 2.5e-6

communicator = DummyCommunicator(n_procs, rank)

first_bunch, n_bunches = xp.split_scheme(filling_scheme=filling_scheme,
                                         communicator=communicator)

particles = xp.generate_matched_gaussian_beam(
         filling_scheme=filling_scheme,
         num_particles=n_part_per_bunch,
         total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         line=line, bunch_spacing_buckets=10,
         first_bunch=first_bunch, n_bunches=n_bunches,
         particle_ref=line.particle_ref
)

print(fr"target sigma_z: {sigma_z}")
for i_bunch in range(n_bunches):
    zeta_rms = np.std(particles.zeta[i_bunch*n_part_per_bunch:
                                     (i_bunch+1)*n_part_per_bunch])
    print(f'zeta rms of bunch {i_bunch}: {zeta_rms}')
    assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)

plt.plot(particles.zeta, particles.delta, 'bx')
rank = 1
communicator = DummyCommunicator(n_procs, rank)

first_bunch, n_bunches = xp.split_scheme(filling_scheme=filling_scheme,
                                         communicator=communicator)

particles = xp.generate_matched_gaussian_beam(
    filling_scheme=filling_scheme,
    num_particles=n_part_per_bunch,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
    line=line, bunch_spacing_buckets=10,
    first_bunch=first_bunch, n_bunches=n_bunches,
    particle_ref=line.particle_ref
)

print(fr"target sigma_z: {sigma_z}")
for i_bunch in range(n_bunches):
    zeta_rms = np.std(particles.zeta[i_bunch*n_part_per_bunch:
                                     (i_bunch+1)*n_part_per_bunch])
    print(f'zeta rms of bunch {i_bunch}: {zeta_rms}')
    assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)

plt.plot(particles.zeta, particles.delta, 'rx')
plt.show()
