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

filename = xt._pkg_root.joinpath('/Users/lorenzogiacomel/xsuite/xtrack/test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)

line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])

line.build_tracker()
#line_dict = xp._characterize_line(line, particle_ref)

circumference = line.get_length()
h_list = [35640]

bunch_spacing_in_buckets = 10
filling_scheme_array = np.zeros(3564)
n_bunches = 10
filling_scheme_array[0:int(n_bunches/2)] = 1

filling_scheme_array[n_bunches:int(3*n_bunches/2)] = 1

n_procs = 2
rank = 0

bunch_intensity = 1e11
sigma_z = 22.5e-2/2
n_part = int(1e4)
nemitt_x = 2e-6
nemitt_y = 2.5e-6

communicator = DummyCommunicator(n_procs, rank)
filling_scheme = xp.FillingScheme(bunch_spacing_in_buckets=bunch_spacing_in_buckets,
                                  filling_scheme_array=filling_scheme_array, communicator=communicator,
                                  circumference=circumference, harmonic_list=h_list)

particles = xp.generate_matched_gaussian_beam(
         filling_scheme=filling_scheme,
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         line=line)

#assert (filling_scheme.bunches_per_rank[0] == np.linspace(0, 33, 34)).all()
#assert (filling_scheme.bunches_per_rank[1] == np.linspace(34, 66, 33)).all()
#assert (filling_scheme.bunches_per_rank[2] == np.linspace(67, 99, 33)).all()

plt.plot(particles.zeta, particles.delta, 'bx')
rank=1
communicator = DummyCommunicator(n_procs, rank)
filling_scheme = xp.FillingScheme(bunch_spacing_in_buckets=bunch_spacing_in_buckets,
                                  filling_scheme_array=filling_scheme_array, communicator=communicator,
                                  circumference=circumference, harmonic_list=h_list)

particles = xp.generate_matched_gaussian_beam(
         filling_scheme=filling_scheme,
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         line=line)

plt.plot(particles.zeta, particles.delta, 'rx')
plt.show()


