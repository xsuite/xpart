import xpart as xp
import numpy as np
import xtrack as xt
import json
import matplotlib.pyplot as plt


filename = xt._pkg_root.joinpath('../../../xsuite/xtrack/test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)

line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])

line.build_tracker()
circumference = line.get_length()
h_list = [35640]
bunch_spacing_in_buckets = 10
bucket_length = circumference/h_list[0]
filling_scheme = np.zeros(int(np.amin(h_list)/bunch_spacing_in_buckets))
n_bunches_tot = 10
filling_scheme[0:int(n_bunches_tot/2)] = 1

filling_scheme[n_bunches_tot:int(3*n_bunches_tot/2)] = 1

bunch_intensity = 1e11
sigma_z = 22.5e-2/5
n_part_per_bunch = int(1e5)
nemitt_x = 2e-6
nemitt_y = 2.5e-6

n_procs = 2
bunche_numbers_per_rank = xp.split_scheme(filling_scheme=filling_scheme,
                                        n_chunk=n_procs)

colors = ['b','g']
for rank in range(n_procs):
    particles = xp.generate_matched_gaussian_multibunch_beam(
             filling_scheme=filling_scheme,
             num_particles=n_part_per_bunch,
             total_intensity_particles=bunch_intensity,
             nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
             line=line, bunch_spacing_buckets=10,
             bunch_numbers=bunche_numbers_per_rank[rank],
             particle_ref=line.particle_ref
    )

    print(fr"target sigma_z: {sigma_z}")
    for i_bunch,bunch_number in enumerate(bunche_numbers_per_rank[rank]):
        zeta_rms = np.std(particles.zeta[i_bunch*n_part_per_bunch:
                                         (i_bunch+1)*n_part_per_bunch])
        print(f'zeta rms of bunch {i_bunch}: {zeta_rms}')

    plt.figure(0)
    plt.plot(particles.zeta/bucket_length, particles.delta, 'x',color=colors[rank])
filled_slots = filling_scheme.nonzero()[0]
for filled_slot in filled_slots:
    plt.figure(0)
    plt.axvline(filled_slot*bunch_spacing_in_buckets,color='k',ls='--')

plt.show()
