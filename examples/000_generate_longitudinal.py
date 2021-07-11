import numpy as np
import xpart as xp
from scipy.constants import e as qe
from scipy.constants import m_p
from scipy.constants import c as clight


rfbucket = xp.RFBucket(circumference=6911., gamma=27.6433,
                       mass=m_p, charge=qe, alpha_array=[0.003077672469],
                       harmonic_list=[4620], voltage_list=[3e6],
                       p_increment=0, phi_offset_list=[0.])

matcher = xp.RFBucketMatcher(rfbucket=rfbucket,
        distribution_type=xp.rfbucket_matching.ThermalDistribution,
        sigma_z=22.5e-2)

n_part = int(1e6)

z_particle, delta_particles, _, _ = matcher.generate(macroparticlenumber=n_part)

