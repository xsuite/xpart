import numpy as np
import xpart as xp
from scipy.constants import e as qe
from scipy.constants import m_p
from scipy.constants import c as clight


#from PyHEADTAIL.machines.synchrotron import Synchrotron
#
#synchrotron = Synchrotron(
#        optics_mode='smooth',
#        charge = qe,
#        mass=m_p,
#        circumference=6911.,
#        n_segments=1,
#        p0=26e9/clight*qe,
#        beta_x=100.,
#        beta_y=100.,
#        D_x=0.,
#        D_y=0.,
#        accQ_x=20.13,
#        accQ_y=20.18,
#        longitudinal_mode='non-linear',
#        alpha_mom_compaction=0.003077672469,
#        h_RF=4620,
#        dphi_RF=0,
#        p_increment=0)



rfbucket = xp.RFBucket(circumference=6911., gamma=27.6433,
                       mass=m_p, charge=qe, alpha_array=[0.003077672469],
                       harmonic_list=[4620], voltage_list=[3e6],
                       p_increment=0, phi_offset_list=[0.])

matcher = xp.RFBucketMatcher(rfbucket=rfbucket,
        distribution_type=xp.rfbucket_matching.ThermalDistribution,
        sigma_z=22.5e-2)



