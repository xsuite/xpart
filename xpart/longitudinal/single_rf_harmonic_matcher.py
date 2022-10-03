import numpy as np
from scipy.constants import c
import scipy.special

from .generate_longitudinal import _characterize_tracker

class SingleRFHarmonicMatcher:
    def __init__(self, tracker=None, particle_ref=None,
                 rms_bunch_length=None, distribution="parabolic",
                 transformation_particles=400000, verbose=0):

        assert tracker is not None
        assert particle_ref is not None
        
        self.verbose = verbose
        self.transformation_particles = transformation_particles

        dct = _characterize_tracker(tracker, particle_ref)
        atol = 1.e-10
        voltage_list = dct["voltage_list"]
        ## mean_voltage = np.mean(voltage_list)
        ## if not np.allclose(mean_voltage, voltage_list, atol=atol, rtol=0.):
        ##     raise Exception(f"Multiple harmonics detected in lattice: {voltage_list}")

        harmonic_list = dct["h_list"]
        harmonic_number = np.round(harmonic_list).astype(int)[0]
        if not np.allclose(harmonic_number, harmonic_list, atol=atol, rtol=0.):
            raise Exception(f"Multiple harmonics detected in lattice: {harmonic_list}")

        freq = dct['freq_list'][0]
        length = harmonic_number * particle_ref.beta0[0] * c / freq
        p0c_eV = particle_ref.p0c[0]
        V0 = np.sum(voltage_list)
        eta = dct['slip_factor']

        # Hamoltonian: H = A cos(B tau) - C ptau^2
        # normalized Hamiltonian: m = ( sin(B/2*tau) )^2 + C/(2A) ptau^ 2
        self.A = V0/(2.*np.pi*freq*p0c_eV/c*length)
        self.B = 2*np.pi*freq/c
        self.C = eta/2.

        tau_ufp = self.get_unstable_fixed_point()
        dtau = 0.01*tau_ufp # don't get too close to the separatrix
        if distribution == "parabolic":
            tau_max = np.sqrt(5)*rms_bunch_length
            self.tau_distr_x = np.linspace(- tau_ufp + dtau, tau_ufp - dtau, 300)
            self.tau_distr_y = 1 - (self.tau_distr_x/tau_max)**2
            self.tau_distr_y[abs(self.tau_distr_x) > tau_max] = 0
            if tau_max >= tau_ufp:
                print(f"WARNING SingleRFHarmonicMatcher: longitudinal profile larger than bucket, truncating to unstable fixed point. tau_max = {tau_max:.4f}, tau_ufp = {tau_ufp:.4f} ")
        elif distribution == "gaussian":
            self.tau_distr_x = np.linspace(- tau_ufp + dtau, tau_ufp - dtau, 300)
            self.tau_distr_y = np.exp(-self.tau_distr_x**2/2./rms_bunch_length**2)
        else:
            raise NotImplementedError

        self.m_distr_x, self.m_distr_y = self.transform_tau_distr_to_m_distr()


    def transform_tau_distr_to_m_distr(self):
        xp = self.tau_distr_x.copy()
        yp = self.tau_distr_y.copy()
        N = int(len(xp)/2.)
        dx = xp[1] - xp[0]
        m_distr_x = []
        m_distr_y = []
        for ii in range(N):
            jj = len(xp) - ii - 1 #start from end
            tau = xp[jj]
            dens = yp[jj]
            if dens == 0.:
                continue
            if self.verbose:
                print(f"tau = {tau:.3f}, f(tau) = {dens:.3f}")
            
            m0 = self.get_m(tau=tau - dx/2.)
            m1 = self.get_m(tau=tau + dx/2.)
            dm = m1 - m0
            m = self.get_m(tau=tau)
            if m == 1.0:
                continue
            print('SingleRFHarmonicMatcher: Transforming distribution: '
                        f'{round(ii/N*100):2d}%  ',end="\r", flush=True)
            tau_test, ptau_test = self.get_airbag_from_m(m=m, n_particles=self.transformation_particles)
            hist, bin_edges = np.histogram(tau_test, bins=len(xp), range=(min(xp)-dx/2., max(xp)+dx/2.))
        
            hist = (hist + hist[::-1])/2.
            factor = dens/hist[jj]
            yp -= hist*factor
            m_distr_x.append(m)
            m_distr_y.append(np.sum(hist)*factor/dm)

        m_distr_x.append(0)
        m_distr_y.append(0)
        m_distr_x = m_distr_x[::-1]
        m_distr_y = m_distr_y[::-1]

        print('SingleRFHarmonicMatcher: Done transforming distribution.')
        return m_distr_x, m_distr_y


    def get_separatrix(self):
        ufp = self.get_unstable_fixed_point()
        xx = np.linspace(-ufp,ufp, 1000)
        yy = np.sqrt(2*self.A/self.C) * np.cos(self.B/2.*xx)
        return xx, yy

    def get_unstable_fixed_point(self):
        return np.pi/self.B

    def get_m(self, tau=0, ptau=0):
        return ( np.sin(self.B/2. * tau) )**2 + self.C / 2. / self.A * (ptau ** 2)

    def get_airbag_from_m(self, m, n_particles=20000):
        if n_particles is None:
            n_particles = len(m)

        K = scipy.special.ellipk(m)
        G = 2.*K/np.pi
        theta = np.random.uniform(size=n_particles)*2.*np.pi
        sn, cn, dn, ph = scipy.special.ellipj(G*theta,m)
    
        tau = 2./self.B*np.arcsin(np.sqrt(m)*sn)
        ptau = np.sqrt(2*self.A*m/self.C)*cn
    
        return tau, ptau

    def sample_tau_ptau(self, n_particles=20000):
        max_m = max(self.m_distr_x)
        tau_new = []
        ptau_new = []
        counter = 0
        max_y = np.max(self.m_distr_y)
        chunk = 20000
        ### Acceptance-rejection algorithm sampling from distribution of m and a random "angle"
        ### The random angle is the conjugate variable to the action variable and is only
        ### approximately equal to the angle in the tau-ptau space. 
        while counter < n_particles:
            m = np.random.random(size=chunk)*max_m
            rand_test = np.random.random(size=chunk)*max_y
            yy = np.interp(m, self.m_distr_x, self.m_distr_y)
            
            tau, ptau = self.get_airbag_from_m(m, n_particles=None)
            
            mask = rand_test < yy
        
            tau_new.extend(list(tau[mask]))
            ptau_new.extend(list(ptau[mask]))
            counter += sum(mask)
            print('SingleRFHarmonicMatcher: Sampling particles: '
                        f'{round(counter/n_particles*100):2d}%  ',end="\r", flush=True)
        print(f"SingleRFHarmonicMatcher: Sampled {n_particles} particles")
        
        return tau_new[:n_particles], ptau_new[:n_particles]