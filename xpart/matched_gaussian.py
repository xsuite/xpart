# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

from .general import _print

from .longitudinal import generate_longitudinal_coordinates, _characterize_line
from .build_particles import build_particles

# To get the right Particles class depending on pyheatail interface state
import xpart as xp

def generate_matched_gaussian_bunch(num_particles,
                                    nemitt_x, nemitt_y, sigma_z,
                                    total_intensity_particles=None,
                                    particle_on_co=None,
                                    R_matrix=None,
                                    circumference=None,
                                    momentum_compaction_factor=None,
                                    rf_harmonic=None,
                                    rf_voltage=None,
                                    rf_phase=None,
                                    energy_ref_increment=None,
                                    tracker=None,
                                    line=None,
                                    particle_ref=None,
                                    engine=None,
                                    return_matcher=False,
                                    _context=None, _buffer=None, _offset=None,
                                    **kwargs,  # Passed to build_particles
                                    ):
    """
    Generate a matched Gaussian bunch.

    The transverse coordinates are generated from independent Gaussian
    distributions in normalized phase space and converted to physical
    coordinates using `build_particles`. The longitudinal coordinates are
    matched to the RF bucket using `generate_longitudinal_coordinates` with
    `distribution='gaussian'`.

    Parameters
    ----------
    num_particles : int
        Number of macroparticles to generate.
    nemitt_x : float
        Normalized horizontal emittance in m rad.
    nemitt_y : float
        Normalized vertical emittance in m rad.
    sigma_z : float
        RMS bunch length in m.
    total_intensity_particles : float, optional
        Total bunch intensity in physical particles. If not provided, the
        particle weights are set to one.
    particle_on_co : xpart.Particles, optional
        Particle on the closed orbit used as reference for the generated bunch.
        Cannot be provided together with `particle_ref`.
    R_matrix : array_like, optional
        Linear transfer matrix passed to `build_particles`.
    circumference : float, optional
        Ring circumference in m. Required when no `line` is provided.
    momentum_compaction_factor : float, optional
        Momentum compaction factor. Required when no `line` is provided.
    rf_harmonic : float or array_like, optional
        RF harmonic number or numbers. Required when no `line` is provided.
    rf_voltage : float or array_like, optional
        RF voltage or voltages in V. Required when no `line` is provided.
    rf_phase : float or array_like, optional
        RF phase or phases in rad. Required when no `line` is provided.
    energy_ref_increment : float, optional
        Reference energy increment used for the longitudinal matching.
    tracker : xtrack.Tracker, optional
        Deprecated. Use `line` instead.
    line : xtrack.Line, optional
        Line for which the bunch is generated. If provided, missing RF and
        lattice parameters are inferred from the line.
    particle_ref : xpart.Particles, optional
        Reference particle. If not provided, `line.particle_ref` is used when
        available. Cannot be provided together with `particle_on_co`.
    engine : str, optional
        Longitudinal matching engine passed to
        `generate_longitudinal_coordinates`.
    return_matcher : bool, optional
        If True, also return the longitudinal matcher object.
    _context : xobjects.Context, optional
        Context on which to allocate the returned particles.
    _buffer : xobjects.Buffer, optional
        Buffer on which to allocate the returned particles.
    _offset : int, optional
        Offset in `_buffer` at which to allocate the returned particles.
    **kwargs
        Additional keyword arguments passed to `generate_longitudinal_coordinates`
        and `build_particles`.

    Returns
    -------
    particles : xpart.Particles
        Generated matched Gaussian bunch.
    matcher : object
        Longitudinal matcher used for the generation. Returned only when
        `return_matcher` is True.

    Example
    -------

    .. code-block:: python

        import numpy as np
        import xpart as xp
        import xtrack as xt

        np.random.seed(12345)

        circumference = 26658.883
        line = xt.Line(elements=[
            xt.LineSegmentMap(
                length=circumference,
                betx=1.0, qx=0.31,
                bety=1.0, qy=0.32,
                longitudinal_mode='linear_fixed_rf',
                voltage_rf=16e6,
                frequency_rf=400.8e6,
                phase_rf=np.pi,
                slippage_length=circumference,
                momentum_compaction_factor=3.225e-4,
            )
        ])
        line.set_particle_ref('proton', p0c=7e12)

        particles = xp.generate_matched_gaussian_bunch(
            num_particles=4,
            total_intensity_particles=1e11,
            nemitt_x=2e-6,
            nemitt_y=2e-6,
            sigma_z=0.08,
            line=line)

        len(particles.x)  # 4
        particles.weight  # [2.5e+10, 2.5e+10, 2.5e+10, 2.5e+10]
        particles.zeta    # [-0.077427, -0.044981, 0.025293, -0.110336]
    """

    if line is not None and tracker is not None:
        raise ValueError(
            'line and tracker cannot be provided at the same time.')

    if tracker is not None:
        _print(
            "The argument tracker is deprecated. Please use line instead.",
            DeprecationWarning)
        line = tracker.line

    if line is not None and line.tracker is None:
        line.build_tracker()

    if particle_ref is not None and particle_on_co is not None:
        raise ValueError("`particle_ref` and `particle_on_co`"
                         " cannot be provided at the same time")

    if particle_ref is None:
        if particle_on_co is not None:
            particle_ref = particle_on_co
        elif line is not None and line.particle_ref is not None:
            particle_ref = line.particle_ref
        else:
            raise ValueError(
                "`line`, `particle_ref` or `particle_on_co` must be provided!")

    zeta, delta, matcher = generate_longitudinal_coordinates(
        distribution='gaussian',
        num_particles=num_particles,
        particle_ref=(particle_ref if particle_ref is not None
                      else particle_on_co),
        line=line,
        circumference=circumference,
        momentum_compaction_factor=momentum_compaction_factor,
        rf_harmonic=rf_harmonic,
        rf_voltage=rf_voltage,
        rf_phase=rf_phase,
        energy_ref_increment=energy_ref_increment,
        sigma_z=sigma_z,
        engine=engine,
        return_matcher=True,
        **kwargs)

    assert len(zeta) == len(delta) == num_particles

    x_norm = np.random.normal(size=num_particles)
    px_norm = np.random.normal(size=num_particles)
    y_norm = np.random.normal(size=num_particles)
    py_norm = np.random.normal(size=num_particles)

    if total_intensity_particles is None:
        # go to particles.weight = 1
        total_intensity_particles = num_particles

    part = build_particles(_context=_context, _buffer=_buffer, _offset=_offset,
                      R_matrix=R_matrix,
                      particle_on_co=particle_on_co,
                      particle_ref=(
                          particle_ref if particle_on_co is  None else None),
                      line=line,
                      zeta=zeta, delta=delta,
                      x_norm=x_norm, px_norm=px_norm,
                      y_norm=y_norm, py_norm=py_norm,
                      nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                      weight=total_intensity_particles/num_particles,
                      **kwargs)
    if return_matcher:
        return part, matcher
    else:
        return part


def split_scheme(filling_scheme, n_chunk=1):
    """
    Distribute the filling scheme between the processes, i.e. assign to each
    processor its bunches
    """
    total_n_bunches = len(filling_scheme.nonzero()[0])
    if n_chunk > 1:

        # create the array containing the id of the bunches on each rank
        # (copied from PyHEADTAIL.mpi.mpi_data)
        n_bunches = total_n_bunches
        n_bunches_on_rank = [n_bunches//n_chunk+1 if i < n_bunches % n_chunk
                             else n_bunches // n_chunk + 0
                             for i in range(n_chunk)]
        n_tasks_cumsum = np.insert(np.cumsum(n_bunches_on_rank), 0, 0)
        total_bunch_ids = np.unique(
            np.cumsum(filling_scheme == 1)) - 1
        bunches_per_rank = [total_bunch_ids[n_tasks_cumsum[i]:
                                            n_tasks_cumsum[i + 1]]
                            for i in range(n_chunk)]
    else:
        bunches_per_rank = [np.linspace(0,
                                        total_n_bunches - 1,
                                        total_n_bunches)]

    bunches_per_rank = list(map(np.int64, bunches_per_rank))

    return bunches_per_rank


def generate_matched_gaussian_multibunch_beam(filling_scheme,
                                              bunch_num_particles,
                                              nemitt_x, nemitt_y, sigma_z,
                                              bunch_intensity_particles=None,
                                              particle_on_co=None,
                                              R_matrix=None,
                                              circumference=None,
                                              momentum_compaction_factor=None,
                                              rf_harmonic=None,
                                              rf_voltage=None,
                                              rf_phase=None,
                                              bucket_length = None,
                                              energy_ref_increment=None,
                                              tracker=None,
                                              line=None,
                                              particle_ref=None,
                                              engine=None,
                                              _context=None, _buffer=None, _offset=None,
                                              bunch_selection=None,
                                              bunch_spacing_buckets=1,
                                              prepare_line_and_particles_for_mpi_wake_sim=False,
                                              communicator=None,
                                              **kwargs,  # Passed to build_particles
                                              ):
    """
    Generate a matched Gaussian multibunch beam.

    Each selected bunch is generated with `generate_matched_gaussian_bunch` and
    then shifted in `zeta` according to the filled bucket positions in
    `filling_scheme`. The returned object contains the selected bunches
    concatenated in bunch-selection order.

    Parameters
    ----------
    filling_scheme : array_like
        One-dimensional array indicating which RF buckets are filled. Non-zero
        entries are treated as filled buckets.
    bunch_num_particles : int
        Number of macroparticles to generate per bunch.
    nemitt_x : float
        Normalized horizontal emittance in m rad.
    nemitt_y : float
        Normalized vertical emittance in m rad.
    sigma_z : float
        RMS bunch length of each bunch in m.
    bunch_intensity_particles : float, optional
        Intensity of each bunch in physical particles.
    particle_on_co : xpart.Particles, optional
        Particle on the closed orbit used as reference for the generated
        bunches. Cannot be provided together with `particle_ref`.
    R_matrix : array_like, optional
        Linear transfer matrix passed to `build_particles`.
    circumference : float, optional
        Ring circumference in m. If not provided, it is taken from `line`.
    momentum_compaction_factor : float, optional
        Momentum compaction factor passed to the longitudinal matching.
    rf_harmonic : float or array_like, optional
        RF harmonic number or numbers. Used to infer the bucket length when
        `bucket_length` is not provided.
    rf_voltage : float or array_like, optional
        RF voltage or voltages in V. Used together with `rf_harmonic` for the
        longitudinal matching and to infer the main harmonic.
    rf_phase : float or array_like, optional
        RF phase or phases in rad.
    bucket_length : float, optional
        Bucket spacing in m. If provided, `rf_harmonic` and `rf_voltage` must
        not be provided.
    energy_ref_increment : float, optional
        Reference energy increment used for the longitudinal matching.
    tracker : xtrack.Tracker, optional
        Deprecated. Use `line` instead.
    line : xtrack.Line, optional
        Line for which the beam is generated. If provided, missing RF and
        lattice parameters are inferred from the line.
    particle_ref : xpart.Particles, optional
        Reference particle. If not provided, `line.particle_ref` is used when
        available.
    engine : str, optional
        Longitudinal matching engine passed to
        `generate_longitudinal_coordinates`.
    _context : xobjects.Context, optional
        Context on which to allocate the returned particles.
    _buffer : xobjects.Buffer, optional
        Buffer on which to allocate the returned particles.
    _offset : int, optional
        Offset in `_buffer` at which to allocate the returned particles.
    bunch_selection : iterable of int, optional
        Indices, within the list of filled buckets, of the bunches to generate.
        If not provided, all filled bunches are generated, unless MPI wake
        preparation is enabled.
    bunch_spacing_buckets : int, optional
        Spacing between consecutive entries of `filling_scheme`, expressed in
        RF buckets. The physical spacing is
        `bunch_spacing_buckets * bucket_length`.
    prepare_line_and_particles_for_mpi_wake_sim : bool, optional
        If True, split the filled bunches over MPI ranks when `bunch_selection`
        is not provided and configure the line and particles for wakefield
        simulations.
    communicator : mpi4py communicator, optional
        MPI communicator used when
        `prepare_line_and_particles_for_mpi_wake_sim` is True. If not provided,
        `mpi4py.MPI.COMM_WORLD` is used.
    **kwargs
        Additional keyword arguments passed to
        `generate_matched_gaussian_bunch`.

    Returns
    -------
    particles : xpart.Particles
        Particles object containing the generated selected bunches.

    Example
    -------

    .. code-block:: python

        import numpy as np
        import xpart as xp
        import xtrack as xt

        np.random.seed(12345)

        circumference = 26658.883
        line = xt.Line(elements=[
            xt.LineSegmentMap(
                length=circumference,
                betx=1.0, qx=0.31,
                bety=1.0, qy=0.32,
                longitudinal_mode='linear_fixed_rf',
                voltage_rf=16e6,
                frequency_rf=400.8e6,
                phase_rf=np.pi,
                slippage_length=circumference,
                momentum_compaction_factor=3.225e-4,
            )
        ])
        line.set_particle_ref('proton', p0c=7e12)

        filling_scheme = np.zeros(4, dtype=int)
        filling_scheme[[0, 2]] = 1

        particles = xp.generate_matched_gaussian_multibunch_beam(
            filling_scheme=filling_scheme,
            bunch_num_particles=3,
            bunch_intensity_particles=1e11,
            nemitt_x=2e-6,
            nemitt_y=2e-6,
            sigma_z=0.08,
            line=line,
            bucket_length=299792458 / 400.8e6)

        len(particles.x)       # 6
        particles.weight[:3]   # [3.333333e+10, 3.333333e+10, 3.333333e+10]
        particles.zeta         # [-0.016377, 0.038315, -0.041555, ...]
    """

    if particle_ref is None and line is not None:
        particle_ref = line.particle_ref

    assert ((line is not None and particle_ref is not None) or
            (rf_harmonic is not None and rf_voltage is not None) or
            bucket_length is not None)

    if circumference is None:
        circumference = line.get_length()
    assert circumference > 0.0

    if bucket_length is not None:
        assert (rf_harmonic is None and rf_voltage is None),(
                'Cannot provide bucket length together with RF voltage+harmonic')
    else:
        if rf_harmonic is not None and rf_voltage is not None:
            main_harmonic_number = rf_harmonic[np.argmax(rf_voltage)]
        else:
            dct_line = _characterize_line(line, particle_ref)
            assert len(dct_line['voltage_list']) > 0
            main_harmonic_number = int(np.floor(
                        dct_line['h_list'][np.argmax(dct_line['voltage_list'])]+0.5))
        bucket_length = circumference/main_harmonic_number
    bunch_spacing = bunch_spacing_buckets * bucket_length
    assert filling_scheme is not None
    assert len(filling_scheme) <= np.floor(circumference/bunch_spacing+0.5)

    if len(filling_scheme) < np.floor(circumference/bunch_spacing+0.5):
        filling_scheme = np.concatenate(
            (filling_scheme,
            np.zeros(int(np.floor(circumference/bunch_spacing+0.5) - len(filling_scheme)),
                    dtype=np.int64)))

    if prepare_line_and_particles_for_mpi_wake_sim and bunch_selection is None:
        if communicator is None:
            from mpi4py import MPI
            communicator = MPI.COMM_WORLD

        if communicator.Get_size() <= 1:
            raise ValueError('when `prepare_line_and_particles_for_mpi_wake_sim` is True, '
                             'MPI communicator must have more than one rank')
        bunch_selection_rank = split_scheme(filling_scheme=filling_scheme,
                                             n_chunk=int(communicator.Get_size()))
        bunch_selection = bunch_selection_rank[communicator.Get_rank()]

    if bunch_selection is None:
        bunch_selection = range(len(filling_scheme.nonzero()[0]))

    macro_bunch = generate_matched_gaussian_bunch(
        num_particles=bunch_num_particles * len(bunch_selection),
        nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
        total_intensity_particles=bunch_intensity_particles * len(bunch_selection),
        particle_on_co=particle_on_co,
        R_matrix=R_matrix,
        circumference=circumference,
        momentum_compaction_factor=momentum_compaction_factor,
        rf_harmonic=rf_harmonic,
        rf_voltage=rf_voltage,
        rf_phase=rf_phase,
        energy_ref_increment=energy_ref_increment,
        tracker=tracker,
        line=line,
        particle_ref=particle_ref,
        engine=engine,
        _context=_context, _buffer=_buffer, _offset=_offset,
        **kwargs,  # They are passed to build_particles
    )

    filled_buckets = filling_scheme.nonzero()[0]
    count = 0
    for bunch_number in bunch_selection:
        bucket_n = filled_buckets[bunch_number]
        macro_bunch.zeta[count * bunch_num_particles:
                         (count+1) * bunch_num_particles] -= (bunch_spacing *
                                                        bucket_n)
        count += 1

    if prepare_line_and_particles_for_mpi_wake_sim:
        import xwakes as xw
        xw.config_pipeline_for_wakes(
            particles=macro_bunch,
            line=line,
            communicator=communicator)

    return macro_bunch
