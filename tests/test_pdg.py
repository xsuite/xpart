# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xpart as xp
from xpart.pdg import _PDG, _elements, _elements_long, _pdg_id_ion, _mass_consistent

from xobjects.test_helpers import for_all_test_contexts
import xobjects as xo

def test_names():
    names = [val[1] for val in _PDG.values()]
    names += [f"{val}174" for val in _elements.values()]
    names += [f"{val} 134" for val in _elements_long.values()]
    for name in [val[1] for val in _PDG.values()]:
        pdg_id = xp.pdg.get_pdg_id_from_name(name)
        assert xp.pdg.get_name_from_pdg_id(pdg_id) == name
        assert xp.pdg.get_pdg_id_from_name(
            xp.pdg.get_name_from_pdg_id(pdg_id)) == pdg_id


def test_lead_208():
    pdg_id = 1000822080
    assert _pdg_id_ion(208, 82) == pdg_id
    assert xp.pdg.get_pdg_id_from_mass_charge(xp.Pb208_MASS_EV, 82) == pdg_id
    assert xp.pdg.get_name_from_pdg_id(pdg_id) == 'Pb208'
    assert xp.pdg.get_pdg_id_from_name('Pb208')    == pdg_id
    assert xp.pdg.get_pdg_id_from_name('Pb 208')   == pdg_id
    assert xp.pdg.get_pdg_id_from_name('Pb-208')   == pdg_id
    assert xp.pdg.get_pdg_id_from_name('Pb_208')   == pdg_id
    assert xp.pdg.get_pdg_id_from_name('Pb.208')   == pdg_id
    assert xp.pdg.get_pdg_id_from_name('pb208')    == pdg_id
    assert xp.pdg.get_pdg_id_from_name('lead208')  == pdg_id
    assert xp.pdg.get_pdg_id_from_name('Lead 208') == pdg_id
    assert xp.pdg.get_pdg_id_from_name('Lead_208') == pdg_id
    assert _mass_consistent(pdg_id, xp.Pb208_MASS_EV)
    assert xp.pdg.get_element_name_from_Z(82) == 'Pb'
    assert xp.pdg.get_element_full_name_from_Z(82) == 'Lead'
    assert np.allclose(xp.pdg.get_mass_from_pdg_id(pdg_id), xp.Pb208_MASS_EV,
                       rtol=1e-10, atol=0)
    assert xp.pdg.get_properties_from_pdg_id(pdg_id) == (82., 208, 82, 'Pb208')

@for_all_test_contexts
def test_build_reference_from_pdg_id(test_context):
    particle_ref_proton  = xp.reference_from_pdg_id(pdg_id='proton',
                                                    _context=test_context)
    particle_ref_proton.move(_context=xo.context_default)
    assert particle_ref_proton.pdg_id == 2212
    particle_ref_lead = xp.reference_from_pdg_id(pdg_id='Pb208',
                                                 _context=test_context)
    particle_ref_lead.move(_context=xo.context_default)
    assert np.allclose(particle_ref_lead.q0, 82.)
    assert np.allclose(particle_ref_lead.mass0, xp.Pb208_MASS_EV)
