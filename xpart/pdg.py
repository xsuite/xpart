# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import numpy as np
from numbers import Number

from xtrack.particles.constants import U_MASS_EV, PROTON_MASS_EV, ELECTRON_MASS_EV, MUON_MASS_EV, Pb208_MASS_EV


# Monte Carlo numbering scheme as defined by the Particle Data Group
# See https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf for implementation
# details.
# Not all particles are implemented yet; this can be appended later


_PDG = {
#   ID       q  NAME
    0:     [0.,  'undefined'],
    11:    [-1., 'electron'],
    -11:   [1.,  'positron'],
    13:    [-1., 'muon'],
    15:    [-1., 'tau'],
    111:   [0.,  'pi0'],
    211:   [1.,  'pi+'],
    -211:  [-1., 'pi-'],
    311:   [0.,  'K0'],
    321:   [1.,  'K+'],
    -321:  [-1., 'K-'],
    2212:  [1.,  'proton'],
    2112:  [0.,  'neutron'],
    2224:  [2.,  'Delta++'],
    2214:  [1.,  'Delta+'],
    2114:  [0.,  'Delta0'],
    1114:  [-1., 'Delta-'],
    3122:  [0.,  'Lambda'],
    3222:  [1.,  'Sigma+'],
    3212:  [0.,  'Sigma0'],
    3112:  [-1., 'Sigma-'],
    3322:  [0.,  'Xi'],
    3312:  [-1., 'Xi-'],
    1000010020: [1., 'deuterium'],
    1000010030: [1., 'tritium']
}

_elements = {
     1: "H",    2: "He",   3: "Li",   4: "Be",   5: "B",    6: "C",    7: "N",    8: "O",    9: "F",   10: "Ne",
    11: "Na",  12: "Mg",  13: "Al",  14: "Si",  15: "P",   16: "S",   17: "Cl",  18: "Ar",  19: "K",   20: "Ca",
    21: "Sc",  22: "Ti",  23: "V",   24: "Cr",  25: "Mn",  26: "Fe",  27: "Co",  28: "Ni",  29: "Cu",  30: "Zn",
    31: "Ga",  32: "Ge",  33: "As",  34: "Se",  35: "Br",  36: "Kr",  37: "Rb",  38: "Sr",  39: "Y",   40: "Zr",
    41: "Nb",  42: "Mo",  43: "Tc",  44: "Ru",  45: "Rh",  46: "Pd",  47: "Ag",  48: "Cd",  49: "In",  50: "Sn",
    51: "Sb",  52: "Te",  53: "I",   54: "Xe",  55: "Cs",  56: "Ba",  57: "La",  58: "Ce",  59: "Pr",  60: "Nd",
    61: "Pm",  62: "Sm",  63: "Eu",  64: "Gd",  65: "Tb",  66: "Dy",  67: "Ho",  68: "Er",  69: "Tm",  70: "Yb",
    71: "Lu",  72: "Hf",  73: "Ta",  74: "W",   75: "Re",  76: "Os",  77: "Ir",  78: "Pt",  79: "Au",  80: "Hg",
    81: "Tl",  82: "Pb",  83: "Bi",  84: "Po",  85: "At",  86: "Rn",  87: "Fr",  88: "Ra",  89: "Ac",  90: "Th",
    91: "Pa",  92: "U",   93: "Np",  94: "Pu",  95: "Am",  96: "Cm",  97: "Bk",  98: "Cf",  99: "Es", 100: "Fm",
   101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds",
   111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
}

_elements_long = {
    1: "Hydrogen",        2: "Helium",          3: "Lithium",         4: "Beryllium",       5: "Boron",
    6: "Carbon",          7: "Nitrogen",        8: "Oxygen",          9: "Fluorine",       10: "Neon",
   11: "Sodium",         12: "Magnesium",      13: "Aluminum",       14: "Silicon",        15: "Phosphorus",
   16: "Sulfur",         17: "Chlorine",       18: "Argon",          19: "Potassium",      20: "Calcium",
   21: "Scandium",       22: "Titanium",       23: "Vanadium",       24: "Chromium",       25: "Manganese",
   26: "Iron",           27: "Cobalt",         28: "Nickel",         29: "Copper",         30: "Zinc",
   31: "Gallium",        32: "Germanium",      33: "Arsenic",        34: "Selenium",       35: "Bromine",
   36: "Krypton",        37: "Rubidium",       38: "Strontium",      39: "Yttrium",        40: "Zirconium",
   41: "Niobium",        42: "Molybdenum",     43: "Technetium",     44: "Ruthenium",      45: "Rhodium",
   46: "Palladium",      47: "Silver",         48: "Cadmium",        49: "Indium",         50: "Tin",
   51: "Antimony",       52: "Tellurium",      53: "Iodine",         54: "Xenon",          55: "Cesium",
   56: "Barium",         57: "Lanthanum",      58: "Cerium",         59: "Praseodymium",   60: "Neodymium",
   61: "Promethium",     62: "Samarium",       63: "Europium",       64: "Gadolinium",     65: "Terbium",
   66: "Dysprosium",     67: "Holmium",        68: "Erbium",         69: "Thulium",        70: "Ytterbium",
   71: "Lutetium",       72: "Hafnium",        73: "Tantalum",       74: "Tungsten",       75: "Rhenium",
   76: "Osmium",         77: "Iridium",        78: "Platinum",       79: "Gold",           80: "Mercury",
   81: "Thallium",       82: "Lead",           83: "Bismuth",        84: "Polonium",       85: "Astatine",
   86: "Radon",          87: "Francium",       88: "Radium",         89: "Actinium",       90: "Thorium",
   91: "Protactinium",   92: "Uranium",        93: "Neptunium",      94: "Plutonium",      95: "Americium",
   96: "Curium",         97: "Berkelium",      98: "Californium",    99: "Einsteinium",   100: "Fermium",
  101: "Mendelevium",   102: "Nobelium",      103: "Lawrencium",    104: "Rutherfordium", 105: "Dubnium",
  106: "Seaborgium",    107: "Bohrium",       108: "Hassium",       109: "Meitnerium",    110: "Darmstadtium",
  111: "Roentgenium",   112: "Copernicium",   113: "Nihonium",      114: "Flerovium",     115: "Moscovium",
  116: "Livermorium",   117: "Tennessine",    118: "Oganesson"
}

def get_name_from_pdg_id(pdg_id):
    if hasattr(pdg_id, '__len__'):
        return np.array([get_name_from_pdg_id(pdg) for pdg in pdg_id])
    return get_properties_from_pdg_id(pdg_id)[-1]


def get_pdg_id_from_name(name=None):

    if hasattr(name, 'get'):
        name = name.get()

    if name is None:
        return 0  # undefined
    elif hasattr(name, '__len__') and not isinstance(name, str):
        return np.array([get_pdg_id_from_name(nn) for nn in name])
    elif isinstance(name, Number):
        return int(name) # fallback

    _PDG_inv      = {val[1].lower(): pdg_id for pdg_id, val in _PDG.items()}

    lname = name.lower()
    aname = ""
    if len(lname) > 4 and lname[:5]=="anti-":
        aname = lname[5:]

    # particle
    if lname in _PDG_inv.keys():
        return _PDG_inv[lname]

    # anti-particle
    elif aname in _PDG_inv.keys():
        return -_PDG_inv[aname]

    else:
        ion_long = [[Z, ion.lower()] for Z, ion in _elements_long.items()
                    if lname.startswith(ion.lower())]
        ion      = [[Z, ion.lower()] for Z, ion in _elements.items()
                    if lname.startswith(ion.lower())]

        # ion short name
        if len(ion_long) > 0:
            # Multiple finds are possible (e.g. [15, P] and [82, Pb] in the case of lead).
            # Use the result that has the longest name length (most restrictive result)
            ion_long.sort(key=lambda el: len(el[1]))
            ion_long = ion_long[-1]
            Z = ion_long[0]
            try:
                A = int(lname.replace(ion_long[1], '').replace('.','').replace('_','').replace('-','').replace(' ',''))
            except:
                raise ValueError(f"Wrongly formatted ion name: cannot deduce A from {name}!\n"
                               + f"Use e.g. 'Pb208', 'Pb 208', 'Pb-208', 'Pb_208', or 'Pb.208'.")
            return _pdg_id_ion(A, Z)

        # ion short name
        elif len(ion) > 0:
            # Multiple finds are possible (e.g. [15, P] and [82, Pb] in the case of lead).
            # Use the result that has the longest name length (most restrictive result)
            ion.sort(key=lambda el: len(el[1]))
            ion = ion[-1]
            Z = ion[0]
            try:
                A = int(lname.replace(ion[1], '').replace('.','').replace('_','').replace('-','').replace(' ',''))
            except:
                raise ValueError(f"Wrongly formatted ion name: cannot deduce A from {name}!\n"
                               + f"Use e.g. 'Pb208', 'Pb 208', 'Pb-208', 'Pb_208', or 'Pb.208'.")
            return _pdg_id_ion(A, Z)

        else:
            raise ValueError(f"Particle {name} not found in pdg dictionary!")


# TODO: mass info ?
# q, A, Z, name
def get_properties_from_pdg_id(pdg_id):
    if hasattr(pdg_id, '__len__'):
        result = np.array([get_properties_from_pdg_id(pdg) for pdg in pdg_id]).T
        return result[0].astype(np.float64), result[1].astype(np.int64),\
               result[2].astype(np.int64), result[3]

    if pdg_id in _PDG.keys():
        q = _PDG[pdg_id][0]
        name = _PDG[pdg_id][1]
        if name=='proton' or name=='neutron':
            A = 1
            Z = q
        elif name=='deuterium':
            A = 2
            Z = 1
        elif name=='tritium':
            A = 3
            Z = 1
        else:
            A = 0
            Z = 0
        return float(q), int(A), int(Z), name
    elif -pdg_id in _PDG.keys():
        antipart = get_properties_from_pdg_id(-pdg_id)
        return -antipart[0], antipart[1], -antipart[2], f'anti-{antipart[3]}'

    elif pdg_id > 1000000000:
        # Ion
        tmpid = pdg_id - 1000000000
        L = int(tmpid/1e7)
        tmpid -= L*1e7
        Z = int(tmpid /1e4)
        tmpid -= Z*1e4
        A = int(tmpid /10)
        tmpid -= A*10
        isomer_level = int(tmpid)
        return float(Z), int(A), int(Z), f'{get_element_name_from_Z(Z)}{A}'#, L, isomer_level, get_element_full_name_from_Z(Z)

    else:
        raise ValueError(f"PDG ID {pdg_id} not recognised!")


def get_element_name_from_Z(z):
    if z not in _elements:
        raise ValueError(f"Element with {z} protons not known.")
    return _elements[z]


def get_element_full_name_from_Z(z):
    if z not in _elements_long:
        raise ValueError(f"Element with {z} protons not known.")
    return _elements_long[z]


def _pdg_id_ion(A, Z):
    if hasattr(A, '__len__') and hasattr(Z, '__len__'):
        return np.array([_pdg_id_ion(aa, zz) for aa, zz in zip(A,Z)])
    return  int(1000000000 + Z*10000 + A*10)


# TODO: this should be done a bit nicer, with a lookup table for the masses with A = 0
def get_pdg_id_from_mass_charge(m, q):
    if hasattr(q, '__len__') and hasattr(m, '__len__'):
        return np.array([get_pdg_id_from_mass_charge(mm, qq) for qq, mm in zip(q, m)])
    elif hasattr(q, '__len__'):
        return np.array([get_pdg_id_from_mass_charge(m, qq) for qq in q])
    elif hasattr(m, '__len__'):
        return np.array([get_pdg_id_from_mass_charge(mm, q) for mm in m])

    A = round(m/U_MASS_EV)
    if abs(m-ELECTRON_MASS_EV) < 100:
        return get_pdg_id_from_name('electron')
    elif abs(m-MUON_MASS_EV) < 100:
        return get_pdg_id_from_name('muon')
    elif abs(m-PROTON_MASS_EV) < 1000:
        return get_pdg_id_from_name('proton')
    elif q <= 0 or A <= 0:
        raise ValueError(f"Particle with {q=} and {m=} not recognised!")
    else:
        return _pdg_id_ion(A, q)


def _mass_consistent(pdg_id, m, mask=None):
    if hasattr(pdg_id, '__len__') and hasattr(m, '__len__'):
        if mask is None:
            return np.all([_mass_consistent(pdg, mm) for pdg, mm in zip(pdg_id, m)])
        else:
            pdg_id = np.array(pdg_id)[mask]
            m = np.array(m)[mask]
            return np.all([_mass_consistent(pdg, mm) for pdg, mm in zip(pdg_id, m)])
    elif hasattr(pdg_id, '__len__'):
        if mask is None:
            return np.all([_mass_consistent(pdg, m) for pdg in pdg_id])
        else:
            pdg_id = np.array(pdg_id)[mask]
            return np.all([_mass_consistent(pdg, m) for pdg in pdg_id])
    elif hasattr(m, '__len__'):
        if mask is None:
            return np.all([_mass_consistent(pdg_id, mm) for mm in m])
        else:
            m = np.array(m)[mask]
            return np.all([_mass_consistent(pdg_id, mm) for mm in m])

    q, A, _, name = get_properties_from_pdg_id(pdg_id)
    if name=='proton' or name=='electron' or name=='muon':
        return pdg_id == get_pdg_id_from_mass_charge(m, q)
    elif A > 1:
        return A==round(m/U_MASS_EV)
    else:
        # No check for other particles
        return True


# TODO: this should be done a bit nicer, with a lookup table
def get_mass_from_pdg_id(pdg_id, allow_approximation=True, expected_mass=None):
    if hasattr(pdg_id, '__len__'):
        return np.array([get_mass_from_pdg_id(pdg,
                                      allow_approximation=allow_approximation,
                                      expected_mass=expected_mass)
                         for pdg in pdg_id])

    _, A, _, name = get_properties_from_pdg_id(pdg_id)
    if name == 'proton':
        return PROTON_MASS_EV
    elif name == 'electron':
        return ELECTRON_MASS_EV
    elif name == 'muon':
        return MUON_MASS_EV
    elif name == 'Pb208':
        return Pb208_MASS_EV
    elif allow_approximation and A>0:
        print(f"Warning: approximating the mass as {A}u!")
        return A*U_MASS_EV
    else:
        if expected_mass is not None and _mass_consistent(pdg_id, expected_mass):
            # This is a workaround in case an exact mass is given
            # (like for the reference particle)
            return expected_mass
        else:
            raise ValueError(f"Exact mass for {name} not found.")





