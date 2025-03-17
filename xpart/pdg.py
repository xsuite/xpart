# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import numpy as np
from numbers import Number

from xtrack.particles.constants import U_MASS_EV, PROTON_MASS_EV, ELECTRON_MASS_EV, MUON_MASS_EV, Pb208_MASS_EV


# Monte Carlo numbering scheme as defined by the Particle Data Group
# See https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf for implementation
# details.
# Not all particles are implemented yet; this can be appended later


# This is the internal dictionary of PDG IDs.
# For each pdg id, we get the charge and a list of names.
# The first name is the default short name, the last is the default long name.
# The other names are accepted alternatives. Any superscript or subscript can be used
# interchangeably with its normal script (if both are used, subscript comes first).

_PDG = {
#   ID       q  NAME
    0:     [0.,  'undefined'],
    11:    [-1., 'e⁻', 'e', 'electron'],
    -11:   [1.,  'e⁺', 'positron'],
    12:    [0.,  'νₑ', 'electron neutrino'],
    13:    [-1., 'μ⁻', 'μ', 'muon-', 'muon'],
    -13:   [1.,  'μ⁺', 'muon+', 'anti-muon'],
    14:    [0.,  'νμ', 'muon neutrino'],
    15:    [-1., 'τ⁻', 'τ', 'tau-', 'tau'],
    -15:   [-1., 'τ⁺', 'tau+', 'anti-tau'],
    16:    [0.,  'ντ', 'tau neutrino'],
    22:    [0.,  'γ⁰', 'γ', 'photon'],
    111:   [0.,  'π⁰', 'π', 'pion', 'pion0', 'pi0'],
    211:   [1.,  'π⁺', 'pion+', 'pi+'],
    -211:  [-1., 'π⁻', 'pion-', 'pi-'],
    311:   [0.,  'K⁰', 'kaon', 'kaon0'],
    321:   [1.,  'K⁺', 'kaon+'],
    -321:  [-1., 'K⁻', 'kaon-'],
    130:   [0.,  'KL', 'long kaon'],
    310:   [0.,  'Kₛ', 'short kaon'],
    421:   [0.,  'D⁰', 'D'],
    411:   [1.,  'D⁺'],
    -411:  [-1., 'D⁻'],
    431:   [1.,  'Dₛ⁺'],
    -431:  [-1., 'Dₛ⁻'],
    2212:  [1.,  'p⁺', 'p', 'proton'],
    -2212: [1.,  'p⁻', 'anti-proton'],
    2112:  [0.,  'n⁰', 'n', 'neutron'],
    2224:  [2.,  'Δ⁺⁺', 'delta++'],
    2214:  [1.,  'Δ⁺', 'delta+'],
    2114:  [0.,  'Δ⁰', 'delta0'],
    1114:  [-1., 'Δ⁻', 'delta-'],
    3122:  [0.,  'Λ⁰', 'Λ', 'lambda'],
    4122:  [0.,  'Λc⁺', 'lambdac+'],
    3222:  [1.,  'Σ⁺', 'sigma+'],
    3212:  [0.,  'Σ⁰', 'Σ', 'sigma', 'sigma0'],
    3112:  [-1., 'Σ⁻', 'sigma-'],
    3322:  [0.,  'Ξ⁰', 'Ξ', 'xi', 'xi0'],
    3312:  [-1., 'Ξ⁻', 'xi-'],
    4132:  [0.,  'Ξc⁰', 'Ξc', 'xic', 'xic0'],
    4232:  [0.,  'Ξc⁺', 'xic+'],
    4312:  [0.,  "Ξ'c⁰", "Ξ'c", "xiprimec", "xiprimec0"],
    4322:  [0.,  "Ξ'c⁺", "xiprimec+"],
    3334:  [-1., 'Ω⁻', 'omega-'],
    4332:  [-1., 'Ωc⁰', 'Ωc', 'omegac', 'omegac0'],
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


def is_proton(pdg_id):
    return pdg_id >= 2212

def is_ion(pdg_id):
    return pdg_id >= 1000000000

def is_lepton(pdg_id):
    return 11 <= abs(pdg_id) <= 19


def get_name_from_pdg_id(pdg_id, long_name=True, subscripts=True):
    if hasattr(pdg_id, '__len__') and not isinstance(pdg_id, str):
        return np.array([get_name_from_pdg_id(pdg) for pdg in pdg_id])
    return get_properties_from_pdg_id(pdg_id, long_name=long_name, subscripts=subscripts)[-1]


def get_pdg_id_from_name(name=None):
    if hasattr(name, 'get'):
        name = name.get()

    if name is None:
        return 0  # undefined
    elif hasattr(name, '__len__') and not isinstance(name, str):
        return np.array([get_pdg_id_from_name(nn) for nn in name])
    elif isinstance(name, Number):
        return int(name) # fallback

    _PDG_inv = {}
    for pdg_id, val in _PDG.items():
        for vv in val[1:]:
            _PDG_inv[_to_normal_script(vv).lower()] = pdg_id

    lname = _to_normal_script(name).lower()
    aname = ""
    if len(lname) > 4 and lname[:5]=="anti-":
        aname = _flip_end_sign(lname[5:])

    # particle
    if lname in _PDG_inv.keys():
        return _PDG_inv[lname]

    # anti-particle
    elif aname in _PDG_inv.keys():
        return -_PDG_inv[aname]

    else:
        ion_name = _digits_to_normalscript(lname).lower()
        ion_name.replace('_','').replace('-','').replace(' ','')
        for Z, ion in _elements_long.items():
            if ion.lower() in ion_name:
                A = ion_name.replace(ion.lower(), '')
                if A.isnumeric() and int(A) > 0:
                    return get_pdg_id_ion(int(A), Z)
        for Z, ion in _elements.items():
            if ion.lower() in ion_name:
                A = ion_name.replace(ion.lower(), '')
                if A.isnumeric() and int(A) > 0:
                    return get_pdg_id_ion(int(A), Z)
        raise ValueError(f"Particle {name} not found in pdg dictionary, or wrongly "
                        + f"formatted ion name!\nFor ions, use e.g. 'Pb208', 'Pb 208', "
                        + f"'Pb-208', 'Pb_208', '208Pb', 'lead-208', ...")


# TODO: mass info ?
# q, A, Z, name
def get_properties_from_pdg_id(pdg_id, long_name=False, subscripts=True):
    if hasattr(pdg_id, '__len__') and not isinstance(pdg_id, str):
        result = np.array([get_properties_from_pdg_id(pdg) for pdg in pdg_id]).T
        return result[0].astype(np.float64), result[1].astype(np.int64),\
               result[2].astype(np.int64), result[3]

    if isinstance(pdg_id, str):
        pdg_id = int(pdg_id)

    if pdg_id in _PDG.keys():
        q = _PDG[pdg_id][0]
        if long_name:
            name = _PDG[pdg_id][-1]
        else:
            name = _PDG[pdg_id][1]
            if not subscripts:
                name = _to_normal_script(name)
        if abs(pdg_id)==2212 or abs(pdg_id)==2112:
            A = 1 if pdg_id > 0 else -1
            Z = q
        elif pdg_id==1000010020:
            A = 2
            Z = 1
        elif pdg_id==1000010030:
            A = 3
            Z = 1
        else:
            A = 0
            Z = 0
        return float(q), int(A), int(Z), name
    elif -pdg_id in _PDG.keys():
        antipart = get_properties_from_pdg_id(-pdg_id, long_name=long_name, subscripts=subscripts)
        name = _flip_end_sign(f'anti-{antipart[3]}')
        return -antipart[0], antipart[1], -antipart[2], name

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
        if long_name:
            return float(Z), int(A), int(Z), f'{get_element_full_name_from_Z(Z)}-{A}'
        else:
            if subscripts:
                name = f'{_digits_to_superscript(A)}{get_element_name_from_Z(Z)}'
            else:
                name = f'{get_element_name_from_Z(Z)}{A}'
            return float(Z), int(A), int(Z), name

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


def get_pdg_id_ion(A, Z):
    if hasattr(A, '__len__') and not isinstance(A, str) \
    and hasattr(Z, '__len__') and not isinstance(Z, str):
        return np.array([get_pdg_id_ion(aa, zz) for aa, zz in zip(A,Z)])
    return 1000000000 + int(Z)*10000 + int(A)*10


# TODO: this should be done a bit nicer, with a lookup table for the masses with A = 0
def get_pdg_id_from_mass_charge(m, q):
    if hasattr(q, '__len__') and not isinstance(q, str) \
    and hasattr(m, '__len__') and not isinstance(m, str):
        return np.array([get_pdg_id_from_mass_charge(mm, qq) for qq, mm in zip(q, m)])
    elif hasattr(q, '__len__') and not isinstance(q, str):
        return np.array([get_pdg_id_from_mass_charge(m, qq) for qq in q])
    elif hasattr(m, '__len__') and not isinstance(m, str):
        return np.array([get_pdg_id_from_mass_charge(mm, q) for mm in m])

    A = round(float(m)/U_MASS_EV)
    if abs(m-ELECTRON_MASS_EV) < 100:
        return -int(q)*get_pdg_id_from_name('electron')
    elif abs(m-MUON_MASS_EV) < 100:
        return -int(q)*get_pdg_id_from_name('muon')
    elif abs(m-PROTON_MASS_EV) < 1000:
        return int(q)*get_pdg_id_from_name('proton')
    elif q <= 0 or A <= 0:
        raise ValueError(f"Particle with {q=} and {m=} not recognised!")
    else:
        return get_pdg_id_ion(A, q)


# TODO: this should be done a bit nicer, with a lookup table
def get_mass_from_pdg_id(pdg_id, allow_approximation=True, expected_mass=None):
    if hasattr(pdg_id, '__len__') and not isinstance(pdg_id, str):
        return np.array([get_mass_from_pdg_id(pdg,
                                      allow_approximation=allow_approximation,
                                      expected_mass=expected_mass)
                         for pdg in pdg_id])

    _, A, _, name = get_properties_from_pdg_id(pdg_id, subscripts=False)
    if name == 'p+' or name == 'p-':
        return PROTON_MASS_EV
    elif name == 'e-' or name == 'e+':
        return ELECTRON_MASS_EV
    elif name == 'μ-' or name == 'μ+':
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


def _flip_end_sign(name):
    if name[-2:] == '⁺⁺':
        return name[:-2] + '⁻⁻'
    elif name[-2:] == '++':
        return name[:-2] + '--'
    elif name[-2:] == '⁻⁻':
        return name[:-2] + '⁺⁺'
    elif name[-2:] == '--':
        return name[:-2] + '++'
    elif name[-1] == '⁺':
        return name[:-1] + '⁻'
    elif name[-1] == '+':
        return name[:-1] + '-'
    elif name[-1] == '⁻':
        return name[:-1] + '⁺'
    elif name[-1] == '-':
        return name[:-1] + '+'
    else:
        return name


def _digits_to_superscript(val):
    val = f'{val}'.replace('0', '⁰').replace('1', '¹').replace('2', '²').replace('3', '³')
    val = val.replace('4', '⁴').replace('5', '⁵').replace('6', '⁶').replace('7', '⁷')
    return val.replace('8', '⁸').replace('9', '⁹')

def _digits_to_normalscript(val):
    val = f'{val}'.replace('⁰', '0').replace('¹', '1').replace('²', '2').replace('³', '3')
    val = val.replace('⁴', '4').replace('⁵', '5').replace('⁶', '6').replace('⁷', '7')
    return val.replace('⁸', '8').replace('⁹', '9')

def _to_normal_script(val):
    val = _digits_to_normalscript(val).replace('⁻', '-').replace('⁺', '+').replace('⁰', '0')
    return val.replace('ₛ', 's').replace('ₑ', 'e')


def _mass_consistent(pdg_id, m, mask=None):
    if hasattr(pdg_id, '__len__') and not isinstance(pdg_id, str) \
    and hasattr(m, '__len__') and not isinstance(m, str):
        if mask is None:
            return np.all([_mass_consistent(pdg, mm) for pdg, mm in zip(pdg_id, m)])
        else:
            pdg_id = np.array(pdg_id)[mask]
            m = np.array(m)[mask]
            return np.all([_mass_consistent(pdg, mm) for pdg, mm in zip(pdg_id, m)])
    elif hasattr(pdg_id, '__len__') and not isinstance(pdg_id, str):
        if mask is None:
            return np.all([_mass_consistent(pdg, m) for pdg in pdg_id])
        else:
            pdg_id = np.array(pdg_id)[mask]
            return np.all([_mass_consistent(pdg, m) for pdg in pdg_id])
    elif hasattr(m, '__len__') and not isinstance(m, str):
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
