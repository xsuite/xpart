# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from xtrack.particles.constants import U_MASS_EV, PROTON_MASS_EV, ELECTRON_MASS_EV, MUON_MASS_EV, Pb208_MASS_EV



source = r'''
#ifndef XPART_MASSES_H
#define XPART_MASSES_H

// 1u = 931494102.42(28)    2018 CODATA

//       name                            mass [eV]         PDG ID
#define  ELECTRON_MASS_EV              510998.9506916          11   // best known value:        510998.95069(16) eV
#define  MUON_MASS_EV               105658375.523              13   // best known value:     105658375.5(23) eV
#define  PION_MASS_EV               134976850.0000            211   // best known value:   1349768(50).0 eV
#define  KAON_MASS_EV               493677000.0000            321   // best known value:   4936(77)000.0 eV
#define  PROTON_MASS_EV             938272310.0000           2212   // best known value:     938272088.15(29) eV
#define  DEUTERIUM_MASS_EV         1876123927.7411     1000010020   // best known value:    2.014101777844(15) u
#define  TRITIUM_MASS_EV           2809432118.1669     1000010030   // best known value:    3.016049281320(81) u
#define  He3_MASS_EV               2809413526.1471     1000020030   // best known value:    3.016029321967(60) u
#define  He4_MASS_EV               3728401325.5605     1000020040   // best known value:    4.002603254130(158) u
#define  C10_MASS_EV               9330639706.7168     1000060100   // best known value:    10.01685322(8) u
#define  C11_MASS_EV              10257084531.7151     1000060110   // best known value:    11.01143260(6) u
#define  C12_MASS_EV              11177929229.0736     1000060120   // best known value:    12 u
#define  C13_MASS_EV              12112548340.8267     1000060130   // best known value:    13.003354835336(252) u
#define  C14_MASS_EV              13043937327.9254     1000060140   // best known value:    14.003241989(4) u
#define  C15_MASS_EV              13982284805.6163     1000060150   // best known value:    15.0105993(9) u
#define  O14_MASS_EV              13048925215.1100     1000080140   // best known value:    14.008596706(27) u
#define  O15_MASS_EV              13975267171.2371     1000080150   // best known value:    15.0030656(5) u
#define  O16_MASS_EV              14899168636.5944     1000080160   // best known value:    15.994914619257(319) u
#define  O17_MASS_EV              15834590976.9790     1000080170   // best known value:    16.999131755953(692) u
#define  O18_MASS_EV              16766111027.2720     1000080180   // best known value:    17.999159612136(690) u
#define  O19_MASS_EV              17701720858.0135     1000080190   // best known value:    19.0035780(28) u
#define  O20_MASS_EV              18633678343.3555     1000080200   // best known value:    20.0040754(9) u
#define  Ne19_MASS_EV             17700140004.1889     1000100190   // best known value:    19.00188091(17) u
#define  Ne20_MASS_EV             18622840116.3475     1000100200   // best known value:    19.9924401753(16) u
#define  Ne21_MASS_EV             19555644382.6294     1000100210   // best known value:    20.99384669(4) u
#define  Ne22_MASS_EV             20484845537.9765     1000100220   // best known value:    21.991385114(19) u
#define  Ne23_MASS_EV             21419210316.0459     1000100230   // best known value:    22.99446691(11) u
#define  Ne24_MASS_EV             22349906825.6188     1000100240   // best known value:    23.9936106(6) u
// TODO: Ar, Fe, Cs, Pb, Au, Xe

#endif /* XPART_MASSES_H */
'''

fluka_masses = {
    int(line.split()[3]): [line.split()[1].split('_')[0], float(line.split()[2])]
    for line in source.split('\n')
    if len(line.split()) > 1 and len(line.split()[1]) > 6 and line.split()[1][-6:] == '_FLUKA'
}
