# Generated from npzfile: fitd1patchsample2-D1-D1_Matrix_Sample_3_NSG_full_tmp_1753.npz of fit number: 7389
# Generated from npzfile: fitd1d2-D1-D1_Patch_Sample_2_post_injection_curve_tau_and_full_charging_curve_tmp_358.npz of fit number: 10253
#Do not define EREST_ACT or ELEAK here - they are in the .p file
# Contains maximal conductances, name of .p file, and some other parameters
# such as whether to use GHK, or whether to have real spines

import numpy as np

from moose_nerp.prototypes import util as _util

#if ghkYesNo=0 then ghk not implemented
#Note that you can use GHK without a calcium pool, it uses a default of 5e-5 Cin
if False: # param_sim.Config['ghkYN']:
    ghKluge=0.35e-6
else:
    ghKluge=1

#using 0.035e-9 makes NMDA calcium way too small, using single Tau calcium
ConcOut=2e-3     # mM, default for GHK is 2e-3
Temp=30         # Celsius, needed for GHK objects, some channels

neurontypes = None

NAME_SOMA='soma'

# helper variables to index the Conductance and synapses with distance
# UNITS: meters
prox = (0, 26.1e-6)
med =  (26.1e-6, 50e-6)
dist = (50e-6, 1000e-6)
#If using swc files for morphology, can add with morphology specific helper variables
#e.g. med=(26.1e-6, 50e-6,'_2')
#_1 as soma, _2 as apical dend, _3 as basal dend and _4 as axon
#Parameters used by optimization from here down
#morph_file = {'D1':'MScell-primDend.p', 'D2': 'MScell-primDend.p'} # test_version.
morph_file = {'D1':'D1_long_matrix_1753_D1_7389.p'}
#morph_file = {'D1':'D1_patch_sample_3.p'} # old_version.

#CONDUCTANCES - UNITS of Siemens/meter squared
_D1 = _util.NamedDict(
    'D1',
    Krp = {prox:0.0322224076814701, med:0.0322224076814701, dist:0.0322224076814701},
    KaF = {prox:8830.632073798613, med:376.93279289355166, dist:37.45100766382804},
    KaS = {prox:151.2142634250082, med: 991.0831967559702, dist: 111.73232953491907},
    Kir = {prox:14.480152870026739, med: 14.480152870026739, dist: 14.480152870026739},
    CaL13 = {prox:3.2288404672427236*ghKluge, med: 0.4912001270984747*ghKluge, dist: 1.3598975643029629e-05*ghKluge},
    CaL12 = {prox:4.521378570464625*ghKluge, med: 0.948022376960737*ghKluge, dist: 3.784443442379062*ghKluge},
    CaR = {prox:4.309373703465322*ghKluge, med: 1.53635633532918*ghKluge, dist: 8.827847821134656*ghKluge},
    CaN = {prox:0.8889413610609473*ghKluge, med: 0.0*ghKluge, dist: 0.0*ghKluge},
    CaT33 = {prox:0.0*ghKluge, med: 0.08646314643834485*ghKluge, dist: 0.0002164636310379938*ghKluge},
    CaT32 = {prox:0.0*ghKluge, med: 0.005658932268133553*ghKluge, dist: 4.75526926996265*ghKluge},
    NaF = {prox:20000.046330534624, med:10721.496467328767 , dist: 2697.9740180582794},
    SKCa = {prox:0.5562017016753129, med: 0.5562017016753129, dist:0.5562017016753129 },
    BKCa = {prox:13.526810725326499, med: 13.526810725326499, dist:13.526810725326499},
    CaCC = {prox:2.6584370255396985, med: 2.6584370255396985, dist:2.6584370255396985 },
)

Condset  = _util.NamedDict(
    'Condset',
    D1 = _D1,
)
