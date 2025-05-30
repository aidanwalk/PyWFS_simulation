The goal of this simulation is to see how well we can estimate the ground 
layer turbulence using only 11 modes: 
    tip
    tilt
    focus
    2 x astigmatism
    2 x coma
    2 x trefoil
    spherical
    1 "high" order mode
    
This replicates the LBT First Light AO (FLAO) "Enhanced Seeing Mode" (ESM)
ESM is capable of offering a factor of >2 improvement in the FWHM in even the 
worst seeing conditions. 

ESM parameters:
fq = 100 Hz
# subaps = 7x7 grid
n_modes = 11
py_mod = 6 lam/D
