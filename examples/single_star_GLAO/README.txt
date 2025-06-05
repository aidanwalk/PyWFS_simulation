For each simulation case, the wavefront recovery error is plotted as a function
of the number of upper-layer phase screens averaged over. 

make_phase_screens.py
This script is used to generate random sets of phase screens so that each 
simulation can use a standard (common) set of turbulence. Random phase screens 
are saved as .FITS files to ./phase_screens/. The simulations can then reference
the same standard set of phase screens. 

-------------------------------------------------------------------------------
SIMULATION 1
-------------------------------------------------------------------------------
Reconstruct the ground layer in a "traditional" GLAO approach. 
- There are four individual wavefront sensors 
- Each WFS has its own star
- pyramids are modulated by 0.5 arcsec
- each star experiences a unique set of random phase screens (e.g. 10 random 
  phase screens, RMS WFE = 1.5 rad [should be P-V = ~6 rad]), and one "common" 
  lower order phase screen (RMS WFE = 5 rad [should be P-V ~20 rad]). 
- Total phase screens averaged over = unique phase screens per star * N_stars
- The common phase "i.e. ground layer" is recovered by averaging the four WFS 
  signals. 
  
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
SIMULATION 2
-------------------------------------------------------------------------------
The goal of this simulation is to see how well we can estimate the ground 
layer turbulence using only 11 modes and a single star: 
    tip
    tilt
    focus
    2 x astigmatism
    2 x coma
    2 x trefoil
    spherical
    1 "high" order mode -- I don't know what this means, leaving it out for now. 
    The simulation will only contain the first 10 modes. 
    
This replicates the LBT First Light AO (FLAO) "Enhanced Seeing Mode" (ESM)
ESM is capable of offering a factor of >2 improvement in the FWHM in even the 
worst seeing conditions. 

like in simulation 1:
- The star experiences a set of random phase screens (10 random phase 
  screens, RMS WFE = 2 rad [should be P-V = 6 rad]), and one lower 
  order phase screen (RMS WFE = 10 rad [should be P-V 20 rad]). 


ESM parameters:
    fq = 100 Hz
    N_subaps = 7x7 grid
    n_modes = 11
    py_mod = 6 lam/D
    
Simulation Parameters:
    py_mod = 0.25 arcsec
    N_subaps = 36x36 grid
    n_modes = 10
    
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
SIMULATION 3
-------------------------------------------------------------------------------
The purpose of this simulation is to see how well an NxN pyramid array can 
reconstruct the ground layer. 

For a direct comparison to simulation 1, 40 stars fall onto the pyramid array. 
Instead of having four stars with 10 unique phase screens each, we have 40 
stars, each with a single unique phase screen along with the lower order 
common phase screen. 

The number of stars falling onto the pyramid array is a function of the number 
of phase screens we are averaging over. N_stars = Number of unique phase screens
to average over. 


MASKED EDGES SIMULATION
The pyramid array is 1x1 (only a single pyramid). This simulates the case where
all stars in the field that would fall on a negative pyramid facet have been 
masked out. Stars are restricted to falling within 1" of the pyramid tip. 
Therefore, on average, a star should fall within 0.5" of the pyramid tip -->
an effective modulation of 0.5" like simulations 1 and 2. 

UNMASKED EDGES SIMULATION
The pyramid array is 2x2, pyramids are 1" in width and length

To match simulations 1 and 2, the this simulation will have a 2x2 pyramid array
with pyramid pitches equal to 1 arcsec per pyramid. This should (I think) be 
effectively equivalent to a 0.5 arcsecond modulation. 

-------------------------------------------------------------------------------
