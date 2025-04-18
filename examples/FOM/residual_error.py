"""
We want to see the RMS wavefront error between the input and output -- 
How is the error related to the spatial order of the reconstructed wavefront? 

@author : Aidan Walk
created on : Thu Mar 28 11:07 2025

"""

import numpy as np
import hcipy as hp
# import simulation modules
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix
from ModulatedPyWFS import ModulatedWavefrontSensor
import aberrations
import stars



if __name__ == "__main__":
    radius = 0.4  # Modulation radius in arcseconds
    N_stars = 12
    WFE = 0.1/206265
    modulation_points = stars.random_radius(radius, N_points=N_stars) /206265
    powers = np.linspace(-15, -1, 3)
    
    
    
    pupil = ModulatedWavefrontSensor.circular_aperture((32,32),16)
    # Init the wavefront sensor
    WFS = ModulatedWavefrontSensor(pupil)
    
    # Initialize the Zernike class 
    Z = aberrations.Zernike(WFS.input_pupil_grid,
                            WFS.telescope_diameter)

    # Create the interaction matrix
    imat = interaction_matrix(WFS.N_elements)

    input_phases = []
    output_phases = []
    for p in powers:
        phase = aberrations.make_noise_pl(2, 
                                       WFS.pupil.shape[0],
                                       WFS.pupil.shape[0], 
                                       p, 
                                       WFS.N_elements**2).ravel()
        phase = hp.Field(phase, WFS.input_pupil_grid)


        # Create a wavefont incoming to the WFS
        incoming_wavefront = WFS.flat_wavefront()
        incoming_wavefront = aberrations.aberrate(incoming_wavefront, phase)
        input_phases.append(phase.shaped)
        # Pass the wavefront through the WFS
        signal = WFS.discrete_modulation(incoming_wavefront,
                                         modulation_points)
        
        # Recover the slopes
        sx, sy = WFS.measure_slopes(signal)
        # Use it to solve for phases
        recovered_phase = imat.slope2phase(sx, sy)
        output_phases.append(recovered_phase)
        
        
    
