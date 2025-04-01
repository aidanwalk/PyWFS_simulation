"""
This is a template script for running a modulated PyWFS simulation.
This script is designed to be modified for different modulation schemes or
parameters. It provides a structure for setting up the simulation, running it,
and visualizing the results. The actual implementation of the modulation will
mostly depend on the modulation scheme (i.e. constant radius, random radius,
etc.)

@author : Aidan Walk
created on : Thu Mar 28 11:07 2025

"""

import sys
import numpy as np
import hcipy as hp
from astropy.table import Table


# Import plotting functions
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/examples/modulation/')
import plot_helper
import template


# import simulation modules
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix
from ModulatedPyWFS import ModulatedWavefrontSensor
import aberrations
import stars




if __name__ == '__main__':
    plot_title = 'Random Radius Sampling'
    points_generator=stars.random_radius
    out_file='response_curves.txt' 
    out_dir = './'
    N_stars=256
    modulation_radii=np.linspace(0, 0.4, 11)
    input_WFE = np.linspace(0, 0.5, 41)
    WFS_kwargs={}
    N_iters=1
    
    
    # Init the wavefront sensor and the interaction matrix
    WFS = ModulatedWavefrontSensor(**WFS_kwargs)
    imat = interaction_matrix(WFS.N_elements)
    Z = aberrations.Zernike(WFS.input_pupil_grid, WFS.telescope_diameter)
    
    
    
    template.visualize_modulation(radius=0.4, 
                                  WFS=WFS,
                                  points_generator=points_generator,
                                  N_stars=N_stars,
                                  out_dir=out_dir,
                                  title=plot_title,
                                  )
    
    
    template.verify_reconstruction(radius=0.4, 
                                   WFS=WFS,
                                   points_generator=points_generator,
                                   N_stars=N_stars,
                                   WFE=0.1/206265,
                                   out_dir='./', 
                                   title=plot_title
                                )
    
    
    # template.make_response_table(modulation_radii, 
    #                     points_generator,
    #                     N_stars=N_stars,
    #                     input_WFE=input_WFE, # type: ignore
    #                     out_dir=out_dir,
    #                     out_file='response_curves.txt',
    #                     WFS=WFS,
    #                     N_iters=N_iters,
    #                     Z=Z, 
    #                     )
    
    plot_helper.plot_response(data_file=out_dir+out_file, 
                              title=plot_title, 
                              fname=out_dir+'response.png')

    
    
    
