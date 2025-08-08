"""
This script is used to verify the wavefront reconstruction class. 
This is seperate from the wavefront_reconstruction.py file, which verifies the 
wavefront sensor can sense and reconstruct the input wavefront.

We will generate a wavefront using the Zernike aberrations class, 
compute the gradient of the wavefront, and then reconstruct the wavefront
using the wavefront reconstruction class.


"""

import numpy as np
import hcipy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
plt.close('all')


import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix, zernike_decomposition
from PyWFS import WavefrontSensor
import aberrations



def circular_mask(grid, radius):
    """Create a circular mask for the given grid."""
    x, y = grid.x.shaped, grid.y.shaped
    mask = np.sqrt(x**2 + y**2) <= radius
    return mask



if __name__ == "__main__":
    telescope_diameter = 2.0  # meters
    # Generate a wavefront with Zernike aberrations
    N_pupil_px = 36
    pupil_grid = hp.make_pupil_grid((N_pupil_px, N_pupil_px), diameter=telescope_diameter)
    ap = circular_mask(pupil_grid, telescope_diameter/2)
    
    # Init the Zernike aberrations class
    Z = aberrations.Zernike(grid=pupil_grid, D=telescope_diameter+2*telescope_diameter/N_pupil_px)
    
    phase = Z.evaluate_named('tilt x') * 1
    
    phase = np.array(phase.shaped)
    # phase[ap==0] = np.nan
    
    # Compute the gradient of the wavefront
    sy, sx = np.gradient(phase, 1/N_pupil_px)
    sx *= ap
    sy *= ap
    phase *= ap
    phase -= np.nanmean(phase)  # Center the phase
    
    sx *= 1 / N_pupil_px
    sy *= 1 / N_pupil_px
    
    plt_kwargs = {'origin': 'lower', 'cmap': 'bone'}
    
    plt.figure(figsize=(15, 4), tight_layout=True)
    plt.subplot(141)
    plt.title('Wavefront Phase')
    plt.imshow(phase, **plt_kwargs)
    plt.colorbar(fraction=0.046, pad=0.04)

    # Plot the gradients
    plt.subplot(142)
    plt.title('sx')
    plt.imshow(sx, **plt_kwargs)
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(143)
    plt.title('sy')
    plt.imshow(sy, **plt_kwargs)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # plt.savefig('wavefront_gradient.png', dpi=300)
    
    
    imat = interaction_matrix(N_pupil_px)
    recovered_phase = imat.slope2phase(sx, sy)
    recovered_phase *= ap
    recovered_phase -= np.nanmean(recovered_phase)  # Center the phase
    
    plt.subplot(144)
    plt.title('Recovered Phase')
    plt.imshow(recovered_phase, **plt_kwargs)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.savefig('wavefront_gradient.png', dpi=300)
    
    