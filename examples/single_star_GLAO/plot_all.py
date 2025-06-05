import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import hcipy as hp
import astropy.io.fits as fits
from astropy.table import Table
from scipy.ndimage import zoom
# import simulation modules
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from ModulatedPyWFS import ModulatedWavefrontSensor
import aberrations


def plot_focal_plane(fname='focal_plane.png'):
    # Init the WFS
    WFS = ModulatedWavefrontSensor(focal_extent=2/206265,)
    # Create a wavefront with no aberrations
    wavefront = WFS.flat_wavefront()
    
    # Open the common phase screen
    common_phase = fits.getdata('./phase_screens/common_phase.fits')
    # Open the unique phase screen
    unique_phase = fits.getdata('./phase_screens/random_phase_000.fits')
    
    # Put the waveront phases into a Field
    common_phase = hp.Field(common_phase.ravel(), WFS.input_pupil_grid)
    unique_phase = hp.Field(unique_phase.ravel(), WFS.input_pupil_grid)
    
    wavefront_common = aberrations.aberrate(wavefront.copy(), common_phase)
    wavefront_unique = aberrations.aberrate(wavefront.copy(), unique_phase)
    
    # pupil_image, focal_image, pyramid_image, WFS_signal
    flat = WFS.light_progression(wavefront)[1]
    common = WFS.light_progression(wavefront_common)[1]
    unique = WFS.light_progression(wavefront_unique)[1]
    
    # flat = np.log10(flat / flat.max())
    # common = np.log10(common / common.max())
    # unique = np.log10(unique / unique.max())
    
    # Plot the results
    plt.figure(figsize=(12,4), tight_layout=True)
    plt.suptitle('WFS Focal Plane Images, Linear Scale', fontsize=16)
    
    plt.subplot(131)
    plt.title('Flat Wavefront')
    kwargs = {'cmap':'bone',
              'origin':'lower',
              'grid_units':1/206265
              }
    im = hp.imshow_field(hp.Field(flat.ravel(), WFS.focal_grid), **kwargs)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel('Focal Plane X (arcsec)')
    plt.ylabel('Focal Plane Y (arcsec)')
    
    plt.subplot(132)
    plt.title('Common Phase')
    im = hp.imshow_field(hp.Field(common.ravel(), WFS.focal_grid), **kwargs)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel('Focal Plane X (arcsec)')
    plt.ylabel('Focal Plane Y (arcsec)')
    
    plt.subplot(133)
    plt.title('Unique Phase')
    im = hp.imshow_field(hp.Field(unique.ravel(), WFS.focal_grid), **kwargs)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel('Focal Plane X (arcsec)')
    plt.ylabel('Focal Plane Y (arcsec)')
    
    # plt.show()
    # Save the results
    plt.savefig(fname, dpi=300)
    return




def plot_errors(fname='performance.png'):
    files = ['./simulation1/simulation1.txt', 
             './simulation2/simulation2.txt', 
             './simulation3/simulation3.txt', 
             './simulation3_unmasked_edges/simulation3.txt']
    
    names = ['GLAO', 'ESM', '$PyArr_{masked}$', '$PyArr_{unmasked}$']
    linestyles = ['-', '--', ':', ':']
    colors = ['k', 'k', 'k', 'k']
    alphas = [1, 1, 1, 0.5]
    
    plt.figure(figsize=(6, 4), tight_layout=True)
    plt.suptitle('WFS Performance')
    
    for i, file in enumerate(files):
        data = Table.read(file, format='ascii.fixed_width')
        
        plt.plot(data['num_screens'], data['error'],
                 label=names[i], linestyle=linestyles[i], 
                 color=colors[i], alpha=alphas[i], linewidth=1)
    
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    # plt.xscale('log')
    plt.xlabel('Number of Phase Screens Averaged')
    plt.ylabel('Relative RMS Error')
    # plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=10, frameon=True)
    
    plt.savefig(fname, dpi=300)
    return
        




if __name__ == "__main__":
    plot_focal_plane(fname='focal_plane.png')
    plot_errors(fname='performance.png')