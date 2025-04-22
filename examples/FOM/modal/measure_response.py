"""
The purpose of this code is to measure the WFS response to a modulated
wavefront. 

Given a modulation radius, the output of this code is the slope and intercept 
of the linear response curve 

"""

import numpy as np
import scipy
import hcipy as hp
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import cm
plt.close('all')


# import simulation modules
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix
from ModulatedPyWFS import ModulatedWavefrontSensor
import aberrations
import stars
from pyramid_array_optic import PyramidArrayOptic
from pyramid_array_optic_2x2 import PyramidArrayOptic2x2


def plot_response_fits(response_file, order):
    """
    Plots the response curves and the fit to the data. for each column in the 
    response curves file. 

    Parameters
    ----------
    response_file : str
        file name containing the response curves data. 
    """
    # Open the file 
    response_curves = Table.read(response_file, format='ascii.fixed_width')
    
    plt.figure('fit')
    plt.title(f'{order} order poly fit')
    
    colors = cm.inferno(np.linspace(0, 1, len(response_curves.columns[:])))
    
    for i, col in enumerate(response_curves.columns[1:]):
        x = response_curves['input_WFE']
        y = response_curves[col]
        
        # Fit the data
        poly = np.polyfit(x, y, order)
        y_fit = np.polyval(poly, x)
        
        # Plot the data and the fit
        plt.scatter(x, y, color=colors[i], s=3)
        plt.plot(x, y_fit, label=f'{col}', color=colors[i], linewidth=1)
        
    plt.xlabel('RMS input WFE (rad)')
    plt.ylabel('RMS output signal')
    
    plt.legend()
    plt.savefig('response_curve_fit.png', dpi=300)
        
    return
    




    
    

def WFS_light_prop(WFS, phase, signal, wavefront, positions, fname='test.png'):
    focal_plane, WFS_signal0 = WFS.visualize_discrete_modulation(wavefront, positions)
    pyramid = WFS.pyramidOptic.pyramid.phase(800e-9).shaped
    # Make a plot of the light progression through the WFS
    fig, ax = plt.subplots(nrows=1, ncols=4, 
                           tight_layout=True, 
                           figsize=(13,3))
    plt.suptitle('WFS Light Progression')
    
    # First, plot the incoming wavefront aberration
    ax[0].set_title('Incoming Wavefront Phase')
    im = ax[0].imshow(phase, cmap='bone', origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Phase [rad]')
    # Overplot the aperture of the telescope
    # alpha = ~WFS.aperture
    # ax[0].imshow(~WFS.aperture, alpha=alpha.astype(float), cmap='Greys')
    ax[0].axis('off')
    
    
    ax[1].set_title('Focal Plane PSF')
    img = np.log10(focal_plane / focal_plane.max())
    img = hp.Field(img.ravel(), WFS.focal_grid)
    plt.subplot(142)
    im = hp.imshow_field(img, cmap='bone', vmin=-3, vmax=0, grid_units=1/206265, origin='lower') # type: ignore
    # im = ax[1].imshow(img, cmap='bone', vmin=-6, vmax=0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    
    ax[2].set_title('Pyramid Phase Mask')
    im = ax[2].imshow(pyramid, cmap='hsv', vmax=0, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Phase [rad]')
    ax[2].axis('off')
    
    ax[3].set_title('WFS Signal')
    im = ax[3].imshow(signal, cmap='bone', origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax[3].axis('off')
    
    plt.savefig(fname, dpi=300)
    return
    



if __name__ == "__main__":
    r_mod = 1
    gain_polymonial_order = 5
    N_stars = 2**8
    input_rms_WFE = 3
    # orders = np.arange(-10, -5, 1)
    orders = [-8]
    curves_file = '/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/examples/modulation/random_radius_pl_N256/response_curves.txt'
    # curves_file = '/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/examples/modulation/random_radius_N256_pl_r08/response_curves.txt'
    
    # plot_response_fits(curves_file, order)
    
    # Open the response curves file
    response_curves = Table.read(curves_file, format='ascii.fixed_width')
    # Find the column name associated to this modulation radius
    col = f'r_{r_mod:.2f}'
    # col = 'r_0.40'
    x = response_curves['input_WFE']
    y = response_curves[col]
    
    # Fit a polymonial to the data
    poly = np.polyfit(x, y, gain_polymonial_order)
    # Evaluate the polynomial
    y_fit = np.polyval(poly, x)
    # Find the WFS gain
    gain = input_rms_WFE / np.polyval(poly, input_rms_WFE)
    
    
    # -------------------------------------------------------------------------
    # We are going to create a plot with two lines:
    # 1. The modulated WFS with a single pyramid, random radius (stars are
    #    distributed randomly in the field). 
    # 2. The modulated WFS with a 2x2 pyramid array, random radius. 
    #
    # The focal extent between the two cases will be the same, so the pyramids
    # in the 2x2 array will be smaller  by a factor of 2. 
    #
    # x axis: power law phase screen order
    # y axis: RMS(output-input)
    # -------------------------------------------------------------------------
    
    # Initialize the WFSs
    WFS1 = ModulatedWavefrontSensor(pyramidOptic=PyramidArrayOptic, 
                                    focal_extent=r_mod*2/206265,
                                    N_elements=36
                                    )
    WFS2 = ModulatedWavefrontSensor(pyramidOptic=PyramidArrayOptic2x2, 
                                    focal_extent=r_mod*2/206265,
                                    N_elements=36
                                    )
    
    imat = interaction_matrix(WFS1.N_elements)
    Z = aberrations.Zernike(WFS1.input_pupil_grid, WFS1.telescope_diameter)
    
    errs1, errs2 = [], []
    for mode in list(Z.COMMON_MODES.keys())[3:]:
        print(f'order = {mode}')
        
        phasefield = Z.from_name(mode, 0.1/206265, WFS1.wavelength)
        phase = phasefield.shaped
        
        
        
        # Initialize the wavefront
        wavefront1 = WFS1.flat_wavefront()
        wavefront2 = WFS2.flat_wavefront()
        # Apply the aberration to the wavefront
        wavefront1 = aberrations.aberrate(wavefront1, phasefield)
        wavefront2 = aberrations.aberrate(wavefront2, phasefield)
        # Propagate the wavefront to the WFS
        positions1 = stars.random_radius(WFS1.focal_extent*206265/2, N_points=N_stars) / 206265
        # positions2 = stars.random_radius(WFS2.focal_extent*206265/2, N_points=N_stars) / 206265
        positions2 = positions1
        
        
        
        
        
        # Get the signal from the WFS
        signal_raw1 = WFS1.discrete_modulation(wavefront1, positions1)
        signal_raw2 = WFS2.discrete_modulation(wavefront2, positions2)
        # Rotate (to acommadate the rotated pyramid)
        signal1 = WFS1.rotate(signal_raw1, crop=True)
        signal2 = WFS2.rotate(signal_raw2, crop=True)
        
        
        # Measure slopes
        sx1, sy1 = WFS1.measure_slopes(signal1)
        sx2, sy2 = WFS2.measure_slopes(signal2)

        # Solve for the phases
        recovered_phase1 = imat.slope2phase(sx1, sy1)
        recovered_phase2 = imat.slope2phase(sx2, sy2)
        
        
        # Rotate the recovered phase
        ap = WFS1.circular_aperture((WFS1.N_elements, WFS1.N_elements), WFS1.N_elements/2)
        recovered_phase1 = WFS1.rotate(recovered_phase1, angle=-45) * ap
        # recovered_phase2 = scipy.ndimage.rotate(recovered_phase2, -45, reshape=False, order=5, prefilter=False) * ap
        recovered_phase2 = WFS2.rotate(recovered_phase2, angle=-45) * ap
        phase = scipy.ndimage.zoom(phase, WFS1.N_elements/phase.shape[0]) * ap 
        
        
        WFS_light_prop(WFS1, phase, signal1, wavefront1, positions1, 
                    fname='WFS1_light_progression.png')
        WFS_light_prop(WFS2, phase, signal2, wavefront2, positions2,
                    fname='WFS2_light_progression.png')
        
        
        # recovered_phase1 = recovered_phase1 * gain
        # recovered_phase2 = recovered_phase2 * gain
        recovered_phase1 = recovered_phase1 * phase.max() / recovered_phase1.max()
        recovered_phase2 = recovered_phase2 * phase.max() / recovered_phase2.max()
        err1 = recovered_phase1 - phase
        err2 = recovered_phase2 - phase
        print(np.std(err1))
        print(np.std(err2))
        
        
        plt.figure(figsize=(12, 7), tight_layout=True)
        plt.clf()
        
        plt.subplot(231)
        plt.title('input phase')
        im = plt.imshow(phase, origin='lower', cmap='bone', vmin=-3, vmax=3)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        plt.subplot(232)
        plt.title('recovered phase 1x1')
        im = plt.imshow(recovered_phase1, origin='lower', cmap='bone', vmin=-3, vmax=3)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        plt.subplot(233)
        plt.title('recovered phase error')
        im = plt.imshow(err1, origin='lower', cmap='bone', vmin=-3, vmax=3)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        plt.subplot(235)
        plt.title('recovered phase 2x2')
        im = plt.imshow(recovered_phase2, origin='lower', cmap='bone', vmin=-3, vmax=3)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        plt.subplot(236)
        plt.title('recovered phase error')
        im = plt.imshow(err2, origin='lower', cmap='bone', vmin=-3, vmax=3)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        FOM1 = np.std(err1)
        FOM2 = np.std(err2)
        plt.annotate(f'RMS 1x1: {FOM1:0.3f}', (0.15, 0.3), xycoords='figure fraction',)
        plt.annotate(f'RMS 2x2: {FOM2:0.3f}', (0.15, 0.25), xycoords='figure fraction',)
        fname = mode.replace(' ', '_')+'.png'
        plt.savefig(fname, dpi=200)
        
        # assert False
        
        
    
    
    
    
    
    
    plt.figure('lightprog')
    plt.title('Light propagation')
    img, sig = WFS1.visualize_discrete_modulation(wavefront1, positions1)
    plt.imshow(img, origin='lower', cmap='bone')
    plt.savefig('lightprop.png', dpi=200)
    