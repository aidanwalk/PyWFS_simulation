import numpy as np
import hcipy as hp
from tqdm import tqdm
import astropy.io.fits as fits
from matplotlib import animation
import matplotlib.pyplot as plt
plt.close('all')
from astropy.table import Table
from scipy.ndimage import zoom
# import simulation modules
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix
from ModulatedPyWFS import ModulatedWavefrontSensor
import aberrations

sys.path.append('/home/arcadia/mysoft/gradschool/useful/code_fragments/strehlometer/')
import strehl

from photutils.profiles import CurveOfGrowth


def get_phases(start, end):
    phases = []
    for num in range(start, end):
        # Load the unique phase screen for this WFS
        fname = f'../phase_screens/random_phase_{num:03d}.fits'
        phases.append(fits.getdata(fname))
        
    return phases


def get_phase(index):
    # Load the unique phase screen for this WFS
    fname = f'../phase_screens/random_phase_{index:03d}.fits'
    return fits.getdata(fname)



def make_atmospheres(N):
    atmospheres = []
    for w in range(N):
        # Create a layer for this atmosphere
        print(f'Creating atmosphere layers for WFS {w+1} of {N_WFSs}...')
        # Seed the layers such that the ground layer is always the same,
        # but the upper layers are different.
        layers = aberrations.make_unmoving_ground_layers(WFS.input_pupil_grid, 
                                                  seeds=[2, *np.arange(3+(w*7), 3+(w*7)+6)])
        atmospheres.append(hp.atmosphere.MultiLayerAtmosphere(layers))
    return atmospheres



def simulate_this_WFS(wfs_index, step, atmospheres, WFS):
    # Advance the atmosphere for this timestep and WFS
    atmospheres[wfs_index].evolve_until(dt * step)
    atmospheres[wfs_index].layers[0] = atmospheres[0].layers[0]  # Keep the ground layer the same for all WFSs
    
    # Create a new wavefront 
    wavefront = WFS.flat_wavefront()
    # Pass the wavefront through the atmosphere
    wavefront = atmospheres[wfs_index].forward(wavefront)
    
    # Simulate the WFS signal and add it to the WFS signal integration
    signals[wfs_index] += WFS.modulate(wavefront, radius=modulation_radius,
                                num_steps = 12)
    # Compute the WFS slopes
    slopes = WFS.measure_slopes(signals[wfs_index]) 
    # Recover the wavefront phase
    recovered_phase = imat.slope2phase(*slopes)
    
    return recovered_phase



def compute_strehl(atmosphere, mean_phase):
    """Compute the Strehl ratio for two cases: 
        1. uncorrected (just the atmosphere unaltered)
        2. corrected (mean_phase is removed from the atmosphere ground layer)
    """
    flat_wavefront = WFS.flat_wavefront()
    fp_perfect = WFS.pupil2image(flat_wavefront).intensity.shaped
    
    uncorrected_wavefront = atmosphere.forward(flat_wavefront.copy())
    fp_uncorrected = WFS.pupil2image(uncorrected_wavefront).intensity.shaped
    
    # apply the negative phase to the uncorrected wavefront
    corrected_wavefront = aberrations.aberrate(uncorrected_wavefront.copy(), -mean_phase.flatten())
    fp_corrected = WFS.pupil2image(corrected_wavefront).intensity.shaped
    
    
    fp_perfect = np.array(fp_perfect)
    fp_uncorrected = np.array(fp_uncorrected)
    fp_corrected = np.array(fp_corrected)
    
    values = []
    for image in [fp_perfect, fp_uncorrected, fp_corrected]:
        st = strehl.strehl(image, fp_perfect, 
                           photometryradius=WFS.focal_extent/2*206265, 
                           pixelscale = WFS.focal_extent / WFS.Npx_foc*206265)
        values.append(st)
    
    
    # print(f'Strehl: {values}')
    return values



def save(animation_frames):
    """Save the animation frames to a file."""
    # Save the animation frames as a numpy array
    np.save('animation_frames.npy', animation_frames)
    return
    


def peak_location(image):
    """Find the peak location in the image."""
    # maxloc = np.where(image == np.max(image))
    # xc = maxloc[1][0]
    # yc = maxloc[0][0]
    
    xc = image.shape[0] // 2
    yc = image.shape[1] // 2
    
    return xc, yc



def find_ee50(image):
    """Find the radius at which the encircled energy is 50%."""
    # find the peak of the image
    maxloc = peak_location(image)
    # maxloc = image.shape[0] // 2, image.shape[1] // 2
    cog = CurveOfGrowth(image, xycen=maxloc, radii=range(3, int(WFS.Npx_foc/2), 1))
    
    ee50 = cog.calc_radius_at_ee(0.5)
    
    return ee50


def compute_ee50(atmosphere, mean_phase):
    """Compute the encircled energy at 50% for two cases: 
        1. uncorrected (just the atmosphere unaltered)
        2. corrected (mean_phase is removed from the atmosphere ground layer)
    """
    flat_wavefront = WFS.flat_wavefront()
    fp_perfect = WFS.pupil2image(flat_wavefront).intensity.shaped
    
    uncorrected_wavefront = atmosphere.forward(flat_wavefront.copy())
    fp_uncorrected = WFS.pupil2image(uncorrected_wavefront).intensity.shaped
    
    # apply the negative phase to the uncorrected wavefront
    corrected_wavefront = aberrations.aberrate(uncorrected_wavefront.copy(), -mean_phase.flatten())
    fp_corrected = WFS.pupil2image(corrected_wavefront).intensity.shaped
    
    
    fp_perfect = np.array(fp_perfect/np.sum(fp_perfect))
    fp_uncorrected = np.array(fp_uncorrected/np.sum(fp_uncorrected))
    fp_corrected = np.array(fp_corrected/np.sum(fp_corrected))
    
    frames = [fp_perfect, fp_uncorrected, fp_corrected]
    values = []
    for image in frames:
        rad50 = find_ee50(image)
        # print(rad50)
        values.append(rad50)
    
    return values, frames
    
# %%


# =============================================================================
# SIMULATION
# =============================================================================

# %% [Simulation Parameters]
# -------------------------------------------------------------------------
# SIMULATION PARAMETERS -- CHANGE ME
# -------------------------------------------------------------------------
# Total number of WFSs
N_WFSs = 4
# Fq of WFS (measurements per second)
Hz = 50
# simulation time step (number of steps per second)
dt = 1 / 100 * 1/Hz
# Size of the pyramid optic
py_size = 3/206265
# Pyramid modulation radius in radians
modulation_radius = 0.5/206265 # radians
# -------------------------------------------------------------------------

# Total number of simulation steps
N_measurements = int(Hz**-1 / dt)


# %% [Create atmosphere and WFS]

# Init the WFS and interaction matrix
WFS = ModulatedWavefrontSensor(focal_extent=py_size)
imat = interaction_matrix(N=WFS.N_elements)


# Make an atmospheric model for each WFS
# Each WFS will have its own unique phase screens, but the ground layer
# will be the same for all WFSs.
atmospheres = make_atmospheres(N_WFSs)
# for i in range(len(atmospheres)):
#     atmospheres[i].reset()

# %% [Simulation Loop]

# Create a circular aperture in WFS signal grid units
pupil_small = WFS.circular_aperture(WFS.signal_grid.shape, WFS.signal_grid.shape[0]/ 2)


# Init an array to store the WFS intensity signals
signals = np.zeros((N_WFSs, *WFS.output_pupil_grid.shape))
# And the ground layer image at each time step
ground_layers = np.zeros((N_measurements, *WFS.input_pupil_grid.shape))



recovered_ground_layer = []
time_steps = []
errs = []
uncorrected_strehl = []
corrected_stehl = []
animation_frames = np.zeros((N_measurements, 3, *WFS.focal_grid.shape))
for step in tqdm(range(N_measurements)):
    # FOR EACH WFS
    recovered_phases = []
    for i in range(N_WFSs):
        recovered_phase = simulate_this_WFS(i, step, atmospheres, WFS)
        recovered_phases.append(recovered_phase)
        
        if i == 0:
            # Save the ground layer for this timestep
            ground_layers[step] = atmospheres[i].layers[0].phase_for(WFS.wavelength).shaped
    
    
    # Compute the mean phase over each WFS
    mean_phase = np.mean(recovered_phases, axis=0)
    recovered_ground_layer.append(mean_phase)
    time_steps.append(step*dt)
    
    
    
    # Compute the average ground layer over the simulation so far
    ground_layer = np.mean(ground_layers, axis=0)
    # Zoom the ground layer to match the number of WFS elements
    ground_layer = zoom(ground_layer, WFS.N_elements / ground_layer.shape[0])
    ground_layer *= pupil_small
    
    # Apply the WFS gain to the recovered phase
    mean_phase *= np.abs(ground_layer).max() / np.abs(mean_phase).max()
    mean_phase *= pupil_small
    
    # Compute the error between the recovered phase and the ground layer
    err = np.std(mean_phase - ground_layer) / np.std(ground_layer)
    errs.append(err)
    
    # Conpute the strehl ratio for this timestep
    big_mean_phase = zoom(mean_phase, WFS.input_pupil_grid.shape[0] / mean_phase.shape[0])
    # p_st, c_st, uc_st = compute_strehl(atmospheres[0], big_mean_phase)
    ee50, frames = compute_ee50(atmospheres[0], big_mean_phase)
    corrected_stehl.append(ee50[2])
    uncorrected_strehl.append(ee50[1])
    
    animation_frames[step] = frames
    
    
# %%
# plot_frames(animation_frames)
save(animation_frames)
    
    
# %%
# Write out the results to a file
tab = Table([time_steps, errs, uncorrected_strehl, corrected_stehl],
            names=['num_screens', 'error', 'uncorrected_ee50', 'corrected_ee50']) 
tab.write('simulation1.txt', format='ascii.fixed_width', overwrite=True)

# save the final recovered phase
fits.writeto('final_recovered_phase.fits', mean_phase, overwrite=True)
# Save the zoomed common phase
fits.writeto('common_phase_zoomed.fits', ground_layer, overwrite=True)
    
    