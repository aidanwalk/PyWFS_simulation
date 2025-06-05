import numpy as np
import hcipy as hp
from tqdm import tqdm
import astropy.io.fits as fits
from astropy.table import Table
from scipy.ndimage import zoom
# import simulation modules
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix, zernike_decomposition
from ModulatedPyWFS import ModulatedWavefrontSensor
import aberrations




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



if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # SIMULATION PARAMETERS -- CHANGE ME
    # -------------------------------------------------------------------------
    # Total number of WFSs
    N_WFSs = 1
    # Number of unique phase screens
    N_phase_screens = 2**8
    # Size of the pyramid optic
    py_size = 2/206265
    # Pyramid modulation radius in radians
    modulation_radius = 0.5/206265 # radians
    # Number of elements across the WFS
    N_elements = 36
    # Number of modes for the modal wavefront reconstruction
    N_modes = 11 # Use 11 modes, since zernike_decomposition starts at mode 1, 
                 # going up to mode 11 includes spherical aberration
    # -------------------------------------------------------------------------
    
    
    # Init the WFS and interaction matrix
    WFS = ModulatedWavefrontSensor(focal_extent=py_size,
                                   N_elements=N_elements,)
    imat = interaction_matrix(N=WFS.N_elements)
    Z_decomp = zernike_decomposition(N_modes, grid=WFS.signal_grid, 
                                     D=WFS.telescope_diameter,)
    
    
    # Open the common phase screen
    common_phase_arr = fits.getdata('../phase_screens/common_phase.fits')
    common_phase = hp.Field(common_phase_arr.ravel(), WFS.input_pupil_grid)
    # Zoom the common phase array so we can compare it to the recovered phase
    common_phase_arr = zoom(common_phase_arr, 
                            WFS.N_elements / common_phase_arr.shape[0], 
                            prefilter=False, order=1)
    pupil_small = WFS.circular_aperture(WFS.signal_grid.shape, WFS.signal_grid.shape[0]/ 2)
    common_phase_arr *= pupil_small
    
    # Calculate the number of screens per WFS
    screens_per_WFS = N_phase_screens // N_WFSs
    # Init an array to store the WFS intensity signals
    signals = np.zeros((N_WFSs, *WFS.output_pupil_grid.shape))
    
    errs = []
    num_screens = []
    for count in tqdm(range(screens_per_WFS)):
        # FOR EACH WFS
        recovered_phases = []
        for i in range(N_WFSs):
            
            # Open the unique phase screen for this iteration and WFS
            # This is the index of the unique phase screen for this WFS
            # The first WFS will have screens 0 to screens_per_WFS - 1
            # The second WFS will have screens screens_per_WFS to 2 * screens_per_WFS - 1
            # The third WFS will have screens 2 * screens_per_WFS to 3 * screens_per_WFS - 1
            # and so on...
            start = i * screens_per_WFS
            index = count + start
            unique_phase = get_phase(index)
            # Convert each phase into an HCIPy Field object
            unique_phase = hp.Field(unique_phase.ravel(), WFS.input_pupil_grid)
            
            # Create a new wavefront 
            wavefront = WFS.flat_wavefront()
            # Apply the common phase to the wavefront
            wavefront = aberrations.aberrate(wavefront, common_phase)
            # Apply the unique phases to the wavefront
            wavefront = aberrations.aberrate(wavefront, unique_phase)
            # SImulate the WFS signal and add it to the WFS signal integration
            signals[i] += WFS.modulate(wavefront, radius=modulation_radius,
                                        num_steps = 12)
            slopes = WFS.measure_slopes(signals[i]) 
            # Wavefront phase reconstruction zonally
            zonal_phase = imat.slope2phase(*slopes)
            
            # Perform a modal decomposition on the recovered phase
            coeffs = Z_decomp.decompose(zonal_phase.flatten())
            # Reconstruct the phase just based on the Zernike decomposition
            projection = Z_decomp.project(coeffs).shaped
            
            # Append the zernike projected phase to the recovered phases
            recovered_phases.append(projection)
            
        mean_phase = np.mean(recovered_phases, axis=0)
        # mean_phase = recovered_phases[0]
        # Solve for the WFS gain (this is kind of cheating b/c we are assuming
        # we know the input wavefront phase apriori. In the real world, this is 
        # not true, and we would need to solve for the WFS gain before hand. 
        # Therefore, this simulation represents an ideal case.)
        mean_phase *= common_phase.max() / mean_phase.max()
        mean_phase *= pupil_small
        
        
        err = np.std(mean_phase - common_phase_arr) / np.std(common_phase_arr)
        errs.append(err)
        num_screens.append((count* N_WFSs) + N_WFSs)
        
    
    # Write out the results to a file
    tab = Table([num_screens, errs], names=['num_screens', 'error'])
    tab.write('simulation2.txt', format='ascii.fixed_width', overwrite=True)
    
    # save the final recovered phase
    fits.writeto('final_recovered_phase.fits', mean_phase, overwrite=True)
    # Save the zoomed common phase
    fits.writeto('common_phase_zoomed.fits', common_phase_arr, overwrite=True)