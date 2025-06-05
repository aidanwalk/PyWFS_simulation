import numpy as np
import hcipy as hp
from tqdm import tqdm
import astropy.io.fits as fits
from astropy.table import Table
from scipy.ndimage import zoom, rotate
# import simulation modules
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix, zernike_decomposition
from ModulatedPyWFS import ModulatedWavefrontSensor
import aberrations
import stars
from pyramid_array_optic import PyramidArrayOptic



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
    # number of pyramids across one dimension in the array
    N_pyramids = 3
    # Number of unique phase screens
    N_phase_screens = 2**8
    # Size of the pyramid optic
    py_size = 1/206265
    # Number of elements across the WFS
    N_elements = 36
    # Number of iterations to average over
    iters = 10
    # -------------------------------------------------------------------------
    
    
    # Init the WFS and interaction matrix
    WFS = ModulatedWavefrontSensor(pyramidOptic = PyramidArrayOptic,
                                   focal_extent=py_size*N_pyramids,
                                   N_elements=N_elements,
                                   py_kwargs = {'N_pyramids':N_pyramids}
                                   )
    imat = interaction_matrix(N=WFS.N_elements)
    # Z_decomp = zernike_decomposition(N_modes, grid=WFS.signal_grid, 
    #                                  D=WFS.telescope_diameter,)
    
    
    # Open the common phase screen
    common_phase_arr = fits.getdata('../phase_screens/common_phase.fits')
    common_phase = hp.Field(common_phase_arr.ravel(), WFS.input_pupil_grid)
    # Zoom the common phase array so we can compare it to the recovered phase
    common_phase_arr = zoom(common_phase_arr, 
                            WFS.N_elements / common_phase_arr.shape[0], 
                            prefilter=False, order=1)
    pupil_small = WFS.circular_aperture(WFS.signal_grid.shape, WFS.signal_grid.shape[0]/ 2)
    # pupil_small = WFS.rotate(pupil_small, angle=-45, crop=False)
    pupil_small = WFS.rotate(pupil_small.astype('int'))
    common_phase_arr *= pupil_small
    
    # Since the stars fall randomly on the pyramid, performance is somewhat 
    # noisey. Average over multiple runs to get a better estimate of the error.
    errss = []
    for i in tqdm(range(iters)):
        # Init an array to store the WFS intensity signals
        signal = np.zeros((WFS.output_pupil_grid.shape))
        
        errs = []
        num_screens = []
        positions = []
        for count in range(N_phase_screens):
            # Generate a random positions for this star to fall onto the pyramid 
            position = stars.random_radius(WFS.focal_extent*206265/2, N_points=1) / 206265
            # position = stars.uniform_azimuth(0.25, N_points=24) / 206265
            
            
            # Open the unique phase screen for this star 
            unique_phase = get_phase(count)
            # Convert each phase into an HCIPy Field object
            unique_phase = hp.Field(unique_phase.ravel(), WFS.input_pupil_grid)
            
            # Create a new wavefront 
            wavefront = WFS.flat_wavefront()
            # Apply the common phase to the wavefront
            wavefront = aberrations.aberrate(wavefront, common_phase)
            # Apply the unique phases to the wavefront
            wavefront = aberrations.aberrate(wavefront, unique_phase)
            
            
            # Simulate the WFS signal and add it to the WFS signal integration
            signal += WFS.discrete_modulation(wavefront, position)
            # Rotate (to acommadate the rotated pyramid optic)
            # signal += WFS.rotate(signal_raw, crop=True)
            
            slopes = WFS.measure_slopes(signal) 
            # Wavefront phase reconstruction zonally
            recovered_phase = imat.slope2phase(*slopes)
            # Derotate the recovered phase to account for the pyramid optic rotation
            recovered_phase = WFS.rotate(recovered_phase, angle=-45)
            
                
                
            # Solve for the WFS gain (this is kind of cheating b/c we are assuming
            # we know the input wavefront phase apriori. In the real world, this is 
            # not true, and we would need to solve for the WFS gain before hand. 
            # Therefore, this simulation represents an ideal case.)
            recovered_phase *= common_phase.max() / recovered_phase.max()
            recovered_phase *= pupil_small
            
            
            err = np.std(recovered_phase - common_phase_arr) / np.std(common_phase_arr)
            errs.append(err)
            num_screens.append(count)
            positions.append(position[0])
    
        errss.append(errs)
        
    error = np.mean(errss, axis=0)
    # Write out the results to a file
    tab = Table([num_screens, error], names=['num_screens', 'error'])
    tab.write('simulation3.txt', format='ascii.fixed_width', overwrite=True)
    
    # save the final recovered phase
    fits.writeto('final_recovered_phase.fits', recovered_phase, overwrite=True)
    # Save the zoomed common phase
    fits.writeto('common_phase_zoomed.fits', common_phase_arr, overwrite=True)
