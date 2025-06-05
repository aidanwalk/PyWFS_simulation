"""
This is a modified version of the hcipy 
wavefront_sensing.PyramidWavefrontSensorOptics class. It is modified such
that the pyramid optic is an array of pyramids, instead of a single 
pyramid.


@author: Aidan Walk
Created on 01 April 2025

"""


import numpy as np
from scipy.ndimage import zoom

import hcipy as hp
from hcipy.wavefront_sensing import WavefrontSensorOptics
from hcipy.optics import SurfaceApodizer, Apodizer
from hcipy.field import make_pupil_grid, make_focal_grid, Field
from hcipy.propagation import FraunhoferPropagator




class PyramidArrayOptic(WavefrontSensorOptics):
    '''The optical elements for a pyramid wavefront sensor.

    Parameters
    ----------
    input_grid : Grid
        The grid on which the input wavefront is defined.
    N_pyramids : int
        The number of pyramids along each dimension in the array. The total 
        number of pyramids is N_pyramids**2. The default is 2. 
    separation : scalar
        The separation between the pupils. The default takes the input grid extent as separation.
    wavelength_0 : scalar
        The reference wavelength that determines the physical scales.
    q : scalar
        The focal plane oversampling coefficient. The default uses the minimal required sampling.
    refractive_index : callable
        A callable that returns the refractive index as function of wavelength.
        The default is a refractive index of 1.5.
    num_airy : scalar
        The radius of the focal plane spatial filter in units of lambda/D at the reference wavelength.
    '''
    def __init__(self, input_grid, N_pyramids=1, separation=None, wavelength_0=1.0, q=None, num_airy=None, refractive_index=lambda x: 1.5, rotated=True):
        if not input_grid.is_regular:
            raise ValueError('The input grid must be a regular grid.')

        self.rotated = rotated
        self.input_grid = input_grid
        D = np.max(input_grid.delta * (input_grid.shape - 1))

        if separation is None:
            separation = D

        # Oversampling necessary to see all frequencies in the output wavefront sensor plane
        qmin = max(2 * separation / D, 1)
        if q is None:
            q = qmin
        elif q < qmin:
            raise ValueError('The requested focal plane sampling is too low to sufficiently sample the wavefront sensor output.')

        if num_airy is None:
            self.num_airy = np.max(input_grid.shape - 1) / 2
        else:
            self.num_airy = num_airy

        self.focal_grid = make_focal_grid(q, self.num_airy, reference_wavelength=wavelength_0, pupil_diameter=D, focal_length=1)
        # Increase the size of the output grid by a factor of sqrt(2) to 
        # accommodate the rotated pyramid optic
        self.output_grid = make_pupil_grid(qmin * input_grid.dims*2**0.5, qmin * D*2**0.5)
        
        
        # Make all the optical elements
        # -----------------------------
        
        # What is the point of this spatial filter? I do not think it is needed
        # for the pyramid array. 
        # self.spatial_filter = Apodizer(circular_aperture(2 * self.num_airy * wavelength_0 / D)(self.focal_grid))
        
        self.pyramid_array = self.make_pyramid_array(N_pyramids, refractive_index, wavelength_0, separation)
        
        self.pyramid = SurfaceApodizer(Field(self.pyramid_array.ravel(), self.focal_grid), refractive_index)

        # Make the propagators
        self.pupil_to_focal = FraunhoferPropagator(input_grid, self.focal_grid)
        self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.output_grid)
        
        
    def make_pyramid_array(self, N_pyramids, refractive_index, wavelength_0, separation):
        '''Creates a pyramid array with the specified number of pyramids.

        Parameters
        ----------
        N_pyramids : int
            The number of pyramids along each dimension in the array. The total 
            number of pyramids is N_pyramids**2. The default is 2. 

        Returns
        -------
        pyramid_array : np.ndarray
            The pyramid array.
        '''
        # Create the pyramid surface on a rotated focal grid, since the default
        # hcipy is really a pyramid optic with its corners cut off. 
        # This will be a complete pyramid optic with no cut corners, but 
        # rotated by 45 degrees so the whole pyramid is contained by a square.
        x, y = self.focal_grid.rotated( np.pi/4 ).points.T
        
        # self.pyramid_surface = -separation / (2 * (refractive_index(wavelength_0) - 1)) * (np.abs(self.focal_grid.x) + np.abs(self.focal_grid.y))
        self.pyramid_surface = -separation / (2 * (refractive_index(wavelength_0) - 1)) * (np.abs(x) + np.abs(y))
        
        # Reshape the pyramid surface to be a 2D array
        pyramid_surface = self.pyramid_surface.reshape(self.focal_grid.shape)
        width, height = pyramid_surface.shape
        
        # Initialize the pyramid array (we are going to insert four pyramids
        # into the array)
        # Initially we make this a factor of N_pyramids too large. This enables
        # us to insert pyramids into the array without having to worry about 
        # rounding errors / centering pyramids at the sub-pixel level. 
        pyramid_array = np.zeros(self.focal_grid.shape*N_pyramids)
        
        # Find the points at which the top corner of the pyramid will be placed
        # in the array
        i = np.linspace(0,  width*N_pyramids, N_pyramids, endpoint=False)
        j = np.linspace(0, height*N_pyramids, N_pyramids, endpoint=False)
        pyramid_locs = [(x,y) for x in i for y in j]
        
        for point in pyramid_locs:
            # Insert the pyramid surface into the array
            x = int(point[0])
            y = int(point[1])
            # the pyramid array phase is divided by the number of pyramids to
            # reduce the amount of phase introduced by the pyramid. 
            # This works because the final pyramid array is reduced in size
            # by a factor of N_pyramids, so this is effectively the same as 
            # making a pyramid that was originally a factor of N smaller. 
            # i.e. instead of making the pyramid full sized, cropping it by a 
            # factor of N_pyramids, then inserting it into the array, we make a
            # pyramid that has a phase delay of N_pyramids times 
            # less than the full sized pyramid, and crop it down later. 
            pyramid_array[x:x+width, y:y+height] = pyramid_surface/N_pyramids
        
        # Finally, reduce the size of the pyramid array to the same size as 
        # the focal grid (it was initially oversized by a factor of N_pyramids)
        pyramid_array = zoom(pyramid_array, 1/N_pyramids, prefilter=False)
        
        return pyramid_array
    
    

    def forward(self, wavefront):
        '''Propagates a wavefront through the pyramid wavefront sensor.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf_wfs : Wavefront
            The output wavefront.
        '''
        wf_focus = self.pupil_to_focal.forward(wavefront)
        # wf_pyramid = self.pyramid.forward(self.spatial_filter.forward(wf_focus))
        wf_pyramid = self.pyramid.forward(wf_focus)
        wf_wfs = self.focal_to_pupil.forward(wf_pyramid)

        return wf_wfs

    def backward(self, wavefront):
        '''Propagates a wavefront backwards through the pyramid wavefront sensor.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf_pupil : Wavefront
            The output wavefront.
        '''
        wf_focus = self.focal_to_pupil.backward(wavefront)
        wf_pyramid = self.pyramid.backward(self.spatial_filter.backward(wf_focus))
        wf_pupil = self.pupil_to_focal.backward(wf_pyramid)

        return wf_pupil
        







def plot_phases(input_phases, output_phases, fname='phase_comparison.html', title="input vs recovered phase"):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    plt.suptitle(title)
    for i in range(len(input_phases)):
        input_phase = input_phases[i]
        recovered_phase = output_phases[i]
        # Create a plot for the x-slope
        pltkwargs = {'origin':'lower',
                    'cmap':'bone',
                    }
        im = axs[0,i].imshow(input_phase, **pltkwargs)
        plt.colorbar(im,fraction=0.046, pad=0.04)
        axs[0,i].axis('off')
        im = axs[1,i].imshow(recovered_phase, vmin=-recovered_phase.max(), 
                            vmax=recovered_phase.max(), 
                            **pltkwargs
                            )
        plt.colorbar(im,fraction=0.046, pad=0.04)
        axs[1,i].axis('off')
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    
    return



def verify_reconstruction(
                          out_dir='./', 
                          title='Random Radius Reconstruction'):
    
    
    # Initialize zernike aberrations class
    Z = aberrations.Zernike(WFS.input_pupil_grid, WFS.telescope_diameter)
    z1 = Z.from_name('tilt y', WFE=WFE*WFS.telescope_diameter/2, wavelength=WFS.wavelength)
    z1 += Z.from_name('tilt x', WFE=WFE*WFS.telescope_diameter/2, wavelength=WFS.wavelength)
    z2 = Z.from_name('spherical', WFE=WFE, wavelength=WFS.wavelength)
    z3 = aberrations.make_noise_pl(2, 
                                   WFS.pupil.shape[0],
                                   WFS.pupil.shape[0], 
                                   -5, 
                                   WFS.N_elements**2).ravel()
    z3 = hp.Field(z3, WFS.input_pupil_grid)
    
    aberrs = [z1, z2, z3]
    
    
    # Create the interaction matrix
    imat = interaction_matrix(WFS.N_elements)

    
    input_phases = []
    output_phases = []
    for i, phase in enumerate(aberrs):
        # Create a wavefont incoming to the WFS
        incoming_wavefront = WFS.flat_wavefront()
        incoming_wavefront = aberrations.aberrate(incoming_wavefront, phase)
        input_phases.append(phase.shaped)
        # Pass the wavefront through the WFS
        # signal = WFS.pass_through(incoming_wavefront)
        signal = WFS.discrete_modulation(incoming_wavefront, positions)
        
        
        # Rotate the WFS signal to accommodate the pyramid orientation
        signal = scipy.ndimage.rotate(signal, 45, reshape=False)
        # crop the WFS_signal by a factor of 2**0.5 to match the original input
        # grid size. 
        crop_size = round(signal.shape[0] / 2**0.5)
        center=False
        # If the crop size is odd, need to add one pixel to the cropping, otherwise 
        # the output will be off by one pixel. 
        if crop_size % 2 == 1: center=True
        signal = signal[
            signal.shape[0]//2 - crop_size//2 : signal.shape[0]//2 + crop_size//2+center,
            signal.shape[1]//2 - crop_size//2 : signal.shape[1]//2 + crop_size//2+center
        ]
        
        
        
        # Recover the slopes
        sx, sy = WFS.measure_slopes(signal)
        # Use it to solve for phases
        recovered_phase = imat.slope2phase(sx, sy)
        recovered_phase = scipy.ndimage.rotate(recovered_phase, -45, reshape=False) *ap
        output_phases.append(recovered_phase)
    
    # Make a plot of the recovered phase
    plot_phases(input_phases, output_phases,
                            fname=out_dir+'reconstruction.png', 
                            title='Rotated Pyramid Array Optic Reconstruction')
    return



if __name__ == "__main__":
    print("PyramidArrayOptic module loaded successfully.")
    # You can add test cases or example usage here if needed.
    # Example usage
    import matplotlib.pyplot as plt
    plt.close('all')
    import scipy
    # from PyWFS import WavefrontSensor
    from ModulatedPyWFS import ModulatedWavefrontSensor as WavefrontSensor
    from reconstruct import interaction_matrix
    import aberrations
    import stars
    
    WFE = 0.5/206265
    N_stars = 2**9
    
    
    # -------------------------------------------------------------------------
    # Setup the WFS class
    # -------------------------------------------------------------------------
    WFS = WavefrontSensor(pyramidOptic=PyramidArrayOptic, N_elements=36, py_kwargs={'N_pyramids':3},)
    imat = interaction_matrix(WFS.N_elements)
    
    print(WFS.pyramidOptic.pyramid_surface) # type: ignore
    py = WFS.pyramidOptic
    
    pyramid = py.pyramid.phase(800e-9).shaped
    plt.imshow(pyramid, cmap='hsv', origin='lower')
    plt.savefig('pyramid.png', dpi=300)
    
    plt.close('all')
    
    
    # plt.imshow(im, cmap='gray', origin='lower')
    # plt.show()
    
    
    
    # # -------------------------------------------------------------------------
    # # Pass a wavefront through the WFS
    # # -------------------------------------------------------------------------
    
    # # Inject an aberration in to the incoming wavefront
    # Z = aberrations.Zernike(WFS.input_pupil_grid, WFS.telescope_diameter)
    # phase = Z.from_name('tilt x', WFE=WFE*WFS.telescope_diameter/2,
    #                     wavelength=WFS.wavelength)
    # phase += Z.from_name('tilt y', WFE=WFE*WFS.telescope_diameter/2,
    #                     wavelength=WFS.wavelength)  
    # # phase = Z.from_name('spherical', WFE=WFE, wavelength=WFS.wavelength)
    # # phase = aberrations.make_noise_pl(2, WFS.Npx_pupil, WFS.Npx_pupil, -7, WFS.N_elements**2).ravel()
    # phase = hp.Field(phase, WFS.input_pupil_grid)

    # # Initialize the wavefront
    # wavefront = WFS.flat_wavefront()
    # # Apply the aberration to the wavefront
    # wavefront = aberrations.aberrate(wavefront, phase)
    # # Propagate the wavefront to the WFS
    # positions = stars.random_radius(WFS.focal_extent*206265/2, N_points=N_stars) / 206265
    # signal_raw = WFS.discrete_modulation(wavefront, positions)
    
    
    
    
    # # -------------------------------------------------------------------------
    # # Rotate the WFS signal to accommodate the pyramid orientation
    # # -------------------------------------------------------------------------
    # signal = scipy.ndimage.rotate(signal_raw, 45, reshape=False)
    # # crop the WFS_signal by a factor of 2**0.5 to match the original input
    # # grid size. 
    # crop_size = round(signal.shape[0] / 2**0.5)
    # center=False
    # # If the crop size is odd, need to add one pixel to the cropping, otherwise 
    # # the output will be off by one pixel. 
    # if crop_size % 2 == 1: center=True
    # signal = signal[
    #     signal.shape[0]//2 - crop_size//2 : signal.shape[0]//2 + crop_size//2+center,
    #     signal.shape[1]//2 - crop_size//2 : signal.shape[1]//2 + crop_size//2+center
    # ]
    
    
    # # -------------------------------------------------------------------------
    # # Measure WFS slopes
    # # -------------------------------------------------------------------------
    # sx, sy = WFS.measure_slopes(signal)
    
    
    
    
    
    # # Recover the phase
    # p = imat.slope2phase(sx, sy)
    # ap = WFS.circular_aperture(p.shape, p.shape[0]/2).astype(float) 
    # p = scipy.ndimage.rotate(p, -45, reshape=False) *ap
    # # -------------------------------------------------------------------------
    
    
    
    
    # # =========================================================================
    # # LIGHT PROGRESSION PLOT
    # # =========================================================================
    
    
    
    
    # # aberration, focal_plane, pyramid, WFS_signal0 = WFS.light_progression(wavefront)
    
    # focal_plane, WFS_signal0 = WFS.visualize_discrete_modulation(wavefront, positions)
    # # Make a plot of the light progression through the WFS
    # fig, ax = plt.subplots(nrows=1, ncols=4, 
    #                        tight_layout=True, 
    #                        figsize=(13,3))
    # plt.suptitle('WFS Light Progression')
    
    # # First, plot the incoming wavefront aberration
    # ax[0].set_title('Incoming Wavefront Phase')
    # im = ax[0].imshow(phase.shaped, cmap='bone', origin='lower')
    # plt.colorbar(im, fraction=0.046, pad=0.04, label='Phase [rad]')
    # # Overplot the aperture of the telescope
    # # alpha = ~WFS.aperture
    # # ax[0].imshow(~WFS.aperture, alpha=alpha.astype(float), cmap='Greys')
    # ax[0].axis('off')
    
    
    # ax[1].set_title('Focal Plane PSF')
    # img = np.log10(focal_plane / focal_plane.max())
    # img = hp.Field(img.ravel(), WFS.focal_grid)
    # plt.subplot(142)
    # im = hp.imshow_field(img, cmap='bone', vmin=-6, vmax=0, grid_units=1/206265, origin='lower') # type: ignore
    # # im = ax[1].imshow(img, cmap='bone', vmin=-6, vmax=0)
    # plt.colorbar(im, fraction=0.046, pad=0.04)
    
    
    # ax[2].set_title('Pyramid Phase Mask')
    # im = ax[2].imshow(pyramid, cmap='hsv', vmax=0, origin='lower')
    # plt.colorbar(im, fraction=0.046, pad=0.04, label='Phase [rad]')
    # ax[2].axis('off')
    
    # ax[3].set_title('WFS Signal')
    # im = ax[3].imshow(signal, cmap='bone', origin='lower')
    # plt.colorbar(im, fraction=0.046, pad=0.04)
    # ax[3].axis('off')
    
    # plt.savefig('test.png', dpi=300)
    # # plt.show()
    
    
    
    # verify_reconstruction()