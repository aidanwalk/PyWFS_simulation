"""
-------------------------------------------------------------------------------
File containing functions and classes for simulating optical aberrations.
-------------------------------------------------------------------------------


Created on Mon Mar 17 14:43 2025

@author: Aidan Walk
"""


import hcipy as hp
from hcipy.atmosphere import InfiniteAtmosphericLayer
import numpy as np


def aberrate(wavefront, *phase):
    """
    Applies a phase delay to a wavefront's electric field. 

    Parameters
    ----------
    wavefront : hcipy.Field
        The wavefront to be aberrated.
    phase : hcipy.Field
        The phase delay to be applied to the wavefront.
    """
    wavefront.electric_field *= np.exp(1j * np.sum(phase, axis=0))
    return wavefront



class Zernike:
    """
    A class for generating Zernike polynomials.
    ---------------------------------------------------------------------------

    Parameters
    ----------
    grid : hcipy.Grid
        The input pupil grid on which to evaluate the Zernike polynomials.
    D : float
        The diameter of the Zernike polynomials (same units as grid).
    """

    COMMON_MODES = {
        # name                          : (n, m),
        'piston'                        : (0, 0),
        'tilt y'                        : (1,-1),
        'tilt x'                        : (1, 1),
        'oblique astigmatism'           : (2,-2),
        'defocus'                       : (2, 0),
        'vertical astigmatism'          : (2, 2),
        'vertical trefoil'              : (3,-3),
        'vertical coma'                 : (3,-1),
        'horizontal coma'               : (3, 1),
        'oblique trefoil'               : (3, 3),
        'oblique quadrafoil'            : (4,-4),
        'oblique secondary astigmatism' : (4,-2),
        'spherical'                     : (4, 0),
        'vertical secondary astigmatism': (4, 2),
        'vertical quadrafoil'           : (4, 4),
    }


    def __init__(self, grid, D):
        self.grid = grid
        self.D = D


    def evaluate(self, n=None, m=None):
        """
        Evaluates the Zernike polynomial on the grid.

        Parameters
        ----------
        n : int
            The radial Zernike order.
        m : int
            The azimuthal Zernike order.

        Returns
        -------
        hcipy.Field
            The evaluated Zernike polynomial
        """
        return hp.mode_basis.zernike(n, m, D=self.D, grid=self.grid)
    
    
    def evaluate_named(self, name):
        """
        Evaluates a Zernike polynomial by name. A list of common
        Zernike modes is provided in the COMMON_MODES attribute.

        Parameters
        ----------
        name : str
            The name of the Zernike mode.

        Returns
        -------
        hcipy.Field
            The evaluated Zernike polynomial.
        """
        if name not in self.COMMON_MODES:
            raise ValueError(f"Unknown Zernike mode: {name}")
        
        n, m = self.COMMON_MODES[name]
        return self.evaluate(n, m)
    
    

    def phase_delay(self, n, m, WFE, wavelength):
        """
        Calculates the phase delay of a Zernike mode.
        For example, to generate a 1 arcsecond tilt aberration:
        
        Z = Zernike(input_pupil_grid, D)
        phase = from_name('tilt x', WFE= (1/206265 * D/2), wavelength=wvln)

        Parameters
        ----------
        n : int
            The radial Zernike order.
        m : int
            The azimuthal Zernike order.
        WFE : float
            The wavefront error in radians.
        wavelength : float
            The wavelength of light in meters.

        Returns
        -------
        hcipy.Field
            The phase delay of the Zernike mode.
        """
        k = 2*np.pi / wavelength
        return self.evaluate(n, m) * WFE/2 * k

    
    def from_name(self, name, WFE, wavelength):
        """
        Calculates the phase delay of a Zernike mode by name. A list of common
        Zernike modes is provided in the COMMON_MODES attribute.

        Parameters
        ----------
        name : str
            The name of the Zernike mode.
        WFE : float
            The wavefront error in radians.
        wavelength : float
            The wavelength of light in meters.

        Returns
        -------
        hcipy.Field
            The phase delay of the Zernike mode.
        """
        if name not in self.COMMON_MODES:
            raise ValueError(f"Unknown Zernike mode: {name}")
        
        n, m = self.COMMON_MODES[name]
        return self.phase_delay(n, m, WFE, wavelength)

    



def make_keck_layers(input_grid, seeds=None):
    '''Creates a multi-layer atmosphere for Keck Observatory.

    The atmospheric parameters are based off of [Keck AO note 303]_.

    .. [Keck AO note 303] https://www2.keck.hawaii.edu/optics/kpao/files/KAON/KAON303.pdf

    Parameters
    ----------
    input_grid : Grid
        The input grid of the atmospheric layers.

    Returns
    -------
    list
        A list of turbulence layers.
    '''
    heights = np.array([0, 2100, 4100, 6500, 9000, 12000, 14800])
    velocities = np.array([6.7, 13.9, 20.8, 29.0, 29.0, 29.0, 29.0])
    outer_scales = np.array([20, 20, 20, 20, 20, 20, 20])
    Cn_squared = np.array([0.369, 0.219, 0.127, 0.101, 0.046, 0.111, 0.027]) * 1e-12
    
    if seeds is not None: assert len(seeds) == len(heights), "Number of seeds must match number of layers."
    else: seeds = [None] * len(heights)
    
    layers = []
    for h, v, cn, L0, s in zip(heights, velocities, Cn_squared, outer_scales, seeds):
        layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h, 2, seed=s))

    return layers


def make_unmoving_ground_layers(input_grid, seeds=None):
    '''
    Parameters
    ----------
    input_grid : Grid
        The input grid of the atmospheric layers.

    Returns
    -------
    list
        A list of turbulence layers.
    '''
    heights = np.array([0, 2100, 4100, 6500, 9000, 12000, 14800])
    # velocities = np.array([6.7, 13.9, 20.8, 29.0, 29.0, 29.0, 29.0])
    velocities = np.array([0, 13.9, 20.8, 29.0, 29.0, 29.0, 29.0])
    outer_scales = np.array([20, 20, 20, 20, 20, 20, 20])
    Cn_squared = np.array([0.369, 0.219, 0.127, 0.101, 0.046, 0.111, 0.027]) * 1e-12
    
    if seeds is not None: assert len(seeds) == len(heights), "Number of seeds must match number of layers."
    else: seeds = [None] * len(heights)
    
    layers = []
    for h, v, cn, L0, s in zip(heights, velocities, Cn_squared, outer_scales, seeds):
        layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h, 2, seed=s))

    return layers


def make_two_layer_chun(input_grid, seeds=None):
    """
    Makes a multi-layer atmosphere consistent with Chun et al. (2008) [1]_.
    [1] Chun, M., et al. (2008). "Mauna Kea Ground Layer Characterization Campaign."
        doi:10.1111/j.1365-2966.2008.14346.x

    The ground layer strength is the integrated 75 percentile turbulence from 0 - 640 meters.
    The Free atmosphere (FA) is the 75 percentile FA from [1].

    """
    # Parameters from Table 6. 
    heights    = np.array([  0.,    15.,   30.,    45.,   120.,   200.,    280.,    360.,    440.,    520., 600.,  5000.])
    Cn_squared = np.array([0.14, 0.0856, 0.021, 0.0103, 0.0243, 0.0101, 0.00901, 0.00691, 0.00497, 0.00167, 0.00, 0.0937]) * 1e-12

    # Median ground windspeed at the summit of Maunakea is 7 m/s.
    velocities = np.array([7, 29])
    # Find the integrated Cn^2 for the ground layer and free atmosphere
    Cn_layer1 = np.sum(Cn_squared[heights <= 640])  
    Cn_layer2 = np.sum(Cn_squared[heights > 640])

    if seeds is not None: assert len(seeds) == len(velocities), "Number of seeds must match number of layers."
    else: seeds = [None] * len(heights)

    layers = []
    for h, v, cn, s in zip(heights, velocities, [Cn_layer1, Cn_layer2], seeds):
        layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0=20, velocity=v, height=h, seed=s))

    return layers


# =============================================================================
# CODE BELOW IS TAKEN FROM:
# @author: Miles Lucas
# =============================================================================

def pupil_to_image(im):
	return np.fft.fft2(im,norm='ortho')



def image_to_pupil(im):
	return np.fft.ifft2(im,norm='ortho')



def complex_amplitude(mag,phase):
	'''
	complex amplitude in terms of magnitude and phase
	'''
	return mag*np.cos(phase)+1j*mag*np.sin(phase)



def xy_plane(dim):
	'''
	define xy plane to use for future functions
	'''
	grid=np.mgrid[0:dim,0:dim]
	xy=np.sqrt((grid[0]-dim/2.+0.5)**2.+(grid[1]-dim/2.+0.5)**2.)
	return xy



def antialias(phin,imagepix,beam_ratio,Nact):
	'''
	anti-alias via a butterworth filter
	'''
	xy=xy_plane(imagepix)
	buttf = lambda rgrid,eps,r0,n: 1./np.sqrt(1+eps**2.*(xy/r0)**n) #butterworth filter
	phinput=phin-np.min(phin)
	phfilt=np.abs(pupil_to_image(np.fft.fftshift(image_to_pupil(phinput))*(buttf(xy,1,Nact/2*beam_ratio*0.99,100)))) #Nact actuators across the pupil
	phout=phfilt-np.mean(phfilt)
	return phout



p3i=lambda i: int(round(i))



def make_noise_pl(wavefronterror,imagepix,pupilpix,pl,Nact):
	'''
	make noise with a user input power law:

	(1) take white noise in image plane
	(2) IFFT to pupil plane
	(3) weight complex pupil plane (spatial frequency) by power law (power law is specified on the PSD, scaling is half that for amplitude)
	(4) FFT back to image plane and take the real part

	wavefronterror = rad rms WFE
	'''

	white_noise=np.random.random((imagepix,imagepix))*2.*np.pi #phase
	xy=xy_plane(imagepix)
	amplitude=(xy+1)**(pl/2.) #amplitude central value in xy grid is one, so max val is one in power law, everything else lower

	amplitude[p3i(imagepix/2),p3i(imagepix/2)]=0. #remove piston

	#remove alaising effects by cutting off power law just before the edge of the image
	amplitude[np.where(xy>imagepix/2.-1)]=0.

	amp=np.fft.fftshift(amplitude)
	image_wavefront=complex_amplitude(amp,white_noise)
	noise_wavefront=np.real(np.fft.fft2(image_wavefront))

	beam_ratio=p3i(imagepix/pupilpix)

	norm_factor=wavefronterror/np.std(antialias(noise_wavefront,imagepix,beam_ratio,Nact)[np.where(xy<pupilpix/2.)]) #normalization factor for phase error over the pupil of modes within the DM control region
	phase_out_ini=noise_wavefront*norm_factor

	phase_out=phase_out_ini

	return phase_out