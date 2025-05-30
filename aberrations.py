"""
-------------------------------------------------------------------------------
File containing functions and classes for simulating optical aberrations.
-------------------------------------------------------------------------------


Created on Mon Mar 17 14:43 2025

@author: Aidan Walk
"""


import hcipy as hp
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


    def evaluate(self, n, m):
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
    

    def phase_delay(self, n, m, WFE, wavelength):
        """
        Calculates the phase delay of a Zernike mode.

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