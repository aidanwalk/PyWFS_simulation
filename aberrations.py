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

