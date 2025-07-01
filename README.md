# Pyramid Wavefront Sensor Simulation for the VWFWFS

This is a waveoptics simulation utilizing Fraunhofer Diffraction Theory. It is intended to simulate a Pyramid Wavefront Sensor for the VWFWFS project. 


## Wavefront Sensor
The file PyWFS.py contains a pyramid wavefront sensor class that can be used to measure the phase of an incoming wavefront via PyWFS. 
The point of this class is to sample an incoming wavefront and output the measured wavefront slopes in X and Y. 


## Wavefront Reconstruction
The file reconstruct.py contains two classes for wavefront reconstruction: 
interaction_matrix and zernike_decomposition. 

### interaction_matrix
The purpose of this class is to create an interaction matrix, which can then accept wavefront slopes as input to reproduce (reconstruct) the input wavefront phase (see method slope2phase). Wavefront reconstruction is performed via Southwell Geometry. 

#### Basic Usage
<code>

    from reconstruct import interaction_matrix

    # Number of sub-apertures across one dimension of the pupil
    N_subaps = 36
    # Create the interaction matrix (size: N_subaps x N_subaps)
    imat = interaction_matrix(N_subaps)

    # Reconstruct the wavefront phase, 
    # where x_slope and y_slope are the derivative of the wavefront phase.
    phase = imat.slopes2phase(x_slope, y_slope)
    
</code>


### zernike_decomposition
Decomposes a recovered wavefront phase into a zernike basis set. 

#### Basic Usage
<code>

    import hcipy as hp
    from reconstruct import zernike_decomposition

    # Number of zernike modes to decompose into
    N_modes = 10
    # Make the grid the wavefront is defined on
    grid = hp.make_pupil_grid(N_subaps, telescope_diameter)

    # Init the zernike decomposition class
    z_decomp = zernike_decomposition(N_modes, grid, telescope_diameter)

    # Recover the coeffecients for each zernike mode
    coeffs = z_decomp.decompose(wavefront_phase)
    # Re-project the coefficients to Zernike-decomposed wavefront phase
    decomposed_phase = z_decomp(coeffs)

</code>



### Pyramid Array Optics
The file pyramid_array_optic.py can be used to create a custom pyramid optic for the WFS. 
When using this class, it is important to note that the measured WFS signal MUST be rotated by +45 degrees (crop=True) before computing the WFS slopes. 
This is because the pyramid optic is rotated 45 degrees compared to the default HCIpy pyramid optic. 
Resulting, the recovered phase will then also need to be "de-rotated" by -45 degrees to account for the WFS signal being rotated (crop=False). 
Rotation should be done using the static wavefront sensor method "rotate"


## Examples
Examples on how to use the code can be found in /examples/

## Tests
Unit tests for the code can be found in /examples/tests/
