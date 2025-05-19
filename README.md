# Pyramid Wavefront Sensor Simulation for the VWFWFS

This is a waveoptics simulation utilizing Fraunhofer Diffraction Theory. It is intended to simulte a Pyramid Wavefront Sensor for the VWFWFS project. 


## Wavefront Sensor
The file PyWFS.py contains a pyramid wavefront sensor class that can be used to measure the phase of an incoming wavefront via PyWFS. 
The point of this class is to sample an incoming wavefront and output the measured wavefront slopes in X and Y. 


## Wavefront Reconstruction
The file reconstruct.py contains a class for wavefront reconstruction. 
The purpose of this class is to create an interaction matrix, then input WFS slopes to reproduce (reconstruct) the input wavefront phase.

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
