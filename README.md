# Pyramid Wavefront Sensor Simulation for the VWFWFS

This is a waveoptics simulation utilizing Fraunhofer Diffraction Theory. It is intended to simulte a Pyramid Wavefront Sensor for the VWFWFS project. 


## Wavefront Sensor
The file PyWFS.py contains a pyramid wavefront sensor class that can be used to measure the phase of an incoming wavefront via PyWFS. 
The point of this class is to sample an incoming wavefront and output the measured wavefront slopes in X and Y. 


## Wavefront Reconstruction
The file reconstruct.py contains a class for wavefront reconstruction. 
The purpose of this class is to create an interaction matrix, then input WFS slopes to reproduce (reconstruct) the input wavefront phase.


## Examples
Examples on how to use the code can be found in /examples/

## Tests
Unit tests for the code can be found in /examples/tests/
