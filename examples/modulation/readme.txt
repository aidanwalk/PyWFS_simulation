Goal of this code: 

1. Visualize the WFS focal plane and output pupil plane 
2. Analyze the WFS gain (output slope as a function of input slope)
3. Verify wavefront reconstruction 

For several different cases:
1. Modulated pyramid wavefront sensor
   a. The star is modulated at fixed radius and uniform azimuth spacing	
   b. The star is modulated at fixed radius with random azimuth spacing
   c. The star is modulated at both random radii and azimuth spacing
   
   
The script template.py is used for generating a response curve where the input
aberration is an x-tilt. The output slope is measured by the mean of WFS
x-slope image. The input slope is the input x-tilt aberration in [arcsec]

The script template_pl.py is used for generating a response curve where the 
inbput is a random power law phase screen. The output slope is measured by the 
RMS variation in the recovered wavefront phase. The input slope is the RMS
deviation of the input phase in [radians]
