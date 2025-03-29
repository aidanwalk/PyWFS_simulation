Creates a PyWFS response curve by computing the WFS response to an x-tilt 
aberration. Modulation is uniform azimuth spacing at constant radius. 



Plot of the response curve: 
---------------------------
response_uniform_azimuth.png 

is produced by:
For each modulation radius, compute the WFS response for a given input tilt. 
The WFS response is computed by the mean x-slope of the WFS.  


reconstruction_uniform_azimuth_pl.png
-------------------------------------
A plot demonstrating the wavefront reconstruction for this modulation type 
(constant radius, uniform azimuth spacing). This plot is just used to verify
the wavefront reconstruction actually works. 


light_progression_uniform_azimuth.png
-------------------------------------
A plot for visualizing the pyramid modulation. The focal plane subplot shows the
positions in the focal plane the star was projected onto for "modulation". 
The pupil plane plot shows the resulting WFS pupil images. 
