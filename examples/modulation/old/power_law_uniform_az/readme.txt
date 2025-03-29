Creates a PyWFS response curve by computing the WFS response to a random 
power law phase screen. Modulation is uniform azimuth spacing at constant 
radius. 



Plot of the response curve: 
---------------------------
response_uniform_azimuth_pl.png 

is produced by:
For each modulation radius, simulate 50 random power law phase screens. 
Reconstruct the wavefront for each phase screen, compute the RMS deviation 
of the reconstruction, then average the 50 RMS deviation values. This is then 
plotted on the y-axis as "output rms wavefront error". 

Do this for varying powers of the phase screen ==> makes a curve of input vs 
output. 


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
