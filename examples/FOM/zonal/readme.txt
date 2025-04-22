The purpose of this script is to show the negative effect of the concave 
pyramid edges in a pyramid array.

The script works by simultaneously simulating two wavefront sensors: 
1. A wavefront sensor with a 1x1 pyramid array (A single pyramid) 
2. A wavefront sensor with a 2x2 pyramid array (4 pyramids total)

We want to only study the effect of the concave pyramid edges, so we will 
initialize the simulation such that the 1x1 pyramid array is effectively a 
2x2 pyramid array, only without any negative edges. To do this:
1. The relative sizes of the pyramids should be equal. Therefore, the focal
   extent of the 2x2 pyramid array will be 2 times larger than the single 
   pyramid. 
2. The number of stars per pyramid will be four times larger for the single 
   pyramid. This can be thought of as a 2x2 pyramid array, but instead of four
   pyramids, there is only one pyramid with four times as many stars. 
   
Notes:
- The phase reconstruction represents the best possible case. Since the true
  input phase is known, the WFS gain is determined by a normalization step. 
  WFS gain = max(true phase) / max(recovered phase). We do this to increase
  the reconstruction accuracy. Otherwise, we could find the WFS gain by
  averaging the WFS response over a series of random wavefront realizations. 
  However, when we compute the gain this way, there is some noise associated
  with the measurement, since the measured gain depends sensitively on where
  the stars randomly fall on the pyramid. 


