import matplotlib.pyplot as plt
import numpy as np



im1 = np.random.rand(500, 500)
im2 = im1
im3 = im1

circles = []
for im in [im1, im2, im3]:
    xc, yc = im.shape[0] // 2, im.shape[1] // 2
    r = 50
    x, y = np.meshgrid(*(np.arange(s)for s in im.shape))
    circle = np.zeros(im.shape, dtype=bool)
    rs = ((x-xc)**2 + (y-yc)**2)**0.5
    width = 5
    circle[ (rs.astype(int) >= int(r-width/2)) & (rs.astype(int) <= int(r+width/2)) ] = True
    circles.append(circle)

plt.subplot(131)
plt.imshow(im1, origin='lower', cmap='bone')
plt.imshow(circles[0], origin='lower', cmap='bone', alpha=circles[0].astype(float)) 
plt.show()
