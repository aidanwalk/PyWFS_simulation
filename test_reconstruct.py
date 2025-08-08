

from reconstruct import interaction_matrix
import numpy as np

N = 4

imat = interaction_matrix(N)

# If we initialize x-slopes to have a gradient of 1 across the aperture
# We should expect D@sx = 1 everywhere, since D is the matrix that performs
# adjacent slope averaging (where the average slope between any to neighbors is
# 1. 

sx = np.ones((N, N)).flatten()
sy = np.zeros((N, N)).flatten()

slopes = np.concatenate([sx, sy])

print(imat.D.shape)
print(slopes.shape)

print(imat.D @ slopes)


print(np.reshape(imat @ slopes, (N,N)))