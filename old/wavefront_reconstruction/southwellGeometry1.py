# -*- coding: utf-8 -*-
"""
SouthwellGeometry1.py

This is an updated version of SouthwellGeometry.py. The difference is this new version
follows the original Southwell Geometry instroduced in the Southwell paper, the old version
uses a scheme modified by Mark.

This program contains the function that builds the interaction matrix A with
the Southwell Geometry described in the Southwell paper. Then, an SVD is performed
to find the inverse/pseudoInverse M of A, which is useful for recovering phase 
values from measured slopes and estimating wavefronts. 
Inputs: 
        ns (number of subapertures across the pupil instead of total number of subapertures, 
            which is equivalent to number of phases, lead to nRow and nCol)
        w (weights of phases, plays the function of g_jk in the Southwell paper)
        lSub (length of the subaperture)
Output: Matrix M

It also contains a test that generates an array of dummy phase values phi, uses the 
relationship of s=A*phi to calculate the supposed slopes, then uses phi=M*s to 
revocer phi. We will see how good of an recovery we get.

Created on Thu Sep 24 10:28:43 2020

@author: Suzanne Zhang
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from scipy import linalg

# The weights w and lSub has a default value = 1 if nothing else gets passed into it
# ns == number of subaperture cross the pupil (width in units of subaperture)
def southwell1(ns, lSub=1.0):
    
    nSub = ns*ns  # total number of subapertures nSub
                  # for the Southwell Geometry, there are 2*nSub number of slopes (x+y)
    nPhs = ns*ns   # this condition is true for Southwell Geometry
    nSlp = 2*nPhs
    
    # Initializing interation matrices Ax and Ay
    # represented in 3D first for the ease of operation
    # will later be truncated and reformated back to desired size and dimension
    Ax = np.zeros((nPhs,ns,ns))
    Ay = np.zeros((nPhs,ns,ns))
    Dx = Ax.copy()
    Dy = Ay.copy()
    
    # Loop through all the elements and set their values
    # Only the real values are looped through, not the paddings
    for j in range(ns-1):
        for i in range(ns-1):
            phI = j*ns+i # obtain the index of the phase getting updated, which needs to start at 0
            
            # Update Ax and Dx. 
            ax_ij = -1 # obtain the value of Ax[i,j]
            ax_ip1j = 1 # obtain the value of Ax[i+1,j]
            # write the values into the interaction matrix Ax
            Ax[phI,i,j] = ax_ij
            Ax[phI,i+1,j] = ax_ip1j
            
            dx_ij = 1
            dx_ip1j = 1
            Dx[phI, i, j] = dx_ij
            Dx[phI, i+1, j] = dx_ip1j
            
            # repeat the same process for Ax and Ds
            ay_ij = -1 # obtain the value of Ax[i,j]
            ay_ijp1 = 1 # obtain the value of Ax[i+1,j]
            # write the values into the interaction matrix Ax
            Ay[phI,i,j] = ay_ij
            Ay[phI,i,j+1] = ay_ijp1
            
            dy_ij = 1
            dy_ijp1 = 1
            Dy[phI, i, j] = dy_ij
            Dy[phI, i, j+1] = dy_ijp1
    
    # use scalar multiplication to correct the scale of the matrices
    Ax = Ax*2/lSub
    Ay = Ay*2/lSub
    
    # Reshape Ax and Ay to obtain the desired interaction matrix dimension
    Ax = np.reshape(Ax, (nPhs, nPhs))
    Ay = np.reshape(Ay, (nPhs, nPhs))
    Dx = np.reshape(Dx, (nPhs, nPhs))
    Dy = np.reshape(Dy, (nPhs, nPhs))
    
    AxTot = linalg.pinv(Dx).dot(Ax)
    AyTot = linalg.pinv(Dy).dot(Ay)
    
    # combine the xy matrices and keep them as 3d arrays
    A = np.vstack((AxTot,AyTot))
    A = np.reshape(A,(nSlp,nPhs))
    
    return A

# a helper function to find the pseudo inverse of the interaction matrix
def invA1(intM):
    # calculate the pseudo inverse M of intM using SVD 
    M = linalg.pinv(intM)
    return M

# This is a simple wave front reconstruction testing routine for the Southwell geometry 
# The input parameter of the function is the interaction matrix intM (takes the output of the southwell() function)
def testSW1(intM, phases, slopes):
    # Obtain the geometry of phase values through the interaction matrix (nP = number of phases across the pupil)
    nP = np.sqrt(intM[0].size)
    nP = nP.astype(int)
    
    # create the figure object to prepare for later plots
    plt.figure('SW') 
    
    # normalize the colors
    norm = colors.Normalize(vmin=-15,vmax=15)
    
    # use the random function to generate a phi vector between [0,1) the size 
    # of the col number of the provided interaction matrix
    phi = phases.flatten()# np.random.random_sample(intM[0].size)
    # reshape phi to plot the wavefront phases
    phiM = np.reshape(phi,(nP,nP))
    # create the position for the first plot
    ax1 = plt.subplot(2,2,1)
    # Plot the original phase values with color bar
    ax1.set_title("Original Phases (phi)")
    ax1.axis('off')
    plot1 = ax1.imshow(phiM, norm=norm)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size = "5%", pad = 0.05)
    plt.colorbar(plot1,cax=cax1)
    
    # Calculate the corresponding slopes of the phase values would have
    s = intM.dot(phi)
    
    # obtain the pseudo inverse M of intM using the helper function invA(A)
    M = invA1(intM)
    
    # calculate the reconstructed phase values with the calculated pseudo inverse M
    p = M.dot(s)
    # normalize the reconstructed phase calues according to the original phase values
    pN = p + np.average(phi)
    # reshape the matrix for graphing
    pM = np.reshape(pN,(nP,nP))
    # create the position for the second plot
    ax2 = plt.subplot(2,2,2)
    # Plot the reconstructed phase values 
    ax2.set_title("Reconstructed Phases (+ Avg(phi))")
    ax2.axis('off')
    plot2 = ax2.matshow(pM, norm=norm)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size = "5%", pad = 0.05)
    plt.colorbar(plot2,cax=cax2) # note, the color normalization is set according to the original phase values
    
    # calculate the difference between the original phase values and the reconstructed phase values
    diff = phi - p
    # reshape the difference matrix
    diffM = np.reshape(diff,(nP,nP))
    # create the position for the third plot
    ax3 = plt.subplot(2,2,3)
    # plot the difference of the orginal vs calculated phase values
    ax3.set_title("Difference")
    ax3.axis('off')
    plot3 = ax3.matshow(diffM, norm=norm)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size = "5%", pad = 0.05)
    plt.colorbar(plot3, cax=cax3)
    
    # Check how good the inverse is
    # see if the interaction matrix dotted with itself will give the identity matrix
    # or not. Plot the "idensity matrix"
    i = intM.dot(M)
    ax4 = plt.subplot(2,2,4)
    ax4.set_title("[A]*inv[A]")
    ax4.axis('off')
    plot4 = ax4.matshow(i,vmin=0,vmax=1)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size = "5%", pad = 0.05)
    plt.colorbar(plot4, cax=cax4)
    
    # show all the plots
    plt.show()
    return