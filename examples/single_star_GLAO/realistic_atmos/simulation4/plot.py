import matplotlib.pyplot as plt
plt.close('all')
from astropy.table import Table
from astropy.io import fits
import numpy as np
from matplotlib import animation

import sys
# sys.path.append('C:/Users/perfo/Desktop/School/gradschool/699_1/PyWFS_simulation')
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from examples.single_star_GLAO.realistic_atmos.simulation4.simulation4 import peak_location, dt, py_size, N_pyramids



def plot_error(tab, fname='simulation1.png'):
    plt.figure(figsize=(6,3), tight_layout=True)
    plt.title('Simulation 4')
    plt.plot(tab['num_screens'], tab['error'], c='k', linewidth=1)
    
    # plt.ylim()
    plt.xlabel('Phase screens averaged')
    plt.ylabel('std(recovered - true) / std(true)')
    
    plt.savefig('simulation4.png', dpi=300)
    return



def plot_recovery():
    recovered_phase = fits.getdata('final_recovered_phase.fits')
    common_phase = fits.getdata('common_phase_zoomed.fits')
    
    plt.figure(figsize=(8,3), tight_layout=True)
    plt.suptitle('Simulation 4')
    
    plt.subplot(131)
    plt.title('Common Phase')
    im = plt.imshow(common_phase, origin='lower', cmap='bone')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(132)
    plt.title('Final Recovered Phase')
    im = plt.imshow(recovered_phase, origin='lower', cmap='bone')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(133)
    plt.title('Difference')
    im = plt.imshow(recovered_phase - common_phase, origin='lower', cmap='bone')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.savefig('recovery_comparison.png', dpi=300)
    return



def plot_ee50(tab, plate_scale=0.01):
    plt.figure(figsize=(6,3), tight_layout=True)
    plt.title('Encircled Energy')
    plt.plot(tab['num_screens']*1e3, tab['uncorrected_ee50']*plate_scale, c='k', linestyle='dotted', linewidth=1, label='Uncorrected')
    plt.plot(tab['num_screens']*1e3, tab['corrected_ee50']*plate_scale, c='k', linewidth=1, label='Corrected')
    
    plt.ylim(0.2, 0.4)
    plt.xlabel('time [ms]')
    plt.ylabel('EE50 [arcsec]')
    plt.legend()
    
    plt.savefig('strehl_ratio.png', dpi=300)
    return





    
def make_ee_circle(image, r):
    xc, yc = peak_location(image)
    x, y = np.meshgrid(*(np.arange(s)for s in image.shape))
    circle = np.zeros(image.shape, dtype=bool)
    rs = ((x-xc)**2 + (y-yc)**2)**0.5
    width = 3
    circle[ (rs.astype(int) >= int(r-width/2)) & (rs.astype(int) <= int(r+width/2)) ] = True
    return circle
    
    
    
def plot_frames(tab):
    animation_frames = np.load('animation_frames.npy')
    recovered_ground_layer = np.load('recovered_ground_layer.npy')
    average_ground_layer = np.load('average_ground_layer.npy')
    integrated_phase = np.load('integrated_layer.npy')
    integrated_phase_off_axis = np.load('integrated_layer_off_ax.npy')
    plate_scale = py_size*N_pyramids * 206265 / animation_frames.shape[-1]

    corrected = animation_frames[:,2]
    uncorrected = animation_frames[:,1]
    
    plt.clf()
    fig, axs = plt.subplots(2, 3, figsize=(13, 10))
    plt.suptitle(f'Simulation 3: t = {0.:.4f} seconds', fontsize=16)
    
    axs[0][0].set_title('Ground Layer')
    axs[0][1].set_title('Recovered Ground Layer')
    axs[0][2].set_title('Integrated Atmos')
    axs[1][0].set_title('Uncorrected Focus')
    axs[1][1].set_title('Corrected Focus')
    axs[1][2].set_title('Integrated Off-axis Atmos')
    
    
    uc_circle = make_ee_circle(uncorrected[0], tab['uncorrected_ee50'][0])
    cc_circle = make_ee_circle(corrected[0], tab['corrected_ee50'][0])
    circles = [uc_circle, cc_circle]
    
    phase_frames = np.array([average_ground_layer, recovered_ground_layer, 
                           integrated_phase])

    # Initialize the plots
    pims = []
    anns = []
    maxs = []

    for i, ax in enumerate(axs[0]):
        # Get the image at timestep zero
        im = phase_frames[i][0]
        vmax = phase_frames[i].max()
        vmin = phase_frames[i].min()
        pim = ax.imshow(im, origin='lower', cmap='bone')
        plt.colorbar(pim, ax=ax, fraction=0.046, pad=0.04)
        pims.append(pim)
        ax.axis('off')
        pim.set_clim(vmin, vmax)

    for i, ax in enumerate(axs[1]):
        ax.axis('off')
        if i == 2:
            im = integrated_phase_off_axis[0]
            vmax = integrated_phase_off_axis.max()
            vmin = integrated_phase_off_axis.min()
            pim = ax.imshow(im, origin='lower', cmap='bone')
            plt.colorbar(pim, ax=ax, fraction=0.046, pad=0.04)
            pim.set_clim(vmin, vmax)
            pims.append(pim)
            continue

        # Get the image at timestep zero
        im = animation_frames[0][i+1]
        vmax = animation_frames[0][i+1].max()
        vmin = animation_frames[0][i+1].min()
        pim = ax.imshow(im+circles[i]*vmax*0.75, origin='lower', cmap='bone')
        plt.colorbar(pim, ax=ax, fraction=0.046, pad=0.04)
        
        ann = ax.annotate(f'EE50 = ____ arcsec', 
                          xy=(0.05, 0.05), xycoords='axes fraction', 
                          color='white')
        
        pim.set_clim(vmin, vmax)
        pims.append(pim)
        anns.append(ann)
        maxs.append(vmax)
    
    plt.tight_layout()
    
    
    def animate(i):
        # Get the current EE50 values from the table
        uc_ee50 = tab['uncorrected_ee50'][i] * plate_scale
        cc_ee50 = tab['corrected_ee50'][i] * plate_scale
        
        # Update the annotation with the current EE50 values
        anns[0].set_text(f'EE50 = {uc_ee50:.2f} arcsec')
        anns[1].set_text(f'EE50 = {cc_ee50:.2f} arcsec')
        
        # Make a circle for the current EE50 values
        uc_circle = make_ee_circle(animation_frames[i][1], uc_ee50/plate_scale)
        cc_circle = make_ee_circle(animation_frames[i][2], cc_ee50/plate_scale)
        circles = [uc_circle, cc_circle]
        
        for j, ax in enumerate(axs[0]):
            im = phase_frames[j][i]
            pims[j].set_array(im)

        for j, ax in enumerate(axs[1]):
            if j == 2:
                im = integrated_phase_off_axis[i]
                pims[3+j].set_array(im)
                continue

            im = animation_frames[i][j+1]
            pims[3+j].set_array(im+circles[j]*vmax*0.75)
            # pims[3+j].set_clim(0, vmax)
        
        plt.suptitle(f'Simulation 4: t = {i*dt:.4f} seconds', fontsize=16)
        return pims
    
    
    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=120,
        blit=True,
        frames=len(animation_frames),
        repeat_delay=100,
    )
    
    ani.save('animation.html', writer='html', fps=10)
    # ani.save('animation.gif', writer='pillow', fps=10)
    return axs





if __name__ == "__main__":
    f = 'simulation4.txt'
    format = 'ascii.fixed_width'
    
    tab = Table.read(f, format=format)
    
    
    plot_error(tab, fname='simulation4.png')
    plot_recovery()
    plot_ee50(tab, plate_scale=py_size*N_pyramids*206265/500)
    plot_frames(tab)
    

