import matplotlib.pyplot as plt
plt.close('all')
from astropy.table import Table
from astropy.io import fits
import numpy as np
from matplotlib import animation

from simulation1 import peak_location, dt, py_size



def plot_error(tab, fname='simulation1.png'):
    plt.figure(figsize=(6,3), tight_layout=True)
    plt.title('Simulation 1')
    plt.plot(tab['num_screens'], tab['error'], c='k', linewidth=1)
    
    # plt.ylim()
    plt.xlabel('Phase screens averaged')
    plt.ylabel('std(recovered - true) / std(true)')
    
    plt.savefig('simulation1.png', dpi=300)
    return


def plot_recovery():
    recovered_phase = fits.getdata('final_recovered_phase.fits')
    common_phase = fits.getdata('common_phase_zoomed.fits')
    
    plt.figure(figsize=(8,3), tight_layout=True)
    plt.suptitle('Simulation 1')
    
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



def plot_strehl(tab):
    plt.figure(figsize=(6,3), tight_layout=True)
    plt.title('Encircled Energy')
    plt.plot(tab['num_screens'], tab['uncorrected_ee50'], c='k', linestyle='dotted', linewidth=1, label='Uncorrected')
    plt.plot(tab['num_screens'], tab['corrected_ee50'], c='k', linewidth=1, label='Corrected')
    
    plt.xlabel('time [seconds]')
    plt.ylabel('EE50')
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
    plate_scale = py_size * 206265 / animation_frames.shape[-1]
    
    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle(f'Simulation 1: t = {0.:.4f} seconds', fontsize=16)
    
    axs[0].set_title('Uncorrected Focus')
    axs[1].set_title('Corrected Focus')
    
    
    uc_circle = make_ee_circle(animation_frames[0][1], tab['uncorrected_ee50'][0])
    cc_circle = make_ee_circle(animation_frames[0][2], tab['corrected_ee50'][0])
    circles = [uc_circle, cc_circle]
    
    # Initialize the plots
    pims = []
    anns = []
    maxs = []
    for i, ax in enumerate(axs):
        ax.axis('off')
        im = animation_frames[0][i+1]
        vmax = animation_frames[0][i+1].max()
        vmin = animation_frames[0][i+1].min()
        pim = axs[i].imshow(im+circles[i]*vmax*0.75, origin='lower', cmap='bone')
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
        
        for j, ax in enumerate(axs):
            im = animation_frames[i][j+1]
            pims[j].set_array(im+circles[j]*vmax*0.75)
            pims[j].set_clim(0, vmax)
        
        plt.suptitle(f'Simulation 1: t = {i*dt:.4f} seconds', fontsize=16)
        return pims
    
    
    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=120,
        blit=True,
        frames=len(animation_frames),
        repeat_delay=100,
    )
    
    ani.save('animation.html', writer='html', fps=5)
    return axs





if __name__ == "__main__":
    f = 'simulation1.txt'
    format = 'ascii.fixed_width'
    
    tab = Table.read(f, format=format)
    
    
    plot_error(tab, fname='simulation1.png')
    plot_recovery()
    plot_strehl(tab)
    plot_frames(tab)
    
