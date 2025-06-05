import matplotlib.pyplot as plt
plt.close('all')
from astropy.table import Table
from astropy.io import fits


def plot_error(tab, fname='simulation2.png'):
    plt.figure(figsize=(6,3), tight_layout=True)
    plt.title('Simulation 2')
    plt.plot(tab['num_screens'], tab['error'], c='k', linewidth=1)
    
    # plt.ylim()
    
    plt.xlabel('Phase screens averaged')
    plt.ylabel('std(recovered - true) / std(true)')
    
    plt.savefig(fname, dpi=300)
    return


def plot_recovery():
    recovered_phase = fits.getdata('final_recovered_phase.fits')
    common_phase = fits.getdata('common_phase_zoomed.fits')
    
    plt.figure(figsize=(8,3), tight_layout=True)
    plt.suptitle('Simulation 2')
    
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



if __name__ == "__main__":
    f = 'simulation2.txt'
    format = 'ascii.fixed_width'
    
    tab = Table.read(f, format=format)
    
    plot_error(tab, fname='simulation2.png')
    plot_recovery()
    
    