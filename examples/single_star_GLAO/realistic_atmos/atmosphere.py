import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.close('all')
import hcipy as hp
from matplotlib import animation
from scipy.ndimage import zoom
# import simulation modules
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
# sys.path.append('C:/Users/perfo/Desktop/School/gradschool/699_1/PyWFS_simulation')
from ModulatedPyWFS import ModulatedWavefrontSensor
import aberrations

def plot_7_layers(images):
    # plt.figure('layers', figsize=(10, 10), tight_layout=True)
    fig, axs = plt.subplots(3, 3, figsize=(10, 9), tight_layout=True)
    # plt.clf()
    
    # Find the min and max values of every layer over any time step
    maxs, mins = [], []
    for layer_idx in range(images.shape[1]):
        maxs.append(np.max(images[:, layer_idx]))
        mins.append(np.min(images[:, layer_idx]))
        
    
    plt.suptitle('Keck Atmosphere Layers — '+f'time: {0} seconds', fontsize=16)
    pims = []
    for i, layer in enumerate(images[0]):
        ax = axs[i // 3, i % 3]
        # phase = layer.phase_for(WFS.wavelength).shaped
        im = ax.imshow(layer, cmap='bone', origin='lower', vmin=mins[i], vmax=maxs[i])
        ax.set_title(f'Layer {i+1}: Height = {layers[i].height} m')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        pims.append(im)
    
    
    for j in range(i + 1, 9):
        axs[j // 3, j % 3].axis('off')
        
        
        
    def animate(i):
        for j, layer in enumerate(images[i]):
            # ax = axs[j // 3, j % 3]
            pims[j].set_array(layer)
            plt.suptitle('Keck Atmosphere Layers — '+f'time: {i*dt:.2f} seconds', fontsize=16)
            
        return pims
        
    ani = animation.FuncAnimation(
                                fig,
                                animate,
                                interval=120,
                                blit=True,
                                frames=len(images),
                                repeat_delay=100,
                                )
    
    ani.save('animation.gif', writer='pillow', fps=5)
    return axs




def plot_2_layers(images, fname='atmosphere.gif'):
    # plt.figure('layers', figsize=(10, 10), tight_layout=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    axs[0].set_title('Ground Layer')
    axs[1].set_title('Free Atmosphere')
    # plt.clf()
    
    # Find the min and max values of every layer over any time step
    maxs, mins = [], []
    for layer_idx in range(images.shape[1]):
        maxs.append(np.max(images[:, layer_idx]))
        mins.append(np.min(images[:, layer_idx]))
        
    
    plt.suptitle('Maunakea Atmosphere — '+f'time: {0} seconds', fontsize=16)
    pims = []
    for i, layer in enumerate(images[0]):
        ax = axs[i]
        # phase = layer.phase_for(WFS.wavelength).shaped
        im = ax.imshow(layer, cmap='bone', origin='lower', vmin=mins[i], vmax=maxs[i])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        pims.append(im)
        
        
    def animate(i):
        for j, layer in enumerate(images[i]):
            # ax = axs[j // 3, j % 3]
            pims[j].set_array(layer)
            plt.suptitle('Maunakea Atmosphere — '+f'time: {i*dt:.2f} seconds', fontsize=16)
            
        return pims
        
    ani = animation.FuncAnimation(
                                fig,
                                animate,
                                interval=120,
                                blit=True,
                                frames=len(images),
                                repeat_delay=100,
                                )
    
    ani.save(fname, writer='pillow', fps=5)
    return axs




    
# %%

if __name__ == "__main__":
    # HCIPY ATMOSPHERE
    # Init the WFS
    WFS = ModulatedWavefrontSensor(focal_extent=2/206265,
                                   telescope_diameter=10.0,)
    # Create a wavefront with no aberrations
    wavefront = WFS.flat_wavefront()
    
    
    print('Creating Keck atmosphere layers')
    layers = aberrations.make_two_layer_chun(WFS.input_pupil_grid, seeds=[2, 3])
    atmosphere = hp.atmosphere.MultiLayerAtmosphere(layers)
    
    # Propagate the wavefront through the atmosphere
    wavefront = atmosphere.forward(wavefront)
    
    # # Plot the focal plane
    # focal_image = WFS.light_progression(wavefront)[1]
    # plt.figure(figsize=(6, 4), tight_layout=True)
    # plt.title('Focal Plane Image with Keck Atmosphere')
    # im = hp.imshow_field(hp.Field(focal_image.ravel(), WFS.focal_grid), 
    #                      cmap='bone', origin='lower', 
    #                      grid_units=1/206265)
    # plt.colorbar(im, fraction=0.046, pad=0.04)
    # plt.xlabel('Focal Plane X (arcsec)')
    # plt.ylabel('Focal Plane Y (arcsec)')
    
    # plt.savefig('focal_plane_keck_atmosphere.png', dpi=300)
    # plt.show()
    
    
    t = 0.5 # total time in seconds for the simulation
    dt = 2.2 / 29 / 10  # seconds per step
    N_steps = int(t / dt)
    
    ims = np.zeros((N_steps, len(layers), *WFS.input_pupil_grid.shape))
    print('Generating images for each layer')
    for step in tqdm(range(N_steps)):
        for i, layer in enumerate(layers):
            layer.evolve_until(dt * step)
            phase = layer.phase_for(WFS.wavelength).shaped
            ims[step, i] = phase
    
    # # Save the data to a FITS file
    # fits.writeto('keck_atmosphere_layers.fits', ims, overwrite=True)
    # print(f"Saved keck_atmosphere_layers.fits with shape {ims.shape} and dtype {ims.dtype}")
    
    # %%
    print('Making animation')
    ims = np.array(ims)
    # plot_7_layers(ims)
    plot_2_layers(ims, fname='maunakea_atmosphere.gif')
        
    
    
    
    