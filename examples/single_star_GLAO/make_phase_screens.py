import astropy.io.fits as fits
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
import aberrations


def save(data, fname):
    """
    Save the data to a FITS file.
    
    Parameters
    ----------
    data : np.ndarray
        The data to be saved.
    fname : str
        The name of the file to save the data to.
    """
    fits.writeto(fname, data, overwrite=True)
    print(f"Saved {fname} with shape {data.shape} and dtype {data.dtype}")
    return


def make_common_phase(RMS_WFE=10, order=-8):
    phase = aberrations.make_noise_pl(RMS_WFE,
                                      imagepix=Npx_foc,
                                      pupilpix=Npx_pupil, 
                                      pl=order,
                                      Nact=Nact)
    
    save(phase, sdir+'common_phase.fits')
    return


   
def make_random_phase(N_random, RMS_WFE=0.75, order=-5):
    for i in range(N_random):
        phase = aberrations.make_noise_pl(RMS_WFE,
                                          imagepix=Npx_foc,
                                          pupilpix=Npx_pupil, 
                                          pl=order,
                                          Nact=Nact)
        fname = sdir + f'random_phase_{i:03d}.fits'
        save(phase, fname)
        print(f"Saved {fname} with shape {phase.shape} and dtype {phase.dtype}")
        
    return



if __name__ == "__main__":
    sdir = './phase_screens/'
    N_random = 2**8
    Npx_foc = 2**8
    Npx_pupil = 2**8
    Nact = 36**2
    
    
    # make_common_phase(RMS_WFE=5, order=-8)
    # make_random_phase(N_random, RMS_WFE=1.5, order=-5)
    
    # Check the peak to valley values are correct
    phase = aberrations.make_noise_pl(1.5,
                                          imagepix=Npx_foc,
                                          pupilpix=Npx_pupil, 
                                          pl=-5,
                                          Nact=Nact)
    
    print(phase.max()-phase.min())