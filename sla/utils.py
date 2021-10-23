import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
import lmfit

def get_data_nbursts(spec_i, nvbins=51, two_comp=False, losvd_id=0):
    if two_comp:
        par_vel = spec_i['V'][losvd_id][0]
        par_sig = spec_i['SIG'][losvd_id][0]
    else:
        par_vel = spec_i['V']
        par_sig = spec_i['SIG']

    flux = spec_i['FLUX']
    fit = spec_i['FIT']
    fit_star_unconv = spec_i['FIT_UNCONV']
    fit_emis = np.sum(spec_i['FIT_COMP'][1:, :, :],
                      axis=0).flatten()  # np.zeros(len(fit))
    fit_star = fit - fit_emis
    xx = np.arange(flux.size)

    mask = ((np.isfinite(flux)) & (xx > (nvbins-1)/2) & (xx < xx.size-(nvbins-1)/2) &
            (fit_emis <= np.nanmax(fit_emis)/1e6))
    return par_vel, par_sig, flux, fit, fit_star_unconv, fit_emis, fit_star, xx, mask


def _gaussian(x, amp, v, sig):
    yy = (x - v) / sig
    return amp * np.exp(-0.5*yy**2)


def read_nbursts_results(file, two_comp=False, losvd_id=0, bin_sch=False):
    """
    Function to read NBursts results.
    """
    hdr = fits.getheader(file)
    sp = fits.getdata(file, 'SPECTRUM')
    bins = fits.getdata(file, 'BIN_SCHEMA')

    flux = sp['FLUX']
    err = sp['ERROR']
    fit_star_unconv = sp['FIT_UNCONV']
    #fit_emis = np.sum(sp['FIT_COMP'][1:, :, :],
    #                  axis=0).flatten()
    fit = sp['FIT']
    fit_comp = sp['FIT_COMP']
    wave = sp[0]['WAVE']
    goodpixels = sp['GOODPIXELS']
    velscale = np.log(wave[1]/wave[0])*299792.45
    
    if two_comp:
        vels = sp['V'][:, losvd_id]
        sigs = sp['SIG'][:, losvd_id]
    else:
        vels = sp['V']
        sigs = sp['SIG']

    if bin_sch:
        bin_schema = bins['BINNUM'][0]
        return wave, flux, err, fit_star_unconv, fit_comp, goodpixels, vels, sigs, velscale, bin_schema
    
    return wave, flux, err, fit_star_unconv, fit_comp, goodpixels, vels, sigs, velscale


def fit_LOSVD_2gaussian(file, lfile):
    fits.info(file)
    fits.info(lfile)

    bin2d, hdr = fits.getdata(file,'MAP_BINNUM', header=True)
    losvds = fits.getdata(lfile, 'LOSVDS')
    lambdas = fits.getdata(lfile, 'LAMBDAS')
    vbins = fits.getdata(lfile, 'VBINS')
    v0 = np.mean(vbins)
    print(f"Vsys: km/s {v0}")

    g2model = lmfit.Model(_gaussian, prefix="g1_") + \
              lmfit.Model(_gaussian, prefix="g2_")

    ilams = [3,5,10,20]
    for ilam in ilams:
        map_amp1 = np.zeros(bin2d.shape)*np.nan
        map_amp2 = np.zeros(bin2d.shape)*np.nan
        map_v1 = np.zeros(bin2d.shape)*np.nan
        map_v2 = np.zeros(bin2d.shape)*np.nan
        map_sig1 = np.zeros(bin2d.shape)*np.nan
        map_sig2 = np.zeros(bin2d.shape)*np.nan

        map_err_amp1 = np.zeros(bin2d.shape)*np.nan
        map_err_amp2 = np.zeros(bin2d.shape)*np.nan
        map_err_v1 = np.zeros(bin2d.shape)*np.nan
        map_err_v2 = np.zeros(bin2d.shape)*np.nan
        map_err_sig1 = np.zeros(bin2d.shape)*np.nan
        map_err_sig2 = np.zeros(bin2d.shape)*np.nan

        for i in range(losvds.shape[2]):
            idx_bin = (bin2d == i)
        # for i in range(5):
            params = g2model.make_params(g1_amp=0.1, g2_amp=0.2,
                                        g1_v=v0-300, g2_v=v0+300,
                                        g1_sig=100, g2_sig=100)
            g2model.set_param_hint('g1_amp', min=0)
            g2model.set_param_hint('g2_amp', min=0)
            g2model.set_param_hint('g1_sig', min=30, max=400.0)
            g2model.set_param_hint('g2_sig', min=30, max=400.0)
            g2model.set_param_hint('g1_v', min=v0-1000.0, max=v0+1000.0)
            g2model.set_param_hint('g2_v', min=v0-1000.0, max=v0+1000.0)

            result = g2model.fit(losvds[ilam, :, i], params, x=vbins, nan_policy='omit')

            print(result.fit_report())
            plt.cla()
            plt.step(vbins, losvds[ilam, :, i], where='mid')
            # plt.plot(vbins, result.init_fit, 'k--', label='initial fit')
            plt.plot(vbins, result.best_fit, 'r-', label='best fit')
            plt.pause(0.01)

            map_amp1[idx_bin] = result.params['g1_amp'].value
            map_amp2[idx_bin] = result.params['g2_amp'].value
            map_v1[idx_bin] = result.params['g1_v'].value
            map_v2[idx_bin] = result.params['g2_v'].value
            map_sig1[idx_bin] = result.params['g1_sig'].value
            map_sig2[idx_bin] = result.params['g2_sig'].value
        
            map_err_amp1[idx_bin] = result.params['g1_amp'].stderr
            map_err_amp2[idx_bin] = result.params['g2_amp'].stderr
            map_err_v1[idx_bin] = result.params['g1_v'].stderr
            map_err_v2[idx_bin] = result.params['g2_v'].stderr
            map_err_sig1[idx_bin] = result.params['g1_sig'].stderr
            map_err_sig2[idx_bin] = result.params['g2_sig'].stderr

        map_fl1 = np.sqrt(2*np.pi) * map_amp1 * map_sig1
        map_fl2 = np.sqrt(2*np.pi) * map_amp2 * map_sig2
        map_fr1 = map_fl1/ (map_fl1 + map_fl2)
        map_fr2 = map_fl2/ (map_fl1 + map_fl2)

        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(name="g1_amp", data=map_amp1, header=hdr),
            fits.ImageHDU(name="g2_amp", data=map_amp2, header=hdr),
            fits.ImageHDU(name="g1_v", data=map_v1-v0, header=hdr),
            fits.ImageHDU(name="g2_v", data=map_v2-v0, header=hdr),
            fits.ImageHDU(name="g1_sig", data=map_sig1, header=hdr),
            fits.ImageHDU(name="g2_sig", data=map_sig2, header=hdr),

            fits.ImageHDU(name="err_g1_amp", data=map_err_amp1, header=hdr),
            fits.ImageHDU(name="err_g2_amp", data=map_err_amp2, header=hdr),
            fits.ImageHDU(name="err_g1_v", data=map_err_v1, header=hdr),
            fits.ImageHDU(name="err_g2_v", data=map_err_v2, header=hdr),
            fits.ImageHDU(name="err_g1_sig", data=map_err_sig1, header=hdr),
            fits.ImageHDU(name="err_g2_sig", data=map_err_sig2, header=hdr),

            fits.ImageHDU(name="g1_fr1", data=map_fr1, header=hdr),
            fits.ImageHDU(name="g2_fr2", data=map_fr2, header=hdr),
            fits.ImageHDU(name="g2_delta_v", data=map_v2-map_v1, header=hdr),
        ])
        ofile = lfile.replace(".fits", f"_maps_lambda_{lambdas[ilam]:.3f}.fits")
        hdul.writeto(ofile, overwrite=True)
        print(f"Write output in file: {ofile}")