import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import optimize
from scipy import linalg

from tqdm import tqdm


def get_losvd(velocity, sigma, nbins=51, numnorm=True):
    """
    Calculate a gaussian profile.
    """
    x = np.arange(nbins)
    x0 = (nbins-1) / 2
    yy = (x - x0 - velocity) / sigma
    factor = 1 / np.sqrt(2*np.pi) / sigma
    profile = factor * np.exp(-0.5*yy**2)

    if numnorm:
        profile /= np.trapz(profile, x)
    return profile


def get_matrix(template, nbins=51):
    """
    Prepare convolution matrix using unconvolved template.
    """
    matrix = np.zeros((template.size, nbins))
    x0 = int((nbins-1) / 2)
    for i in range(nbins):
        matrix[:, i] = np.roll(template, i-x0, axis=0)
    return matrix


def get_reg_matrix(nvbins, reg_type='L2'):
    """
    Select an appropriate regularization matrix.
    """
    if reg_type == 'L2':
        matrix_regularization = np.identity(nvbins)
    elif reg_type == 'smooth1':
        matrix_regularization = np.zeros((nvbins - 1, nvbins))
        for q in range(nvbins - 1):
            matrix_regularization[q, q:q+2] = np.array([-1, 1])
    elif reg_type == 'smooth2':
        matrix_regularization = np.zeros((nvbins - 2, nvbins))
        for q in range(nvbins - 2):
            matrix_regularization[q, q:q+3] = np.array([-1, 2, 1])

    return matrix_regularization


def solve(flux, matrix, losvd, mask, lam=0, reg_type='L2'):
    """
    Solve linear regularized problem.
    """
    nvbins = losvd.size
    spec_convolved = np.dot(matrix, losvd)

    matrix_regularization = get_reg_matrix(nvbins, reg_type=reg_type)
    nreg_bins = matrix_regularization.shape[0]

    matrix_extended = np.vstack((matrix, lam * matrix_regularization))

    # add spectrum convolved with gaussian referenced LOSVD + zeroed elements in
    # place of regularization matrix
    referenced_vector = np.hstack(
        (spec_convolved, np.zeros(nreg_bins)))

    matrix_extended = np.hstack((
        matrix_extended,
        referenced_vector.reshape(referenced_vector.size, 1)
    ))

    bounds = np.vstack((-losvd, 1.0 + 0*losvd)).T
    bounds = np.vstack((bounds, np.array([1-1e-5, 1])))

    # add elements according regularized part of the problem
    flux_upd = np.hstack((flux, np.zeros(nreg_bins)))
    mask_upd = np.hstack((mask, np.ones(nreg_bins, dtype=bool)))

    res = optimize.lsq_linear(
        matrix_extended[mask_upd, :], flux_upd[mask_upd], bounds=bounds.T, method='bvls', max_iter=1000)
    delta_losvd = res.x[:-1]
    losvd_full = losvd+delta_losvd
    bestfit = np.dot(matrix_extended[:flux.size, :], res.x)

    metrics = dict(chi2=np.sum((flux[mask]-bestfit[mask])**2),
                   reg2=np.sum(np.dot(matrix_regularization, delta_losvd)**2))

    return delta_losvd, bestfit, metrics


def get_data_clear(spec_i):
    print("In development, do not touch this!")
    return [np.nan]*9


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


def recover_losvd_2d(specs, templates, goodpixels, vels, sigs, velscale, nvbins=51,
                     lamdas=np.array([0, 0.5, 1.0]), ofile=None,
                     path='./', reg_type='L2', losvd_id=0, vshift=None,
                     debug=False, nbursts=False, two_comp=False, obj='', wave=None):
    """
    Recover stellar LOSVD in each spectral bin for different smoothing
    parameter lambda (lambdas array).
    """
    if vshift is None:
        vshift = vels[0]
    # import ipdb; ipdb.set_trace()
    nspec = specs.shape[0]  # number of spectra
    npix = specs.shape[1]  # number of pixels in the spectrum
    xx = np.arange(npix)

    # prepare output arrays
    out_losvd2d = np.zeros((lamdas.size, nvbins, nspec))
    out_losvd2d_gau = np.zeros((nvbins, nspec))
    out_chi2 = np.zeros((nspec, lamdas.size))
    out_reg2 = np.zeros((nspec, lamdas.size))
    out_fluxes = np.zeros((nspec, npix))
    out_masks = np.zeros((nspec, npix))
    out_bestfits = np.zeros((nspec, npix, lamdas.size))

    # main loop on separate spectra
    for ibin in tqdm(range(nspec)):
        flux = specs[ibin]
        temp = templates[ibin]
        gpix = goodpixels[ibin]

        factor = np.nanmedian(flux)
        for ff in (flux, temp):
            ff /= factor
            
        losvd = get_losvd((vels[ibin]-vshift)/velscale, sigs[ibin]/velscale,
                          nbins=nvbins, numnorm=False)

        matrix = get_matrix(temp, nbins=nvbins)

        mask = ((np.isfinite(flux)) &
                (gpix == 0) &
                (xx > (nvbins-1)/2) &
                (xx < xx.size-(nvbins-1)/2))  # temporal
        #plt.plot(wave, flux)
        #plt.plot(wave, mask)
        #plt.show()


        # iterate over different lambdas
        for i, llam in enumerate(lamdas):
            delta_losvd, bestfit, metrics = solve(
                flux, matrix, losvd, mask, lam=llam, reg_type=reg_type)
            out_chi2[ibin, i] = metrics['chi2']
            out_reg2[ibin, i] = metrics['reg2']

            out_losvd2d[i, :, ibin] = losvd+delta_losvd
            # losvd2d_full[i, :, msk_bin] = np.resize(
            #     losvd+delta_losvd, (msk_nbins, nvbins))
            out_losvd2d_gau[:, ibin] = losvd
            out_bestfits[ibin, :, i] = bestfit

            # plt.cla()
            # plt.plot(flux)
            # # plt.plot(temp)
            # flux[~mask] = np.nan
            # plt.plot(flux)
            # plt.plot(bestfit, lw=0.3)
            # plt.xlim(1500,3000)
            # plt.pause(0.001)

        out_fluxes[ibin, :] = flux
        out_masks[ibin, :] = mask

    # Write output
    vbins = (np.arange(nvbins) - (nvbins-1) / 2)*velscale + vshift

    hdu0 = fits.PrimaryHDU()
    hdu0.header['VELSCALE'] = velscale
    hdul = fits.HDUList([
        hdu0,
        fits.ImageHDU(data=out_losvd2d, name='LOSVDS'),
        fits.ImageHDU(data=out_losvd2d_gau, name='LOSVDS_GAU'),
        # fits.ImageHDU(data=losvd2d_full, name='LOSVDS_FULL'),
        fits.ImageHDU(data=vbins, name='VBINS'),
        # fits.ImageHDU(data=wave, name='WAVE'),
        fits.ImageHDU(data=out_fluxes, name='FLUX'),
        fits.ImageHDU(data=out_masks, name='MASK'),
        fits.ImageHDU(data=out_bestfits, name='FIT'),
        fits.ImageHDU(data=get_reg_matrix(
            nvbins, reg_type=reg_type), name='REG_MATRIX'),
        fits.ImageHDU(data=out_chi2, name='CHI2'),
        fits.ImageHDU(data=out_reg2, name='REG2'),
        fits.ImageHDU(data=lamdas, name='LAMBDAS'),
    ])

    if ofile is None:
        ofile = f"{path}losvd_lambdas_{obj}.fits"
    hdul.writeto(ofile, overwrite=True)
    print(f"Write output in file: {ofile}")
