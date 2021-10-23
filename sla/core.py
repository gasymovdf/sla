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

def get_weight_matrix(nvbins, nreg_bins, Vscale, lim_V=500):
    xx, yy = np.mgrid[0:nreg_bins, 0:nvbins]
    N = lim_V/Vscale
    w = np.abs(xx-(nreg_bins-1)/2) - N
    w[w >= 0] = np.min(abs(w))
    w = abs(w/w[int((nreg_bins-1)/2), 0])
    w = 1/w**2
    return w


def solve(flux2d, matrix3d, losvd2d, mask2d, velscale=50, lam=0, reg_type_losvd='L2', 
                reg_type_bins='L2', reg_num_bins=1, fit_bin=0, weight_reg=True):
    """
    Solve linear regularized problem.
    """
    nvbins = losvd2d.shape[1]
    npix = matrix3d.shape[1]
    dict_nreg_bins = {'L2':nvbins, 'smooth1':nvbins-1, 'smooth2':nvbins-2}
    nreg_bins = dict_nreg_bins[reg_type_losvd]
    matrix_extended = np.zeros((reg_num_bins * npix + reg_num_bins * nreg_bins, reg_num_bins * nvbins))
    mask_upd = np.ones((reg_num_bins * npix + reg_num_bins * nreg_bins))
    flux_upd = np.zeros((reg_num_bins * npix + reg_num_bins * nreg_bins))
    bounds = np.zeros((2, reg_num_bins * nvbins))
    for ibin in range(reg_num_bins):
        flux = flux2d[ibin]
        matrix = matrix3d[ibin]
        losvd = losvd2d[ibin]
        mask = mask2d[ibin]
        
        spec_convolved = np.dot(matrix, losvd)
        matrix_regularization = get_reg_matrix(nvbins, reg_type=reg_type_losvd)
        
        matrix_extended[ibin * npix:(ibin + 1)*npix, ibin * nvbins:(ibin + 1)*nvbins] = matrix
        matrix_extended[reg_num_bins * npix + ibin * nreg_bins:reg_num_bins * npix + (ibin + 1)*nreg_bins, ibin * nvbins:(ibin+1)*nvbins] = lam * matrix_regularization
        if weight_reg:
            matrix_weight = get_weight_matrix(nvbins, nreg_bins, velscale, lim_V=400)
            matrix_extended[reg_num_bins * npix + ibin * nreg_bins:reg_num_bins * npix + (ibin + 1)*nreg_bins, ibin * nvbins:(ibin+1)*nvbins] *= matrix_weight

        mask_upd[ibin * npix:(ibin + 1)*npix] = mask
        flux_upd[ibin * npix:(ibin + 1)*npix] = (flux - spec_convolved)    
        bounds[:, ibin*nvbins:(ibin + 1)*nvbins] = np.array([-losvd, 1.0 + 0*losvd])

        if reg_type_bins == 'smooth1' and ibin > 0:
            matrix_extended[reg_num_bins * npix + (ibin - 1) * nreg_bins:reg_num_bins * npix + ibin*nreg_bins, ibin * nvbins:(ibin+1)*nvbins] =\
                 -matrix_extended[reg_num_bins * npix + (ibin - 1) * nreg_bins:reg_num_bins * npix + ibin*nreg_bins, (ibin - 1) * nvbins:ibin*nvbins]
        elif reg_type_bins == 'smooth2' and ibin > 1:
            print('In development...')
    mask_upd = np.array(mask_upd, dtype=bool)
    res = optimize.lsq_linear(
        matrix_extended[mask_upd, :], flux_upd[mask_upd], bounds=bounds, method='bvls', max_iter=1000)
    delta_losvd = res.x[fit_bin*nvbins:(fit_bin + 1) * nvbins]
    losvd_full = delta_losvd + losvd2d[fit_bin]
    bestfit = np.dot(matrix_extended[:flux.size, :nvbins], losvd_full)

    # TBD to update
    metrics = dict(chi2=np.sum((flux[mask][:npix]-bestfit[mask])**2),
                   reg2=np.sum(np.dot(matrix_regularization, delta_losvd)**2))

    return losvd_full, delta_losvd, bestfit, metrics


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
                     lamdas=np.array([0, 0.5, 1.0]), ofile=None, error_2d=None,
                     path='./', reg_type_losvd='L2', reg_type_bins='L2', reg_num_bins=1, vshift=None,
                     obj='', wave=None, bin_sch=None, num_iter=1, i_iter=1, temp_losvd=None, 
                     monte_carlo_err=False, num_mc_iter=100, mc_iter=False):
    """
    Recover stellar LOSVD in each spectral bin for different smoothing
    parameter lambda (lambdas array).
    """
    if vshift is None:
        vshift = vels[0]
    nspec = specs.shape[0] 
    npix = specs.shape[1]

    xx_ar =  np.arange(npix)

    R, ind = {}, []
    R_0 = 0
    for i in range(nspec):
        R[i] = np.mean(np.where(bin_sch==i)[0])
        if i == 0:
            R_0 = R[0]
        R[i] -= R_0
    for w in sorted(R, key=R.get, reverse=True):
        ind.append(w) 
    ind = np.array(ind[::-1])

    out_losvd2d = np.zeros((lamdas.size, nvbins, nspec))
    out_delta_losvd2d = np.zeros((lamdas.size, nvbins, nspec))
    out_losvd2d_gau = np.zeros((nspec, nvbins))
    out_chi2 = np.zeros((nspec, lamdas.size))
    out_reg2 = np.zeros((nspec, lamdas.size))
    out_fluxes = np.zeros((nspec, npix))
    out_error3d = np.zeros((nspec, npix, lamdas.size))
    out_masks = np.zeros((nspec, npix))
    out_bestfits = np.zeros((nspec, npix, lamdas.size))
    matrix3d = np.zeros((nspec, templates.shape[1], nvbins))

    factor = np.nanmedian(templates)
    templates /= factor
    if specs.ndim == 2: 
        specs /= factor

    for ibin in range(nspec):
        out_losvd2d_gau[ibin, :] = get_losvd((vels[ibin]-vshift)/velscale, 
                                    sigs[ibin]/velscale, nbins=nvbins, numnorm=False)
        matrix3d[ibin] = get_matrix(templates[ibin], nbins=nvbins)
        out_fluxes[ibin] = specs[ibin]

    out_masks = ((np.isfinite(specs)) &
                (goodpixels == 0) &
                np.tile((xx_ar > (nvbins-1)/2), (nspec, 1)) &
                np.tile(((xx_ar < xx_ar.size-(nvbins-1)/2)), (nspec, 1)))
    out_masks = np.array(out_masks, dtype=int)

    for ibin in tqdm(range(nspec)):
        ib_num = np.where(ind == ibin)[0][0]
        if ib_num > reg_num_bins/2 and ib_num < nspec - 1 - reg_num_bins/2:
            fit_ind = ind[int(ib_num + 1 - (reg_num_bins)/2):int(ib_num + (reg_num_bins)/2)+1]
        elif ib_num <= reg_num_bins/2:
            fit_ind = ind[:reg_num_bins]
        elif ib_num >= 66 - 1 - reg_num_bins/2:
            fit_ind = ind[(66 - reg_num_bins):]

        for i, llam in enumerate(lamdas):
            if i_iter == 1:
                solution = solve(specs[fit_ind], matrix3d[fit_ind], 
                    out_losvd2d_gau[fit_ind], out_masks[fit_ind], 
                    velscale=velscale, lam=llam, reg_type_losvd=reg_type_losvd, 
                    reg_type_bins=reg_type_bins, reg_num_bins=reg_num_bins, fit_bin=np.where(fit_ind == ibin)[0][0])
                
            else:
                solution = solve(specs[fit_ind], matrix3d[fit_ind], 
                    temp_losvd[i, :, fit_ind], out_masks[fit_ind], 
                    velscale=velscale, lam=llam, reg_type_losvd=reg_type_losvd, 
                    reg_type_bins=reg_type_bins, reg_num_bins=reg_num_bins, fit_bin=np.where(fit_ind == ibin)[0][0])
            
            losvd_full, delta_losvd, bestfit, metrics = solution
            out_chi2[ibin, i] = metrics['chi2']
            out_reg2[ibin, i] = metrics['reg2']
            out_losvd2d[i, :, ibin] = losvd_full
            out_delta_losvd2d[i, :, ibin] = delta_losvd
            out_bestfits[ibin, :, i] = bestfit
            out_error3d[ibin, :, i] = error_2d[ibin]
            
    if i_iter != num_iter:
        recover_losvd_2d(specs, templates, goodpixels, vels, sigs, velscale, error_2d=error_2d,
                         nvbins=nvbins, lamdas=lamdas, path=path, reg_type_losvd=reg_type_losvd,
                         obj=obj, wave=wave, num_iter=num_iter, i_iter=i_iter+1, temp_losvd=out_losvd2d,
                         reg_type_bins=reg_type_bins, reg_num_bins=reg_num_bins, bin_sch=bin_sch)
        return 0        
    if mc_iter:
        return out_losvd2d
    if monte_carlo_err:
        mc_rec_losvd = np.zeros((num_mc_iter, lamdas.size, nvbins, nspec))
        for i in range(num_mc_iter):
            noise = np.random.rand(nspec, npix, len(lamdas)) * out_error3d
            bestfit_noise = out_bestfits + noise
            mc_rec_losvd[i] = recover_losvd_2d(bestfit_noise, templates, goodpixels, vels, sigs, velscale,
                         lamdas=lamdas, reg_type_losvd=reg_type_losvd, reg_type_bins='L2', reg_num_bins=1,
                         nvbins=nvbins, obj=obj, wave=wave, mc_iter=True, error_2d=error_2d, bin_sch=bin_sch)
        losvd_err = np.std(np.array(mc_rec_losvd), axis=0)
        errors_name = f"MCers_{num_mc_iter}it"
    else:
        losvd_err = np.zeros_like(out_losvd2d)
        errors_name = "ZEROers"

    # Write output
    vbins = (np.arange(nvbins) - (nvbins-1) / 2)*velscale + vshift
    hdu0 = fits.PrimaryHDU()
    hdu0.header['VELSCALE'] = velscale
    hdu0.header['NVBINS'] = nvbins
    hdu0.header['REG_LSVD'] = reg_type_losvd
    hdu0.header['REG_BINS'] = reg_type_bins
    hdu0.header['REG_NBIN'] = reg_num_bins
    if monte_carlo_err:
        hdu0.header['MC_ITERS'] = num_mc_iter
    hdu0.header['N_ITERS'] = i_iter
    hdul = fits.HDUList([
        hdu0,
        fits.ImageHDU(data=out_losvd2d, name='LOSVDS'),
        fits.ImageHDU(data=out_delta_losvd2d, name='DELTA_LOSVDS'),
        fits.ImageHDU(data=out_losvd2d_gau, name='LOSVDS_GAU'),
        fits.ImageHDU(data=losvd_err, name='ERR_LOSVDS'),
        fits.ImageHDU(data=vbins, name='VBINS'),
        fits.ImageHDU(data=wave, name='WAVE'),
        fits.ImageHDU(data=out_fluxes, name='FLUX'),
        fits.ImageHDU(data=out_masks, name='MASK'),
        fits.ImageHDU(data=out_bestfits, name='FIT'),
        fits.ImageHDU(data=get_reg_matrix(
            nvbins, reg_type=reg_type_losvd), name='REG_MATRIX'),
        fits.ImageHDU(data=out_chi2, name='CHI2'),
        fits.ImageHDU(data=out_reg2, name='REG2'),
        fits.ImageHDU(data=lamdas, name='LAMBDAS'),
    ])

    if ofile is None:
        ofile = f"{path}nonpar_losvd_{obj}_{num_iter}_iter_{reg_type_losvd}_{reg_type_bins}_{reg_num_bins}bins_{errors_name}.fits"
    hdul.writeto(ofile, overwrite=True)
    print(f"Write output in file: {ofile}")

