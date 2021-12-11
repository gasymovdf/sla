from matplotlib.cbook import ls_mapper
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.optimize.nonlin import nonlin_solve
from scipy.special import legendre
from astropy.io import ascii
from tqdm import tqdm
import glob
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import PyPDF2, os
import shutil
from pseudoslit import calc
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 20
plt.rcParams['font.weight']= 'normal' 
c = 299792.458

def gauss(x, sig):
    return 1/(2 * np.pi * sig**2)**0.5 * np.exp(-x**2/sig**2)

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
                reg_type_bins='L2', reg_num_bins=1, fit_bin=0, weight_reg=True, lim_V=400):
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
            matrix_weight = get_weight_matrix(nvbins, nreg_bins, velscale, lim_V=lim_V)
            matrix_extended[reg_num_bins * npix + ibin * nreg_bins:reg_num_bins * npix + (ibin + 1)*nreg_bins, ibin * nvbins:(ibin+1)*nvbins] *= matrix_weight

        mask_upd[ibin * npix:(ibin + 1)*npix] = mask
        flux_upd[ibin * npix:(ibin + 1)*npix] = (flux - spec_convolved)
        k = 0 if np.sum(losvd) == 0 else 1
        bounds[:, ibin*nvbins:(ibin + 1)*nvbins] = np.array([-k * losvd, 1.0 + 0*losvd])

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


def calc_template(wave_s=[], spec_s=[], ssp_dir='', Age0=[5000], Met0=[0], R=0, sig_lsf=0, mdegree=20, z=0, mask=None):
    age = np.array(ascii.read(f"{ssp_dir}/age.csv")['age'])
    met = np.array(ascii.read(f"{ssp_dir}/met.csv")['met'])
    if len(Age0) == 1 or len(Age0) == spec_s.shape[0]:
        spec_t_out = np.zeros((len(Age0), len(wave_s)))
    else:
        raise ValueError('Bad Age0 and Met0!')
    if mask is None:
        mask = np.zeros(np.array(spec_s).shape, dtype=int)

    for Agei, Meti, i in zip(Age0, Met0, range(len(Age0))):
        met_f, age_f = met[np.argmin(abs(met - Meti))], age[np.argmin(abs(age - Agei))]
        Nfile, Nspec = np.argmin(abs(met - Meti)) + 1, np.argmin(abs(age - Agei))
        print(f"Working with template {age_f} Myr and metallicity {met_f} instead of {Agei:.1f} Myr and {Meti:.1f}")

        hdu = fits.open(f'{ssp_dir}/SB_{Nfile}_Salp_0.1_120_paper_ver3_76cour_new.fits')
        hdr = hdu[0].header
        spec_t = hdu[0].data[Nspec]
        lam0, dlam = hdr['CRVAL1'], hdr['CDELT1']
        wave_t = np.arange(lam0, lam0 + (len(spec_t)-1)*dlam+1e-5, dlam) * (1 + z)
        lam_scale = np.mean(np.diff(wave_s))

        if R == 0 and sig_lsf != 0 :
            sig_lam_lsf = sig_lsf/c * np.mean(wave_s) / lam_scale
        elif R != 0 and sig_lsf == 0:
            sig_lam_lsf = np.mean(wave_t / (3 * R) / lam_scale)
        else:
            raise ValueError('Zero resolution and dispersion of LSF!')
        lsf = gauss(np.arange(-10 * sig_lam_lsf, 10*sig_lam_lsf, dlam), sig_lam_lsf)

        spec_t_conv = np.convolve(spec_t, lsf, 'same')
        spec_t_rebin = np.zeros(len(spec_s[i]))
        for j in range(len(wave_s)):
            if j == 0:
                wave_range = (wave_t >= wave_s[j]) & (wave_t <= (wave_s[j] + wave_s[j+1])/2)
            elif j == len(wave_s) - 1:
                wave_range = (wave_t >= (wave_s[j] + wave_s[j-1])/2) & (wave_t <= wave_s[j])
            else:
                wave_range = (wave_t >= (wave_s[j] + wave_s[j-1])/2) & (wave_t <= (wave_s[j] + wave_s[j+1])/2)
            spec_t_rebin[j] = np.mean(spec_t_conv[wave_range])
            
        spec_t_rebin /= np.nanmedian(spec_t_rebin)
        spec_s_i = spec_s[i] / np.nanmedian(spec_s[i])
        rel_s_t = spec_s_i[mask[i]==0] / spec_t_rebin[mask[i]==0]
        x_rel_s_t = np.linspace(-1, 1, len(rel_s_t))
        x_rel_s_t_nonan = np.delete(x_rel_s_t, np.argwhere(np.isnan(rel_s_t)))
        rel_s_t = np.delete(rel_s_t, np.argwhere(np.isnan(rel_s_t)))
        a_fit = np.polynomial.legendre.legfit(x_rel_s_t_nonan, rel_s_t, mdegree)
        leg_fit = np.sum(np.array([a_fit[i] * legendre(i)(x_rel_s_t) for i in range(mdegree)]), axis=0)
        spec_t_out[i][mask[i]==0] = spec_t_rebin[mask[i]==0] * leg_fit
        spec_t_out[i][mask[i]] = spec_t_rebin[mask[i]] * np.mean(leg_fit)
        spec_t_out[i][np.isnan(spec_t_out[i])] = 0
    return spec_t_out


def recover_losvd_2d(specs, templates, goodpixels, vels=None, sigs=None, velscale=50, 
                     lamdas=np.array([0]), ofile=None, error_2d=None, z=0, plot=False,
                     path='./', reg_type_losvd='L2', reg_type_bins='L2', reg_num_bins=1, vshift=None,
                     obj='', wave=None, bin_sch=None, num_iter=1, i_iter=1, temp_losvd=None, manga=False, pa=90,
                     monte_carlo_err=False, num_mc_iter=100, mc_iter=False, lim_V_fit=[-500, 500], lim_V_weight=500):
    """
    Recover stellar LOSVD in each spectral bin for different smoothing
    parameter lambda (lambdas array).
    """
    if vshift is None:
        if vels is None:
            gauss_conv = False
            vshift = c * z
        else:
            gauss_conv = True
            vshift = vels[0]
    nvbins = int((lim_V_fit[1]-lim_V_fit[0])/velscale)
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
    out_error3d = np.zeros((nspec, npix, lamdas.size))
    out_masks = np.zeros((nspec, npix))
    out_bestfits = np.zeros((nspec, npix, lamdas.size))
    matrix3d = np.zeros((nspec, templates.shape[1], nvbins))
    if not mc_iter:
        out_fluxes = np.zeros((nspec, npix))
    templates = (templates.T / np.nanmedian(templates, axis=1)).T
    specs = (specs.T / np.nanmedian(specs, axis=1)).T
    
    for ibin in range(nspec):
        if gauss_conv:
            out_losvd2d_gau[ibin, :] = get_losvd((vels[ibin]-vshift)/velscale, 
                                        sigs[ibin]/velscale, nbins=nvbins, numnorm=False)
        else:
            out_losvd2d_gau[ibin, :] = np.zeros(nvbins)
        matrix3d[ibin] = get_matrix(templates[ibin], nbins=nvbins)
        if not mc_iter:
            out_fluxes[ibin] = specs[ibin]

    out_masks = ((goodpixels == 0) &
                np.tile((xx_ar > (nvbins-1)/2), (nspec, 1)) &
                np.tile(((xx_ar < xx_ar.size-(nvbins-1)/2)), (nspec, 1)))
    if not mc_iter:
        out_masks = (out_masks) & (np.isfinite(specs))
    else:
        out_masks = (out_masks) & (np.isfinite(np.sum(specs, axis=2)))
    out_masks = np.array(out_masks, dtype=int)
    for ibin in tqdm(range(nspec)):
        ib_num = np.where(ind == ibin)[0][0]
        if ib_num > reg_num_bins/2 and ib_num < nspec - 1 - reg_num_bins/2:
            fit_ind = ind[int(ib_num + 1 - (reg_num_bins)/2):int(ib_num + (reg_num_bins)/2)+1]
        elif ib_num <= reg_num_bins/2:
            fit_ind = ind[:reg_num_bins]
        elif ib_num >= nspec - 1 - reg_num_bins/2:
            fit_ind = ind[(nspec - reg_num_bins):]
        for i, llam in enumerate(lamdas):
            if i_iter == 1:
                if mc_iter:
                    solution = solve(specs[fit_ind][:, :, i], matrix3d[fit_ind], 
                        out_losvd2d_gau[fit_ind], out_masks[fit_ind],
                        velscale=velscale, lam=llam, reg_type_losvd=reg_type_losvd, lim_V=lim_V_weight,
                        reg_type_bins=reg_type_bins, reg_num_bins=reg_num_bins, fit_bin=np.where(fit_ind == ibin)[0][0])
                else:
                    solution = solve(specs[fit_ind], matrix3d[fit_ind], 
                        out_losvd2d_gau[fit_ind], out_masks[fit_ind], 
                        velscale=velscale, lam=llam, reg_type_losvd=reg_type_losvd, lim_V=lim_V_weight,
                        reg_type_bins=reg_type_bins, reg_num_bins=reg_num_bins, fit_bin=np.where(fit_ind == ibin)[0][0])                
            else:
                solution = solve(specs[fit_ind], matrix3d[fit_ind], 
                    temp_losvd[i, :, fit_ind], out_masks[fit_ind], 
                    velscale=velscale, lam=llam, reg_type_losvd=reg_type_losvd, lim_V=lim_V_weight,
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
                         lamdas=lamdas, path=path, reg_type_losvd=reg_type_losvd, z=z,
                         obj=obj, wave=wave, num_iter=num_iter, i_iter=i_iter+1, temp_losvd=out_losvd2d, 
                         reg_type_bins=reg_type_bins, reg_num_bins=reg_num_bins, bin_sch=bin_sch, 
                         lim_V_fit=lim_V_fit, lim_V_weight=lim_V_weight, plot=plot, manga=manga,
                         monte_carlo_err=monte_carlo_err, num_mc_iter=num_mc_iter)
        return 0        
    if mc_iter:
        return out_losvd2d
    if monte_carlo_err:
        mc_rec_losvd = np.zeros((num_mc_iter, lamdas.size, nvbins, nspec))
        for i in range(num_mc_iter):
            print(f'{i+1}/{num_mc_iter} iteration of Monte-Carlo')
            noise = np.random.rand(nspec, npix, len(lamdas)) * out_error3d
            bestfit_noise = out_bestfits + noise
            mc_rec_losvd[i] = recover_losvd_2d(bestfit_noise, templates, goodpixels, vels, sigs, velscale,
                         lamdas=lamdas, reg_type_losvd=reg_type_losvd, reg_type_bins='L2', reg_num_bins=1,
                         obj=obj, wave=wave, mc_iter=True, error_2d=error_2d, bin_sch=bin_sch, z=z,
                         lim_V_fit=lim_V_fit, lim_V_weight=lim_V_weight, manga=manga, plot=plot)
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
        fits.ImageHDU(data=bin_sch, name='BIN_SCHEMA'),
    ])

    if ofile is None:
        ofile = f"{path}nonpar_losvd_{obj}_{num_iter}_iter_{reg_type_losvd}_{reg_type_bins}_{reg_num_bins}bins_{errors_name}.fits"
    ofile_pdf = ofile.replace('fits', '.pdf')
    hdul.writeto(ofile, overwrite=True)
    if plot:
        if manga:
            plot_losvd_manga(out_losvd2d, vbins, lamdas, bin_sch, obj=obj, path=path, ofile=ofile_pdf, pa=pa)
        else:
            plot_losvd_longslit(out_losvd2d, vbins, R, lamdas, bin_sch, obj=obj, path=path, ofile=ofile_pdf, vshift=vshift)
    print(f"Write output in file: {ofile}")

def create_pdf(directory_in, out_name='losvd_all'):
    files = glob.glob(f'{directory_in}/nonpar_losvd_*.pdf') # os.listdir(directory_in)     
    files = sorted(files)
    merger = PyPDF2.PdfFileMerger()    
    for filename in files:
        merger.append(fileobj=open(filename,'rb'))    
    merger.write(open(out_name, 'wb'))

def plot_losvd_longslit(losvd, vbins, Rbins, lams, bin_sch, obj='', path='.', ofile='res.pdf', vshift=0):
    save_fold = f'{path}/nonp_lvd_plot/'
    if not os.path.exists(save_fold):
        os.mkdir(save_fold)
    
    for l, lam in tqdm(enumerate(lams)):
        fig, ax = plt.subplots(1, 1, figsize=(13, 8))
        fig.patch.set_facecolor('white')
        losvd2d = []
        for i in bin_sch:
            losvd2d.append(losvd[l, :, i])
        losvd2d = np.rot90(losvd2d)[::-1]
        ax.set_title(f'Non-parametric LOSVD for {obj} with lambda = {lam:.2f}')
        im = ax.imshow(losvd2d, cmap='twilight', extent=[-max(Rbins)/2, max(Rbins)/2, min(vbins-vshift), max(vbins-vshift)])
        # ax.contour(losvd2d, levels=4, cmap='Reds', linewidths=0.5, extent=[min(Rbins), max(Rbins), min(vbins), max(vbins)])
        ax.set_aspect('auto')
        ax.set_xlim(-max(Rbins)/2, max(Rbins)/2)
        ax.set_ylim(min(vbins-vshift), max(vbins-vshift)) 
        ax.set_xlabel('R, pixels')
        ax.set_ylabel('LOSVD, km/s')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='major', length=6)
        ax.tick_params(which='minor', length=4)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='10%', pad=0.1)
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.savefig(f'{save_fold}/nonpar_losvd_{lam:.2f}.pdf')
        plt.close()
    create_pdf(save_fold, ofile)
    if os.path.exists(save_fold):
        shutil.rmtree(save_fold)

def plot_losvd_manga(losvd, vbins, lams, bin_sch, obj='', path='.', ofile='res.pdf', slit_width=1, prec=100, scale=0.5, pa=90):
    save_fold = f'{path}/nonp_lvd_plot/'
    if not os.path.exists(save_fold):
        os.mkdir(save_fold)
    for l, lam in tqdm(enumerate(lams)):
        a, b = bin_sch.shape
        manga2d_losvds = np.zeros((a, b, losvd.shape[1]))
        for i in range(a):
            for j in range(b):
                if bin_sch[i, j] != -1:
                    manga2d_losvds[i, j, :] = losvd[l, :, int(bin_sch[i, j])]
        
        fig, ax = plt.subplots(1, 1, figsize=(17, 8), gridspec_kw={'wspace': 0.3})
        r_ax, losvd_synth1 = calc.pseudoslit(manga2d_losvds, pa, width=slit_width/scale, precision=prec)
        x = np.array(r_ax)*scale
        im = ax.imshow(losvd_synth1, cmap='twilight', extent=[min(x), max(x), min(vbins), max(vbins)]) #, cmap='jet' , norm=LogNorm(vmin=v[0], vmax=v[1])
        ax.set_aspect('auto')
        ax.set_xlabel('R, arcsec')
        ax.set_ylabel('LOSVD, km/s')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='major', length=6)
        ax.tick_params(which='minor', length=4)
        ax.set_title(f'Non-parametric LOSVD from MaNGA SDSS for {obj} with lambda = {lam:.2f}')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='10%', pad=0.1)
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.savefig(f'{save_fold}/nonpar_losvd_{lam:.2f}.pdf')
        plt.close()
    create_pdf(save_fold, ofile)
    if os.path.exists(save_fold):
        shutil.rmtree(save_fold)