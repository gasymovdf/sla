import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
import sla
from sla.utils import read_nbursts_results
c = 299792.458

def processed(file, lambdas, obj='', reg_type='smooth1', lam_range=[4500, 5500]):
    nbres = read_nbursts_results(file, bin_sch=True) # , Age_Met=True)
    wave, spec, err, template, comps, goodpixels, vels, sigs, velscale, bin_schema = nbres
    fit_emis = comps[:, 1, :]
    comps = comps[:, 0, :]
    vels = vels[:, 0, :]
    sigs = sigs[:, 0, :]
    spec -= np.sum(fit_emis, axis=0)
    z = np.mean(vels) / (3*10**5)
    rm_lines=[[5461.0, 5.0], [5577.0, 10.0],[5892, 15],[6280, 10],[6300, 10.0],[6364., 10.0], # night sky lines
                [3727.0*(1.0+z), 10], # [OII] dublet
                [3971.0*(1.0+z), 10], # Bulmer line H5 (Heta) overlapped with H Ca
                [4339.0*(1.0+z), 10], [4861.0*(1.0+z), 10.], [5894.5*(1.0+z), 15.0],
                [4959.0*(1.0+z), 10], [5007.0*(1.0+z), 10.], [5199.0*(1.0+z), 8.0],
                [6563.0*(1.0+z), 10], [6548.0*(1.0+z), 10.], [6584.0*(1.0+z), 8.0],
                [6716.0*(1.0+z), 10], [6731.0*(1.0+z), 10.]]
              # [5175.0*(1.0+z),15]]
    goodpixels_emis_exclude = (goodpixels == 1) | ~np.array([((wave > lam_range[0]) & (wave < lam_range[1]))]*spec.shape[0])
    for line in rm_lines:
        goodpixels_emis_exclude = (goodpixels_emis_exclude == 1) | np.array([((wave > line[0] - line[1]/2) & (wave < line[0] + line[1]/2))]*spec.shape[0])

    sla.recover_losvd_2d(spec, template, goodpixels_emis_exclude, vels, sigs, velscale,
                         lamdas=lambdas, path='../result/', error_2d=err, lim_V_fit=[-400, 400], lim_V_weight=400,
                         reg_type_losvd=reg_type, reg_type_bins='L2', reg_num_bins=1, plot=True,
                         obj=obj, wave=wave, num_iter=1, monte_carlo_err=False, num_mc_iter=100, bin_sch=bin_schema) 

if __name__ == '__main__':
    file, obj = '../data/PGC066551_pegase_emis_4500_5490.fits', '1-179561'  
    lambdas = 10**np.linspace(-0.5, 0.5, 20)
    processed(file, lambdas, obj=obj, reg_type='smooth1')