import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
import sla
from sla.utils import read_manga_data

def processed(file, lambdas, obj='', reg_type='smooth1', lam_range=[4800, 5700]):
    manga = read_manga_data(file, lam_range=lam_range)
    wave, spec, err, goodpixels, velscale, Res, bin_sch2d = manga
    z = 0.0234372

    # Age0, Met0 = template
    Age0, Met0 = [3000], [0]
    Res, mdegree = np.mean(Res), 23
    ssp = '../template/'

    template = sla.calc_template(ssp_dir=ssp, wave_s=wave, spec_s=[spec[10]], Age0=Age0, Met0=Met0, R=Res, mdegree=mdegree, z=z)
    template = np.array([template[0]] * spec.shape[0])    
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

    sla.recover_losvd_2d(spec, template, goodpixels_emis_exclude, velscale=velscale,
                         lamdas=lambdas, path='../result/', error_2d=err, lim_V_fit=[-500, 500], lim_V_weight=500,
                         reg_type_losvd=reg_type, obj=obj, wave=wave, num_iter=3, bin_sch=bin_sch2d, manga=True, plot=True,
                         monte_carlo_err=False, num_mc_iter=100)


if __name__ == '__main__':
    file, obj = '../data/manga-8155-3702-LOGCUBE.fits.gz', '1-38543'
    lambdas = 10**np.linspace(-0.5, 0.5, 20)
    processed(file, lambdas, obj=obj, reg_type='smooth1')