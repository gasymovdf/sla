# Non-parametric LOSVD in __sla__ package

A code of nonparametric LOSVD recover for galaxy spectra

Stellar LOSVD recovery from the observed galaxy spectra is equivalent to a deconvolution and can be solved as a linear inverse problem. To overcome its ill-posed nature we apply smoothing regularization. Searching for an optimal degree of smoothing regularization is a challenging part of this approach. Here we present a non-parametric fitting technique and show real-world application to **MaNGA** spectral data cubes and **long-slit spectrum** of stellar counter-rotating galaxies.

More information can be found [here](https://arxiv.org) and examples [here](https://github.com/gasymovdf/sla)

### Authors
 - [Damir Gasymov](https://github.com/gasymovdf) (SAI Moscow Lomonosov State University)
 - [Ivan Katkov](https://github.com/ivan-katkov) (NYU Abu-Dhabi, SAI Moscow Lomonosov State University)
 
### Needed pip packages

There is many popular packages, [vorbin](https://pypi.org/project/vorbin/) by [Michele Cappellari](https://github.com/micappe) and my [pseudoslit](https://pypi.org/project/pseudoslit/) simple package for obtaining 2D LOSVD from a 3D cube of IFU-LOSVD. 

> pip install vorbin lmfit matplotlib astropy numpy pseudoslit shutil os glob PyPDF2 tqdm scipy sla
>

### Try it!

Run the command and download spectrum data:

> bash ./data/download.sh 
>

Test different use cases (with [NBursts analysis](https://ui.adsabs.harvard.edu/abs/2007IAUS..241..175C/abstract) or for clean IFU-spectrum MaNGA SDSS):

> python3 ./example/example_MaNGA_without_template.py
>
> python3 ./example/example_NBursts_with_template.py
>
> python3 ./example/example_NBursts_without_template.py

The result is shown in fits format and ploted in pdf in the **result** folder. 

