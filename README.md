# Stellar LOSVD Analysis (SLA)

Code for the non-parametrical recovery of stellar Line-Of-Sight Velocity Distributions (LOSVD) from galaxy spectra.

Stellar LOSVD recovery from the observed galaxy spectra is equivalent to a deconvolution and can be solved as a linear inverse problem. To overcome its ill-posed nature we apply smoothing regularization. Here we present a non-parametric fitting technique and show real-world application to **MaNGA** spectral data cubes and **long-slit spectrum** of stellar counter-rotating galaxies.

More information about this technique can be found in [ADASS XXXI conference proceeding](https://arxiv.org/abs/2112.08386) and in [examples](example/)


## Installation
Use the package manager `pip` to install __sla__.

```
pip install sla==1.3.3
```

The __sla__ package uses the following dependencies, which should be installed when installing through the pip manager:

```
pip install -r requirements.txt
```

- numpy
- scipy
- matplotlib
- astropy
- lmfit
- vorbin
- pseudoslit
- glob
- PyPDF2
- tqdm


## Usage

SLA usage examples are shown for stellar counter-rotating galaxy PGC 66551 (Gasymov, Katkov et al. in prep.). To run examples, first download test dataset which includes:

- MaNGA spectral cube
- long-slit spectra.

```
sh ./data/download.sh 
```

### Use case examples

#### Example 1

Example of stellar LOSVD recovery along pseudoslit spectrum taken from MaNGA spectral cube along the major axis
```
cd example
python3 example_MaNGA_without_template.py
```

#### Example 2

LOSVD is determined from the RSS long slit spectrum using the un-broadened stellar population template (SSP PEGASE.HR), which was constructed by applying in advance [NBursts](https://ui.adsabs.harvard.edu/abs/2007IAUS..241..175C/abstract) full spectral fitting method.

```
cd example
python3 example_NBursts_with_template.py
```


#### Example 3

The same as in the previous example, but without using NBursts output. The necessary stellar population template is selected from the model grid for given approximate SSP parameters

```
cd example
python3 example_NBursts_without_template.py
```


The resulting file fits file PDF figure will be stored in the **result** folder. 


## Authors
 - [Damir Gasymov](https://github.com/gasymovdf) (Sternberg Astronomical Institute, Lomonosov Moscow State University)
 - [Ivan Katkov](https://github.com/ivan-katkov) (NYU Abu Dhabi, Lomonosov Moscow State University)
