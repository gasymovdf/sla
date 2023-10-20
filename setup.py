import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='sla',  
     version='1.4.1',
     author="Gasymov Damir",
     author_email="gasymov.df18@physics.msu.ru",
     description="Non-parametric LOSVD analysis for galaxy spectra",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/gasymovdf/sla",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     install_requires=["pseudoslit==0.0.2", 
                        "numpy==1.21.4",
                        "scipy==1.7.3",
                        "matplotlib==3.5.1",
                        "astropy==5.0",
                        "lmfit==1.0.3",
                        "vorbin==3.1.4",
                        "glob2==0.7",
                        "PyPDF2==1.26.0",
                        "tqdm==4.62.3"]
 )