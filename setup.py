import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='sla',  
     version='1.2.1',
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
 )