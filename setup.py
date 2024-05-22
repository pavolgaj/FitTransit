try: from setuptools import setup
except ImportError: from distutils.core import setup

import sys

py_version = sys.version_info[0]

if not py_version==(3):	raise RuntimeError('Python 3 is required!	')

dist=setup(name="FitTransit",
      version="0.1.2",
      description="Fitting transits of exoplanets",
      author="Pavol Gajdos",
      zip_safe = False,
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering :: Astronomy"],
      url='https://github.com/pavolgaj/FitTransit',
      install_requires=['numpy>=1.10.2','matplotlib>=1.5.0','scipy>=1.5.0','batman-package>=2.4.0'],
      extras_require={'MCMC': ['emcee>=3.0.0','corner','tqdm']},
      py_modules=["FitTransit/__init__","FitTransit/Transit_class","FitTransit/info_mc","FitTransit/info_ga","FitTransit/ga"]
)
