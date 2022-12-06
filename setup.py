#! /usr/bin/env python

# System imports
from setuptools import setup, find_packages
from pip import __file__ as pip_loc
from os import path
# Third-party modules - we depend on numpy for everything
import numpy


setup(  name        = "pyjacopo",
        description = "Python radar library for MXPOL",
        version     = "1.0",
        url='http://gitlab.epfl.ch/wolfensb/pyjacopo/',
        author='Daniel Wolfensberger - LTE EPFL',
        author_email='daniel.wolfensberger@epfl.ch',
        license='GPL-3.0',
        packages=['pyjacopo',
                  'pyjacopo.config_parser',
                  'pyjacopo.example',
                  'pyjacopo.algo',
                  'pyjacopo.raw_reader'],
        install_requires=[
          'arm_pyart',
          'numpy',
          'scipy',
          'xarray'],
        zip_safe=False,
        )


