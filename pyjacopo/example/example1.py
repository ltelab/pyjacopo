#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 13:52:04 2018

@author: wolfensb
"""

from pyjacopo import parse_config, read_raw_data
from pyjacopo import write_cfradial, read_cfradial
from pyjacopo import process_dataset

# STEP 1: read user configuration file

config = parse_config('config.yml')

'''
Note that the format of the config is in yaml and that the three files used by
Jacopo's IDL code are merged in one, "main", "location" and "products".
Structure of this config file is quite complicated, check the IDL documentation,
I used the same options and structure. The default values are provided in 
defaults.py in the parse submodule and the valid options in valid_input.py
'''

# STEP 2: read binary data file

# APRES3 campaign FFT RHI
#raw_filename = '/ltedata/APRES3/2015-16/Radar/Raw_data/2016/01/28/XPOL-20160128-055336.dat' 
raw_filename = '/ltedata/PLATO_2019/Radar/Raw_data/2019/01/06/XPOL-20190106-002520.dat' 
header, records = read_raw_data(raw_filename, config)

'''
There can be several scans in the file, they will be separated in the records, 
e.g. records['RHI1'], records['RHI2'], records['PPI1'], records['PPI2'], for 
example. However the header will always be unique and valid for all records
(same type of data aquisition for all records within a file).
Not that the separation between two RHI or PPI scans might not be perfect and 
not correspond totally to what Jacopo did. 
In the config file there is the "ang_tol" option that allows to specify how 
the separation threshold ==> two consecutive records with a difference of more
than ang_tol in their dependent angle (elevation for RHI, azimuth for PPI),
will be considered as from a different scan.
'''

# STEP 3: Process from level 0 to level 1
pyart_instance = process_dataset('RHI',header, records['RHI'], config)
'''
Note that you need to specify explicitely the type of dataset
Processing can take a bit of time for FFT or FFT2 (10 - 30 sec). For DPP it is 
very fast. Note that the code is parallelized if you use parallel: True in
the products part of the config file!

For FFT(2) acquisition modes, the "discard_spectra" option in the config file
allows to choose if 3D variables (spectral) need to be stored in the output or
not. Here they are kept (sPOWH, sPOWH, sCC, sVel)

Important: the output is a slightly modified PyART Core class instance, so all
pyart functions should work directly on it!!!
'''
#
# STEP 4: Apply PyART functions and save to netCDF
import pyart
import matplotlib.pyplot as plt

display = pyart.graph.RadarDisplay(pyart_instance)
fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(111)
display.plot('Zh', 0, vmin=-16, vmax=64.0)


## Save to netCDF
write_cfradial('./files/example1_rhi.nc', pyart_instance)
# When you load, you get back the pyart instance
pyart_instance = read_cfradial('./files/example1_rhi.nc')


