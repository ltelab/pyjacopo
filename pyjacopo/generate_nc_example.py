import pyart
from pyjacopo import parse_config, read_raw_data
from pyjacopo import write_cfradial, read_cfradial
from pyjacopo import process_dataset
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import pandas as pd
import os

config  = parse_config('config_spec.yml')
from netCDF4 import Dataset
import matplotlib.colors as colors
plt.rcParams['figure.facecolor']='white'
from scipy import stats


out_dir = '/ltenas8/users/anneclaire/ICEGENESIS_2021/MXPol/'

for raw in sorted(glob.glob('/ltedata/ICEGENESIS_2021/Radar/Raw_data/2021/01/3*/XPOL-*.dat')):
    print(raw)
    name = raw.split('/')[-1][:-4]
    if os.path.getsize(raw) < 1e7:
        add = 'DPP'
    else:
        add = 'FFT'

    outname_RHI = out_dir + name + '_' + add + '_RHI.nc'
    outname_PPI = out_dir + name + '_' + add + '_PPI.nc'
    
    if (os.path.exists(outname_PPI)) | (os.path.exists(outname_RHI)):
        print('exists')
        continue
    try:
        header, records = read_raw_data(raw, config)    
        if 'RHI' in records:
            print('RHI')
            radar = process_dataset('RHI',header,records['RHI'],config)
            outname = outname_RHI
        if 'PPI' in records:
            print('PPI')
            radar = process_dataset('PPI',header,records['PPI'],config)
            outname = outname_PPI
        
        write_cfradial(outname,radar)
        print(outname, ' success')
        
    except Exception as e:
        print(outname, e)
