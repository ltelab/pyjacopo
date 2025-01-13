import pyart
import numpy as np
from pyjacopo import read_cfradial, write_cfradial, read_raw_data , parse_config, process_dataset
import glob
#from pyart.default_config import DEFAULT_METADATA
import sys
# sys.path.append('/ltenas3/anneclaire/ICEGENESIS_2021/MXPol/')
import scipy.io as sio
import datetime
import os
import re
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.facecolor']='white'
# from ml_detection.ml_detection import detect_ml
from netCDF4 import Dataset

# for raw in sorted(glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/30/*20140130-1[7-9]*.dat')+
#                   glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/30/*20140130-2[0-4]*.dat')+
#                   glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/31/*20140131-0[0-5]*.dat')+
#                   glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/31/*20140131-0[7-9]*.dat')+
#                   glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/31/*20140131-[1-2][0-9]*.dat')):
config  = parse_config('config_spec.yml')

for raw in sorted(glob.glob('/ltedata/PLATO_2019/Radar/Raw_data/2019/01/10/*.dat')):
#                   glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/30/*20140130-2[0-4]*.dat')+
#                   glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/31/*20140131-0[0-5]*.dat')+
#                   glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/31/*20140131-0[7-9]*.dat')+
#                   glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/31/*20140131-[1-2][0-9]*.dat')):
#                  glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/31/*20140131-06*.dat')):
    try:
        if os.path.exists('../PLATO/Radar/spectra_w_covariance/'+raw.split('/')[-1][:-4]+'.nc'):
            continue
        if (os.path.getsize(raw) > int(2e7) & os.path.getsize(raw) < int(5e7)):                        
            header, records = read_raw_data(raw, config)
            if not('PPI' in records):
                continue
            radar = process_dataset('PPI',header,records['PPI'],config)
            if not (np.abs(radar.elevation['data'].mean()-90) < 2):
                continue
            write_cfradial('../PLATO/Radar/spectra_w_covariance/'+raw.split('/')[-1][:-4]+'.nc',radar)
            print('../PLATO/Radar/spectra_w_covariance/'+raw.split('/')[-1][:-4]+'.nc')
    except Exception as e:
        print(e)
        
        

# for raw in sorted(glob.glob('/ltedata/ICEGENESIS_2021/Radar/2021-01-27/*20210127-[1-2][0-9]*.dat')):
# #                   glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/30/*20140130-2[0-4]*.dat')+
# #                  glob.glob('/ltedata/CLACE2014/Radar/Raw_data/2014/01/31/*20140131*.dat')):
#     try:
#         config  = parse_config('/ltenas3/anneclaire/ICEGENESIS_2021/MXPol/config_spec.yml')

#         header, records = read_raw_data(raw, config)
#         if not('PPI' in records):
#             continue
#         radar = process_dataset('PPI',header,records['PPI'],config)
#         if not (np.abs(radar.elevation['data'].mean()-90) < 2):
#             continue
#         write_cfradial('/ltenas3/anneclaire/ICEGENESIS_2021/MXPol/spectra/'+raw.split('/')[-1][:-4]+'.nc',radar)
#     except Exception as e:
#         print(e)