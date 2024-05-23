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

from netCDF4 import Dataset
import matplotlib.colors as colors
plt.rcParams['figure.facecolor']='white'
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


config  = parse_config('config_spec.yml')
out_dir = '/ltenas8/anneclaire/ICEGENESIS_2021/MXPol/Proc_data_w_Kdp/
SIZE_THRES = 1e7

for raw in sorted(glob.glob('/ltedata/ICEGENESIS_2021/Radar/2021-01-26*/XPOL-*.dat')):
    name = raw.split('/')[-1][:-4]
   if os.path.getsize(raw) > SIZE_THRES:
        add = 'FFT'
    else:
        add = 'DPP'
    
    outnameRHI = out_dir+'+name+'_'+add+'_RHI_Kdp.nc'
   
    
    if (os.path.exists(outnameRHI)) | (os.path.exists(outnamePPI)):
        continue
    
    try:
        header, records = read_raw_data(raw, config)    
        if 'RHI' in records: # NB in this campaign, PPI are only vertical -> no Kdp

            radar = process_dataset('RHI',header,records['RHI'],config)
            radar.fields['Psidp']['data'].mask=False
            radar.fields['Psidp']['data'].mask[radar.fields['Rhohv']['data'].data<0.5]=True
            radar.fields['Psidp']['data'].mask[radar.fields['SNRh']['data'].data<0]=True

            radar.fields['Kdp'] = pyart.retrieve.kdp_schneebeli(radar,psidp_field='Psidp',band='X',prefilter_psidp=True,filter_opt={'rhohv_field':'Rhohv'})[0] 
            azmean = radar.azimuth['data'].mean()
            write_cfradial(outnameRHI,radar)


    except Exception as e:
        print(e)
