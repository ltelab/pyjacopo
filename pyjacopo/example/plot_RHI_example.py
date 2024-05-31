import pyart
import matplotlib.pyplot as plt
import numpy as np
import glob
from netCDF4 import Dataset
import glob
from tqdm import tqdm

for f in tqdm(sorted(glob.glob('/ltedata/ICEGENESIS_2021/Radar/Proc_data/2021/01/27/XPOL*.nc'))):
    try:
        name = f.split('/')[-1][:-3]

        radar = pyart.io.read_cfradial(f)

        Rhohv = radar.fields['Rhohv']['data'].data
        Psidp = radar.fields['Psidp']['data'].data
        mask_from_rhohv = (Rhohv<0.2) | (Psidp>-30)
        radar.fields['Zh']['data'].mask=mask_from_rhohv
        radar.fields['Zdr']['data'].mask=mask_from_rhohv
        radar.fields['Rhohv']['data'].mask=mask_from_rhohv
        radar.fields['Kdp']['data'].mask=mask_from_rhohv
        radar.fields['Psidp']['data'].mask=mask_from_rhohv
        radar.fields['RVel']['data'].mask=mask_from_rhohv

        ### We adjust the min/max value of the Doppler velocity colorbar depending on the radar operation mode 
        if 'sPowH' in Dataset(f).variables.keys():
            vdopmin=-11
            vdopmax=11
            dpp_or_fft = 'FFT'
        else:
            vdopmin=-41
            vdopmax=41
            dpp_or_fft = 'DDP'

        ### Make figure

        plt.rcParams['font.size']=15
        
        fig, axs = plt.subplots(2,3,figsize=(25,10))
        display = pyart.graph.RadarDisplay(radar)
        display.plot('Zh',vmin=-20,vmax=30,cmap='turbo',ax=axs[0,0])
        display.plot('Zdr',vmin=-.5,vmax=4,cmap='plasma',ax=axs[0,1])
        display.plot('Rhohv',vmin=0.6,vmax=1,cmap='YlOrRd_r',ax=axs[0,2])
        display.plot('RVel',vmin=vdopmin,vmax=vdopmax,cmap='bwr',ax=axs[1,0])
        display.plot('Kdp',vmin=-.5,vmax=2.5,cmap='plasma',ax=axs[1,1])
        display.plot('Psidp',vmin=-60,vmax=-20,cmap='viridis',ax=axs[1,2])

        for ax in axs.flatten():
            ax.grid()
            ax.set_ylim(0,5)
            ax.set_xlim(-16,0)
        plt.tight_layout()
        
        fig.savefig('/ltenas8/users/anneclaire/ICEGENESIS_2021/MXPol/RHI/ALL/'+name+'_VDop_'+dpp_or_fft,dpi=300,bbox_inches='tight',facecolor='w')
        
        plt.close()
        
    except Exception as e:
        print(f, e)
