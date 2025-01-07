from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor']='white'
import datetime


# Path to the netCDF file
f = '/ltenas8/users/anneclaire/ICEGENESIS_2021/MXPol/Proc_data_v0_clutter_filter_no_thres_PPI/XPOL-20210129-010308_FFT_PPI.nc'

# Read the netCDF file and load the data
nc = Dataset(f)
r = nc.variables['range'][:]
spec_h_lin = nc.variables['sPowH'][:]*r[None,:,None]**2 # reflectivity spectra in linear units, without calibration constant
spec_H_dB = 10*np.log10(spec_h_lin)-nc.ZCal_dB # reflectivity spectra in dBZ, corrected with calibration constant
svel = nc.variables['sVel'][:] # array of spectral Doppler velocity


# Plotting a spectrogram
i_plot = 0 # index of the sweep to plot
plt.rcParams['font.size'] = 15
plt.figure(figsize=(6,8))
dv = svel[1]-svel[0]
im=plt.pcolormesh(svel,r/1e3,spec_H_dB[i_plot],vmin=-35,vmax=20,cmap='turbo')
plt.ylim(0.100,8.8)
plt.colorbar(im,pad=.03,aspect=30,label='Spectral Ze [dBsZ]')
plt.xlabel('Doppler velocity [m s$^{-1}$]')
plt.ylabel('Range [km]')
plt.xlim(-10,2)
# plt.gcf().savefig('MXPol_spectrogram_rain.png',dpi=300,bbox_inches='tight',facecolor='white')


# Plotting reflectivity time series from the PPI
dt = [datetime.datetime.fromtimestamp(tt) for tt in nc.variables['time'][:]]
plt.pcolormesh(dt,nc.variables['range'][:],nc.variables['Zh'][:].T,vmax=30,cmap='turbo')
plt.gcf().autofmt_xdate()
plt.ylim(0,4500)
plt.grid()