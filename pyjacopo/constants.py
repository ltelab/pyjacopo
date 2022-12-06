# -*- coding: utf-8 -*-

# Physical and numerical constants
C = 299792458. # Speed of light [m/s]
NOISE_EST_PER = 0.05 # *100 (%) of lowest elements of power used in noise estimation


# Define here some defaults for PyART
# MXPOL variable names (need to be in agreement with the names in process_dataset)
VARNAMES = ['Zh','Zdr','Kdp','Phidp','Rhohv','ZhCorr','ZdrCorr','RVel','Sw',
            'SNRh','SNRv','Psidp','RVel','sCC','sPowH','sPowV','sVel', 'Signal_h', 
            'Signal_v']

# The corresponding labels in the PyART class
LABELS = ['Reflectivity','Diff. reflectivity','Spec. diff. phase','Diff. phase','Copolar corr. coeff','Att. corr reflectivity',\
        'Att corr. diff. reflectivity.','Mean doppler velocity','Spectral Width','SNR at hor. pol.','SNR at vert. pol.','Total diff. phase',
        'Mean doppler velocity', 'Spectral correlation','Spectral hor. power',
        'Spectral vert. power','Velocity bins','Signal at hor. pol.','Signal at vert. pol.']

# The corresponding units in the PyART class
UNITS = ['dBZ','dB','deg/km','deg','-','dBZ','dB','m/s','m/s','-','-','deg','m/s',
         '-','mW','mW','m/s','mW','mW']

# The corresponding default min. limit in the PyArt class
VMIN = [0.,0.,0.,0.,0.6,0.,0.,-15.,0.,0.,0.,0.,-15.,0,-60,-60,-40,-60,-60]
# The corresponding default max. limit in the PyArt class
VMAX = [55.,3.,4.,45.,1.,55.,3.,15.,3.,20.,20.,45.,15.,1,30,30,40,20,20]


# Name of servmodes
SERVMODE_NAMES = {0 : 'Pulse Pair', 1 : 'Staggered - Double Pulse Pair',
                  2 : 'FFT', 3: 'Double FFT', 4 : 'Staggered - Double FFT'}

# Global attributes
GPS_COORDSYSTEM = 'WGS84'
LATITUDE_UNIT = 'DegreesNorth'
LONGITUDE_UNIT = 'DegreesEast'
ALTITUDE_UNIT = 'MetersAboveSeaLevel'
TIME_ASSIGN_SCAN_UNIT = 'Seconds since 01-01-1970'
RADAR_PARAMETERS = 'Radar NyquistVelocity AzOffset ElOffset RadarFreq BW3dB'
NYQUIST_VELOCITY_UNIT = 'MetersPerSecond'
AZ_OFFSET_UNIT = 'Degrees'
AZ_OFFSET_DESCRIPTION = 'Az. Offset already substracted'
RADAR_FREQ_UNIT = 'GigaHertz'
BW3DB_UNIT = 'Degrees'
BW3DB_DESCRIPTION = 'Theoretical 3dB angular beamwidth'
FILTER_BW_UNITS = 'MegaHertz'
PRI_UNIT = 'MicroSeconds'
PULSEWIDTH_UNIT = 'Microseconds'
RANGE_RESOLUTION_UNIT = 'Meters'
RANGE_GATE_SPACING_UNIT = 'Meters'
RANGE_TO_FIRST_GATE_UNIT = 'Meters'
NUM_RANGES_UNIT = 'dimensionless'
NOISE_HV_EXPLANATION = 'Estimated noise in H(V) channels'
NOISE_H_UNIT = 'Milliwatts'
NOISE_V_UNIT = 'Milliwatts'
AZIMUTH_UNIT = 'Degrees'
ELEVATION_UNIT = 'Degreees'
FFT_LENGTH_UNIT = 'dimensionless'
MISSING_DATA_EXPLANATION = 'NaNs, censored data, missing data'
MISSING_PARAM_EXPLANATION = 'Missing NetCDF parameters/attributes'
CAVE_UNIT = 'dimensionless'
POST_AVE_UNIT = 'dimensionless'
CLUTTER_FILTER_WIDTH_UNIT = 'MetersPerSecond'
SOURCE = 'X-band radar data processed at LTE-EPFL'
INSTITUTION = 'Laboratoire de Teledetection Environnemental - Ecore Polytechnique Federale de Lausanne'
CONTACT_INFORMATION = 'http://lte.epfl.ch'
ZCAL_DB_NOTES = 'The original manufacturer calibration constant was 7.56 dBZ. The value used here derives from comparison with disdrometers during the HyMeX campaign'
FILTER_BW_UNIT = 'MegaHertz'
