# -*- coding: utf-8 -*-
"""
Creation: 3 February 2017, D. Wolfensberger
"""

# Variable types and their sizes (see https://docs.python.org/3/library/struct.html)
# Note that I added the complex type 'cx'
BYTE_SIZES = {'i':4,'f':4,'s':1,'B':1,'I':4,'l':4,'d':8, 'cx':8}

# Map indexes to scan type names
SCAN_TYPES = {0:'PROFILE',3:'Point',5:'PPI',6:'RHI',7:'Az Ras', 8:' El Ras', 9: 'Vol'}

###############################################################################
### HEADER INFO

HEADER = {}

# Names of header entries
HEADER['names'] = ['site','az_off','rcb','cfw','cave','drate','fftl','fftwindow',
                  'res1','nrecords','nscans','recsize','rectime','bitflags',
                  'res2','bandwidth','freqthres','freqadj','res3','pri3','hdbz',
                  'hnoise','inttime','freqerror','freq','rmax','nrg','nps','postdec',
                  'postave','pri1','pri2','prit','cicdec','pl','res5','rgs','res6',
                  'rres','recfftmoments','recraw','recenable','servmode','res7','servstate',
                  'res8','softdec','sumpower','ave','txdel','txdelpwmult','txcenter',
                  'txcenteroffset','txswdel','txswholdoff','cfon','vdbz',
                  'vnoisedbm','zrg','scan_type','host_ts','spare']

# Header entries variable types (see https://docs.python.org/3/library/struct.html)
HEADER['type'] = ['s','d','l','d','l','d','l','l','l','l','l','l','l','l','l',
                 'd','d','l','l','l','d','d','d','d','d','d','l','l','l','l',
                 'l','l','l','l','d','l','d','l','d','l','l','l','l','l','l',
                 'l','l','l','l','l','l','l','l','l','l','l','d','d','d','l',
                 'l','l']

# Length of header entries in multiples of type byte sizes. ex if type = 'd'and
# len = 3, the total size in bytes will be 8 (bytes/double) * 3 = 18 bytes

HEADER['len'] = [1024,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,4]

###############################################################################
### RECORD HEADER INFO

RECORD_HEADER = {}
RECORD_HEADER['names'] = ['host_ts','temp (RF)','temp (plate)',
    'temp (pod air)','temp (PC cover)','incl (roll)','incl (pitch)','fuel',
    'cpu_temp','scan_type','tx_pow_sam','spare1','spare2','spare3','spare4',
    'spare5','az','el','az_vel','el_vel','lat_hem','lat,','lon_hem','lon',
    'heading_ref','heading_deg','speed','gga','gps_ts1','gps_td2','data_type',
    'data_size']

RECORD_HEADER['type'] = ['l','l','l','l','l','l','l','l','l','l','f','l',
                        'l','l','l','l','d','d','d','d','l','d','l','d','l','d',
                        'd','s','l','l','l','l',]

RECORD_HEADER['len'] = [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,96,1,1,1,1]


###############################################################################
### DATA INFO

# Data content depends on the operation mode (PP, DPP,FFT,FFT2,FFT2I)

RECORD_DATA = {'PP':{},'PPSP':{},'DPP':{},'DPPSP':{},'FFT':{},'FFT2':{}}

# Note that complex numbers are first read as double and then put in a complex
# array at runtime

# Pulse-Pair (PP)
RECORD_DATA['PP']['names'] = ['power','pp','cc']
RECORD_DATA['PP']['type'] =  ['f','cx','cx']
RECORD_DATA['PP']['len'] = [[4,'nrg'],[2,'nrg'],[1,'nrg']]

# Pulse-Pair-Summed-Power (PP with sumpower = 1)
RECORD_DATA['PPSP']['names'] = ['power','pp','cc']
RECORD_DATA['PPSP']['type'] =  ['f','cx','cx']
RECORD_DATA['PPSP']['len'] = [[2,'nrg'],[2,'nrg'],[1,'nrg']]

# Dual-Pulse-Pair (DPP)
RECORD_DATA['DPP']['names'] = ['power','pp','cc']
RECORD_DATA['DPP']['type'] =  ['f','c','cx']
RECORD_DATA['DPP']['len'] = [[6,'nrg'],[4,'nrg'],[1,'nrg']]

# Dual-Pulse-Pair-Summed-Power (DPP with sumpower = 1)
RECORD_DATA['DPPSP']['names'] = ['power','pp','cc']
RECORD_DATA['DPPSP']['type'] =  ['f','cx','cx']
RECORD_DATA['DPPSP']['len'] = [[2,'nrg'],[4,'nrg'],[1,'nrg']]

# FFT
RECORD_DATA['FFT']['names'] = ['power-spectra','cross-spectra']
RECORD_DATA['FFT']['type'] =  ['f','cx']
RECORD_DATA['FFT']['len'] = [[2,'nrg','fftl'],[1,'nrg','fftl']]

# FFT2 or FFT2I
RECORD_DATA['FFT2']['names'] = ['power-spectra','cross-spectra']
RECORD_DATA['FFT2']['type'] =  ['f','cx']
RECORD_DATA['FFT2']['len'] = [[4,'nrg','fftl'],[2,'nrg','fftl']]
RECORD_DATA['FFT2I'] = RECORD_DATA['FFT2']
