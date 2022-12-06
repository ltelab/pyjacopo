# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:15:21 2017

@author: wolfensb
"""


import numpy as np
from functools import partial

from .constants import C, NOISE_EST_PER
from .algo.attenuation_correction import corr_att
from .algo.phidp_kdp import phidp_kdp_profile
from .algo.raw_data_processing import get_noise, power_spectra_parameters
from .config_parser.default_input import DATASET_DEFAULT
from .algo.rhohv_noise_correction import rhohv_noise_correction
from .pyart_wrapper import PyartMXPOL

def _worker_fft(input_pow, vel_array, num_incoh):
    # Worker for fft multiprocessing
    input_powh = input_pow[0]
    input_powv = input_pow[1]
 
    # H    
    power_var = power_spectra_parameters(input_powh, vel_array, num_incoh = \
                    num_incoh)
    sh  = power_var['power']
    vd  = power_var['m1_dop']
    sw  = power_var['m2_dop']
    nh  = power_var['noise_floor']

    # V
    power_var = power_spectra_parameters(input_powv,vel_array,num_incoh = \
                    num_incoh)
    sv  = power_var['power']
    nv  = power_var['noise_floor']
    return sh, vd, sw, nh, sv, nh
        
def _worker_fft2(input_pow, vel_array, vel_array2, num_incoh):
    # Worker for fft2 multiprocessing
    input_powh1 = input_pow[0]
    input_powh2 = input_pow[1]
    input_powv1 = input_pow[2]
    input_powv2 = input_pow[3]
    
    # H1
    power_var = power_spectra_parameters(input_powh1, vel_array, num_incoh = \
                    num_incoh)
    sh  = power_var['power']
    vd1  = power_var['m1_dop']
    sw  = power_var['m2_dop']
    nh  = power_var['noise_floor']

    # H2
    power_var = power_spectra_parameters(input_powh2, vel_array2, num_incoh = \
                    num_incoh)
    sh  = 0.5 * (sh + power_var['power'])
    vd2  = power_var['m1_dop']
    sw  = 0.5 * (sw + power_var['m2_dop'])
    nh  = 0.5 * (nh + power_var['noise_floor'])
    
    # V1
    power_var = power_spectra_parameters(input_powv1,vel_array,num_incoh = \
                    num_incoh)
    sv  = power_var['power']
    nv  = power_var['noise_floor']
    
    # V2
    power_var = power_spectra_parameters(input_powv2,vel_array2,num_incoh = \
                    num_incoh)
    sv  = 0.5 * (sv + power_var['power'])
    nv  = 0.5 * (nv + power_var['noise_floor'])
    
    return sh, vd1, vd2, sw, nh, sv, nh

def process_dataset(dataset_name, header, records, config):
    '''
    def process_dataset(dataset_name, header, records, config)

    PURPOSE:
        Processes a dataset given a header, a set of records and the user config
        according to the processing steps defined in the config

    INPUTS:
        dataset_name: name of the dataset type (PPI, RHI, PROFILE_NO_PEDESTAL, etc)
        header : file header of the raw data file
        records : set of records from the raw data file
        config: user configuration stucture

    OUTPUTS:
        proc_data: structure containing the computed variables for the given
                   dataset
    '''

    dname = dataset_name

    # Check if this dataset name is valid
    if dname not in DATASET_DEFAULT.keys():
        print('The specified dataset name is invalid, it must be one of '+\
               '/'.join(DATASET_DEFAULT.keys()))
        raise ValueError('Invalid dataset type')

    # Retrieve some parameters from the user configuration
    p = {} # Putting it into dict makes easier to identify
    p['V'] = config['location']['v_channel']
    p['H'] = config['location']['h_channel']
    p['rad_freq'] = config['location']['radar_frequency'] # [GHz]
    p['zcal'] = config['location']['radar_calibration_constant']
    p['czdr_apply'] = config['products']['datasets'][dname]['processing']['czdr_apply']
    p['discard_spectra'] = config['products']['datasets'][dname]['processing']['discard_spectra']
    p['discard_signal'] = config['products']['datasets'][dname]['processing']['discard_signal']
    p['mask_out'] = config['products']['datasets'][dname]['processing']['mask_out']
    p['snr_pow'] = config['products']['datasets'][dname]['processing']['snr_pow']
    p['phidp_kdp'] = config['products']['datasets'][dname]['processing']['phidp_kdp']
    p['att_corr'] = config['products']['datasets'][dname]['processing']['att_corr']
    p['rhohv_corr'] = config['products']['datasets'][dname]['processing']['rhohv_snr_correction']
    p['zdr_cal'] = config['location']['zdr_offset']
    p['fft2i_bug'] = config['location']['fft2i_bug']
    p['az_offset'] = config['location']['az_offset']
    p['el_offset'] = config['location']['el_offset']
    p['range_min'] = config['products']['datasets'][dname]['range_min']
    p['parallel'] = config['products']['parallel']

    # Add the radar frequency to phidp_kdp structure (some routines need it)
    p['phidp_kdp']['rad_freq'] = p['rad_freq']

    # Retrieve some parameters from raw data file
    pri1 = header['pri1']
    pri2 = header['pri2']
    rres = header['rres']
    nrg = header['nrg']
    rgs = header['rgs']
    zrg =  header['zrg']
    spw =  header['sumpower']
    fftl =  header['fftl']
    cave =  header['cave']
    pave =  header['postave']
    servmode = header['servmode']

    # Array indices for multi-dimensional variables (like pp, cc)
    if p['V'] == 0:
        v1 = 0; v2 = 1
        h1 = 2; h2 = 3
    else:
        v1 = 2; v2 = 3
        h1 = 0; h2 = 1


    # If parallel is enabled, start computation pool
    if p['parallel']:
        import multiprocessing as mp
        pool = mp.Pool(processes = mp.cpu_count())


    # Start processing
    #-----------------------------------------------------------------------------

    ts1 = pri1*1e-6 # [s] Pulse repetition intervals 1
    ts2 = pri2*1e-6 # [s] Pulse repetition intervals 1
    ts =  ts2 - ts1

    Lambda = C / (p['rad_freq'] * 1E9)  # wavelength [m]

    # Range in [m] 3.75 m is: 80 MHz Dig Rec sample rate dec by 2 to 40 MHz
    rrange = np.arange(nrg) * rgs - zrg * 3.75

    valid_range = np.where(rrange >= p['range_min'])[0] # indexes of valid range (not checked)
    rrange = rrange[valid_range]

    # Get azimuth/elevation (corrected)
    az_arr = np.array([rr['az'] for rr in records['header']]) - p['az_offset']
    az_arr[az_arr < 0] += 360
    el_arr = np.array([rr['el'] for rr in records['header']]) - p['el_offset']

    # Settings for FFT modes
    # Define a vector that will be used to center spectra around 0 m/s
    if servmode >= 2:
        correct_order = np.zeros(fftl,dtype = int)
        correct_order[int(fftl/2)-1:] = np.arange(fftl/2 + 1)
        correct_order[0:int(fftl/2-1)] = np.arange(fftl/2-1) + fftl/2+1
        p['correct_order'] = correct_order
    
    # Initialize output
    proc_data = {}
    
    
    if servmode == 0: # Pulse-pair mode (PP)
        # Check sumpower
        if spw != 1:
            print('Processing of PP and DPP data without sumpower is not implemented')

        # Nyquist
        va = Lambda/(4 * ts1) # [m/s]
        proc_data['Va'] = va

        # Power and correlation
        power_v = np.array([r['power'][p['V']][valid_range] for r in records['data']])
        power_h = np.array([r['power'][p['H']][valid_range] for r in records['data']])
        cc = np.array([r['cc'][valid_range] for r in records['data']])

        # Noise level
        nel_powtonoise = int(power_h.size*NOISE_EST_PER)

        noise_h = get_noise(np.sort(power_h.ravel())[0:nel_powtonoise], 
                            cave * pave)
        noise_v = get_noise(np.sort(power_v.ravel())[0:nel_powtonoise], 
                            cave * pave)
        
        # Calculating signal to noise ratio SNR
        signal_h = power_h - noise_h
        signal_v = power_v - noise_v
        snr_h = signal_h / noise_h # SNR [-]
        snr_v = signal_v / noise_v # SNR [-]

        # Calculating velocity from pulse pair

        pp_v = np.array([r['pp'][p['V']][valid_range] for r in records['data']]) 
        pp_h = np.array([r['pp'][p['H']][valid_range] for r in records['data']]) 

        vel_v = - Lambda/(4*np.pi*ts1) * np.arctan(pp_v)
        vel_h = - Lambda/(4*np.pi*ts1) * np.arctan(pp_h)
        vel_dop = - Lambda/(4*np.pi*ts1) * np.arctan(pp_h + pp_v)

        # Doppler spectral (Doviak, Zrnic) width [m/s]
        arg_h = np.log(np.abs(signal_h / pp_h))
        sw_h = Lambda / (2 * np.pi * ts1 *np.sqrt(2)) * np.abs(arg_h)**0.5

        arg_v = np.log(np.abs(signal_v / pp_v))
        sw_v = Lambda / (2 * np.pi * ts1 *np.sqrt(2)) * np.abs(arg_v)**0.5

        arg_dop = np.log(np.abs((signal_h + signal_v) / (pp_h + pp_v)))
        sw_dop = Lambda / (2 * np.pi * ts1 *np.sqrt(2)) * np.abs(arg_dop)**0.5

        # Rhohv
        rhohv = np.abs(cc)/np.sqrt(power_h * power_v)

        # Psidp raw (radians)
        if p['V'] == 1:
            psidp_raw = np.angle(cc, deg = True)
        else:
            psidp_raw = np.angle(np.conjugate(cc), deg = True)

    elif servmode == 1: # Dual-pulse-pair mode (DPP)
        # Check sumpower
        if spw != 1:
            print('Processing of PP and DPP data without sumpower is not implemented')

        # Nyquist
        va = Lambda/(4 * (ts2 - ts1)) # [m/s]
        proc_data['Va'] = va

        # Power and correlation
        power_v = np.array([r['power'][p['V']][valid_range] for r in records['data']])
        power_h = np.array([r['power'][p['H']][valid_range] for r in records['data']])
        cc = np.array([r['cc'][valid_range] for r in records['data']])


        # Noise level
        nel_powtonoise = int(power_h.size*NOISE_EST_PER)

        noise_h = get_noise(np.sort(power_h.ravel())[0:nel_powtonoise], 
                            cave*pave)
        noise_v = get_noise(np.sort(power_v.ravel())[0:nel_powtonoise], 
                            cave*pave)

        # Calculating signal to noise ratio SNR
        signal_h = power_h - noise_h
        signal_v = power_v - noise_v
        snr_h = signal_h / noise_h # SNR [-]
        snr_v = signal_v / noise_v # SNR [-]

        # Calculating velocity from pulse pair
        pp_v1 = np.array([r['pp'][v1][valid_range] for r in records['data']]) 
        pp_v2 = np.array([r['pp'][v2][valid_range] for r in records['data']])

        pp_h1 = np.array([r['pp'][h1][valid_range] for r in records['data']]) 
        pp_h2 = np.array([r['pp'][h2][valid_range] for r in records['data']]) 

        vel_v = Lambda / (4*np.pi*ts)*np.angle(pp_v1/pp_v2)
        vel_h = Lambda / (4*np.pi*ts)*np.angle(pp_h1/pp_h2)
        vel_dop = Lambda / (4*np.pi*ts)*np.angle(pp_v1/pp_v2+pp_h1/pp_h2)

        # Doppler spectral (Doviak, Zrnic) width [m/s], TODO check other methods
        arg_h = np.log(np.abs(pp_h1/pp_h2))
        sw_h = Lambda / (2 * np.pi * np.sqrt(2*(ts2**2 - ts1**2))) * np.abs(arg_h)**0.5

        arg_v = np.log(np.abs(pp_v1/pp_v2))
        sw_v = Lambda / (2 * np.pi * np.sqrt(2*(ts2**2 - ts1**2))) * np.abs(arg_v)**0.5

        arg_dop = np.log(np.abs((pp_h1 + pp_v1)/(pp_h2 + pp_v2)))
        sw_dop = Lambda / (2 * np.pi * np.sqrt(2*(ts2**2 - ts1**2))) * np.abs(arg_dop)**0.5

        # Rhohv
        rhohv = np.abs(cc) / np.sqrt(power_h * power_v)

        # Psidp raw (radians)
        if p['V'] == 1:
            psidp_raw = np.angle(cc, deg = True)
        else:
            psidp_raw = np.angle(np.conjugate(cc), deg = True)

    elif servmode == 2: # Fast Fourier Transform (FFT)
        # Nyquist
        ts = ts1
        va = Lambda / (4*ts)

        # Power and signal, cut the range and order with 0 m/s in the center
        spow_v = np.array([r['power-spectra'][p['V']][valid_range][:,p['correct_order']] 
                for r in records['data']])
        spow_h = np.array([r['power-spectra'][p['H']][valid_range][:,p['correct_order']] 
                for r in records['data']])

        scc = np.array([r['cross-spectra'][valid_range][:,p['correct_order']] 
                for r in records['data']])

        # Velocity array (centered on 0)
        vel_array = ((1. + np.arange(fftl)) - fftl/2.) / fftl*(2.*va)

        # Initialize outputs
        size_d = spow_h.shape
        signal_h = np.zeros((size_d[0],size_d[1])) + np.nan
        signal_v = np.zeros((size_d[0],size_d[1])) + np.nan
        noise_h = signal_h.copy(); noise_v = signal_v.copy()
        vel_dop = signal_h.copy();
        sw_dop = signal_h.copy()

        # Get signal from H and V spectra -- TODO add vectorization but I expect
        # it to be quite tricky...                
        worker = partial(_worker_fft, vel_array = vel_array, 
                         num_incoh= cave * pave)
        if p['parallel']:
            out = list(pool.map(worker, zip(spow_h, spow_v)))
        else:
            out = list(map(worker, zip(spow_h, spow_v)))
        
        signal_h = np.array([o[0] for o in out])
        vel_dop = np.array([o[1] for o in out])
        sw_dop = np.array([o[2] for o in out])
        noise_h = np.array([o[3] for o in out])
        signal_v = np.array([o[4] for o in out])
        noise_v = np.array([o[5] for o in out])
        
        
        # Average the Noise for this level of data and SNR
        snr_v = signal_v/noise_v
        snr_h = signal_h/noise_h  # SNR [-]
        noise_h = np.mean(noise_h)
        noise_v = np.mean(noise_v)

        # Rhohv
        rhohv = np.abs(np.nansum(scc,axis=2)) / np.sqrt(np.nansum(spow_h, axis = 2) *
                       np.nansum(spow_v, axis = 2))        


        # Psidp Raw (radians)
        if p['V'] == 1:
            psidp_raw = np.angle(np.sum(scc,axis = 2), deg = True)
        else:
            psidp_raw = np.angle(np.sum(np.conjugate(scc),axis = 2), deg = True)

        # Add variables to output structure
        proc_data['Va'] = va
        if not p['discard_spectra']:
            proc_data['sVel'] = vel_array
            proc_data['sCC'] = np.abs(scc)
            proc_data['sPowH'] = spow_h
            proc_data['sPowV'] = spow_v

    elif servmode in [3,4]: # FFT2 or FFT2I
        # TODO, never worked in FFT2 but it seems the same as FFT2I

        va1 = p['Lambda'] / (4* ts1) # Nyquist for the first train
        va2 = p['Lambda']  / (4 * ts2) # Nyquist for the scond train
        va = p['Lambda']  / (4 * (ts2 -  ts1)) # Overall Nyquist

        # This mode was buggy for some campaigns (for MXPol, before 2014 Jun)
        if p['fft2i_bug']:
            first_p = 0; second_p = 1
        else:
            first_p = 1; second_p = 0
            va1_s = va1; va2_s = va2; va2 = va1_s; va1 = va2_s

        # Define velocities
        vel_array1 = ((1. + np.arange(fftl)) - fftl/2.) / fftl * (2.*va1)
        vel_array2 = vel_array1 * (va2/va1)
        # Global vel. array
        vel_array = ((1. + np.arange(fftl)) - fftl/2.) / fftl * (2.*va)

        # ----------------------------------
        # Get spectra trains (cut in range and centered on 0)
        
        # Dimensions are nangles x nranges x fftl
        if p['fft2i_bug']:
            
            spow_h1 = np.array([r['power-spectra'][h2][valid_range][:,p['correct_order']] 
                for r in records['data']])
            spow_h2 = np.array([r['power-spectra'][h1][valid_range][:,p['correct_order']] 
                for r in records['data']])
        else:
            spow_h1 = np.array([r['power-spectra'][h1][valid_range][:,p['correct_order']] 
                for r in records['data']])
            spow_h2 = np.array([r['power-spectra'][h2][valid_range][:,p['correct_order']] 
                for r in records['data']])

        if p['fft2i_bug']:
            spow_v1 = np.array([r['power-spectra'][v2][valid_range][:,p['correct_order']] 
                for r in records['data']])
            spow_v2 = np.array([r['power-spectra'][v1][valid_range][:,p['correct_order']] 
                for r in records['data']])
        else:
            spow_v1 = np.array([r['power-spectra'][v1][valid_range][:,p['correct_order']] 
                for r in records['data']])
            spow_v2 = np.array([r['power-spectra'][v2][valid_range][:,p['correct_order']] 
                for r in records['data']])

        scc1 = np.array([r['cross-spectra'][first_p][valid_range][:,p['correct_order']] 
                for r in records['data']])
        scc2 = np.array([r['cross-spectra'][second_p][valid_range][:,p['correct_order']] 
                for r in records['data']])    
    
        
        # Initialize outputs
        size_d = spow_h1.shape
        signal_h = np.zeros((size_d[0],size_d[1])) + np.nan
        signal_v = np.zeros((size_d[0],size_d[1])) + np.nan
        noise_h = signal_h.copy(); noise_v = signal_v.copy()
        vel_dop1 = signal_h.copy(); vel_dop2 = signal_h.copy();
        sw_dop = signal_h.copy()

        # Get signal from H and V spectra -- TODO add vectorization but I expect
        # it to be quite tricky...
        worker = partial(_worker_fft, vel_array1 = vel_array1, vel_array2 = vel_array2,
                         num_incoh= cave*pave)
        if p['parallel']:
            out = list(pool.map(worker, zip(spow_h1, spow_h2,
                                            spow_v1, spow_v2)))
        else:
            out = list(map(worker, zip(spow_h, spow_v)))
        
        signal_h = np.array([o[0] for o in out])
        vel_dop1 = np.array([o[1] for o in out])
        vel_dop2 = np.array([o[2] for o in out])
        sw_dop = np.array([o[3] for o in out])
        noise_h = np.array([o[4] for o in out])
        signal_v = np.array([o[5] for o in out])
        noise_v = np.array([o[6] for o in out]) 
    
        # Average the Noise for this level of data and SNR

        snr_v = signal_v / noise_v
        snr_h = signal_h / noise_h
        noise_h = np.nanmean(noise_h)
        noise_v = np.nanmean(noise_v)

        # Get de-aliased Dopler velocity, in analogy  with respect
        # to DPP mode. (See Doviak and Zrnic book)

        vel_dop = (p['Lambda'] / (4*np.pi * (ts2 - ts1))) * \
            np.angle(np.exp((-1j * 4 * np.pi) * (vel_dop1 * ts1 \
                            - vel_dop2 * ts2)/p['Lambda']))

        # Average powers
        spow_h = 0.5 * (spow_h1 + spow_h2)
        spow_v = 0.5 * (spow_v1 + spow_v2)

        # Cross-correlation
        scc = 0.5 * (scc1 + scc2)
        cc = np.nansum(scc, axis=2)

        # Rhohv
        rhohv = np.abs(cc) / np.sqrt(np.nansum(spow_h, axis = 2) *
                       np.nansum(spow_v, axis = 2))


        
        # Psidp Raw (radians)
        if p['V'] == 1:
            psidp_raw = np.angle(cc, deg = True)
        else:
            psidp_raw = np.angle(np.conjugate(cc), deg = True)

        proc_data['Va'] = va


        if not p['discard_spectra']:
            proc_data['sVel'] = vel_array
            proc_data['sCC'] = np.abs(scc)
            proc_data['sPowH'] = spow_h
            proc_data['sPowV'] = spow_v

    if p['parallel']:
        pool.close()
        pool.join()
        
    # ZH and ZV
    zh = np.zeros((len(az_arr),len(rrange))) + np.nan
    zv = zh.copy()
    condi = signal_v > 0
    icondi = signal_v <= 0

    rrange_mat = np.tile(rrange, (len(az_arr),1))
    if len(condi):
        zv[condi] = 10 *np.log10(signal_v[condi]) - 10 * np.log10(rres) \
                    + 20 * np.log10(rrange_mat[condi]) + p['zcal'] # dBZ
        snr_v[condi] = 10 * np.log10(snr_v[condi])
        if len(icondi):
            snr_v[icondi] = np.nan

    condi = signal_h > 0
    icondi = signal_h <= 0

    if len(condi):
        zh[condi] = 10 *np.log10(signal_h[condi]) - 10 * np.log10(rres) + \
                    20 * np.log10(rrange_mat[condi]) + p['zcal'] # dBZ
        snr_h[condi] = 10 * np.log10(snr_h[condi])
        if len(icondi):
            snr_h[icondi] = np.nan

    # ZDR
    zdr = zh - zv # dB

    # -------------------------------------------------------------------------
    # Apply ZDR calibration constant. Default is  0.
    if p['czdr_apply']:
        zdr -=  p['zdr_cal']

    # -------------------------------------------------------------------------
    # Rhohv noise correction. Default is  0.
    if p['rhohv_corr']:
        rhohv = rhohv_noise_correction(rhohv, snr_h, zdr, noise_h, noise_v)

#    # PHIDP - KDP processing, if required
#    if type(p['phidp_kdp']) == dict:
#        out = phidp_kdp_profile(rres,psidp_raw,p['phidp_kdp'])
#        proc_data['Phidp'] = out['phidp']
#        proc_data['Kdp'] = out['kdp']
#    else:
#        proc_data['Phidp'] = psidp_raw


#    # Attenuation correction (if needed)
#    if type(p['att_corr']) == dict:
#        out = corr_att(rres, zh, zdr, proc_data['Phidp'], p['att_corr'])
#        proc_data['ZhCorr'] = out['zh_corr']
#        if 'zdr_corr' in out.keys():
#            proc_data['ZdrCorr'] = out['zdr_corr']
    
    # Mask output according to SNR and Rhohv (TODO: add possibilities to
    # mask on other var.) (Nan  and 1, binary mask)
    if type(p['mask_out']) == dict:
        snr_threshold = p['mask_out']['snr']
        rhohv_threshold = p['mask_out']['rhohv']
        mask_data = np.logical_and(snr_h > snr_threshold, rhohv > rhohv_threshold)
    else:
        # mask_out undefined
        mask_data = (snr_h > p['snr_pow'])

    # Create output structure
    proc_data['Zh'] = zh
    proc_data['Zv'] = zv
    proc_data['Zdr'] = zdr
    proc_data['RVel'] = vel_dop
    proc_data['Sw'] = sw_dop
    proc_data['Psidp'] = psidp_raw
    proc_data['Noise_h'] = noise_h
    proc_data['Noise_v'] = noise_v
    
    proc_data['SNRh'] = snr_h
    proc_data['SNRv'] = snr_v
    proc_data['Mask_data']= mask_data
    proc_data['Rhohv'] = rhohv
    
    if not p['discard_signal']:
        proc_data['Signal_h'] = signal_h
        proc_data['Signal_v'] = signal_v
        
    proc_data['Time'] = np.array([rr['gps_ts1'] for rr in records['header']])
    proc_data['Azimuth'] = az_arr
    proc_data['Elevation'] = el_arr
    proc_data['Range'] = rrange
    

    return PyartMXPOL(dataset_name, proc_data, config, header)

    
if __name__ == '__main__':
    import pyart
    from pyjacopo.config_parser import parse,CONFIG_DEFAULT,CONFIG_VALID
    from pyjacopo.raw_reader import read_raw_data
    
    f = '/ltedata/Test_MXPol_EPFL_2019/Raw_data/2019/10/01/XPOL-20191001-175738.dat'

    config = parse('config.yml',CONFIG_DEFAULT,CONFIG_VALID)
    header, records = read_raw_data(f)
    
    p = process_dataset('RHI',header,records['RHI'],config)
    pyart.io.write_cfradial('test.nc',p)
