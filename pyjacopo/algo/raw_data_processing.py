#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:37:45 2017

@author: wolfensb
"""

import numpy as np

def get_noise(power,fft_avg, calc_stdv = False, mrr = False):

    '''
        PURPOSE
            Function to get the noise level

            using the method given by Hildebrand.
            Originally developed for spectral processes but can be used also to estimate
            integrated noise-floor in pulse pair modes
            
            Refers to Hildebrand method of Noise calculation.
            "Objective determination of noise level..."
            PH Hildebrand, 1974
            
            Code is adapted from https://github.com/ARM-DOE/pyart/blob/master/
            pyart/util/hildebrand_sekhon.py#L17


        INPUTS
            power : array of measured powers with 1 line and "n" columns...
            fft_avg : number of non-coherent averages
            calc_stdv : if true the stdev of the noise estimates will be returned as well
            mrr : if true, the noise estimation described in the MRR hardware will
                   be used

        OUTPUTS
            lnoise : estimated noise level
            stdev : stdev of noise estimates (only if calc_stdv == TRUE)

    '''

    
    if not mrr:
        sorted_spectrum = np.sort(power)
        npts_min = int(len(power) / 10)

        nnoise = len(power)  # default to all points in the spectrum as noise
        for npts in range(1, len(sorted_spectrum)+1):
            partial = sorted_spectrum[:npts]
            mean = np.mean(partial)
            var = np.var(partial)
            if var * fft_avg <= mean**2.:
                nnoise = npts
            else:
                # partial spectrum no longer has characteristics of white noise
                break
        if nnoise < npts_min:
            nnoise = npts_min
        noise_spectrum = sorted_spectrum[0:nnoise]
        lnoise = np.mean(noise_spectrum)
        var = np.var(noise_spectrum)
    
    else:
        power_trunc = power
        p = 1 # Nb of averages

        power_sort = np.sort(power_trunc)[::-1]

        ratio = (p*np.var(power_sort))/ np.mean(power_sort)**2
        lnoise = np.mean(power_sort)
        counter = 0
        num_pts = len(power_sort.ravel())
        num_pts_new = num_pts

        while ratio > (1./fft_avg) and num_pts_new > 10:
            counter += 1
            num_pts_new = num_pts - counter

            # Remove the highest value
            ratio = np.var(power_sort[counter:])/np.mean(power_sort[counter:])**2

            lnoise = np.mean(power_sort[counter:])

    if calc_stdv and not mrr:
        return lnoise, stdv
    else:
        return lnoise



def get_noise_vectorized(power,fft_avg):
    
    nnoise = len(power[0])  # default to all points in t                                      he spectrum as noise
    sorted_spectrum = np.sort(power, axis=1)
    npts_min = int(len(power[0]) / 10)

    lnoise = np.zeros((len(power))) + np.nan
    
    nsamples = np.arange(len(power[0]))+1
    
    # Compute partial averages and variances
    mean_rolling = np.cumsum(sorted_spectrum, axis=1)/nsamples 
    mean2_rolling = np.cumsum(sorted_spectrum**2, axis=1)/nsamples
    var_rolling = mean2_rolling - mean_rolling**2
    
    condi = var_rolling * fft_avg <= mean_rolling**2.
        
    # Get occurence of first non white noise gate
    first_notwn = np.argmin(condi, axis=1) - 1
    first_notwn[~np.any(condi == 0)] = 0 

    condi_npts = first_notwn < npts_min
    
    
    lnoise = mean_rolling[np.arange(len(first_notwn)),first_notwn]
    lnoise[condi_npts] = mean_rolling[condi_npts,npts_min-1]

    return lnoise
    
def power_spectra_parameters(spec_in, vel_array, notch = True, num_incoh = 1,
                             h_for_noise = 32, n_min_stat = 5):
    '''
    PURPOSE: calculate moments from power spectra of dimension [height,FFTbins]
        Moments can be calculated either by notching the
        signal around its peak value
        (appropriate for non-aliased spectra), or by getting all
        the points that exceed the noise floor. Exception to this
        rule is the moment 0 (power), that is calculated on all
        the points exceeding the noise floor.
        In any case this routine may fail if phyisically relevant
        bimodalities are separated by  bins that are below the noise floor.

        A clarification about the "moments". M0 is the total power (e.g. mw),
        M1 is the mean doppler velocity (e.g. m/s), M2 is the standard
        deviation and keeps the same units as M1.
        M3 and M4 are normalized on the standard
        deviation (to power 3, and 4 respectively) and therefore
        are unitless.
        Folded or especially partially folded spectra may lead to
        biased estimations of the moments.

    AUTHOR(S):
        Jacopo Grazioli: Initially adapted from a routine of Danny
        Scipion, but fully restructured in this version.
        Daniel Wolfensberger: Python version

    INPUTS:
        spec_in  : bi-dimensional spectrum of size [height bin, FFT bin]
            it contains the power spectral density (power spectrum or
            spectrogram.  Usually its units are [mW/bin] or [W/bin], but it is
            not mandatory. It accepts also digital units.
        vel_array: vector containing the X-values of the spectrogram (same
             for each height level). The units are not important (Hz, m/s)
             but the results will depend on the input units.
        notch  : boolean yes/no. Default:1 (yes). Notch the spectra
                 before calculating the moments >= 1
        num_incoh  : [-] number of averaging cycles used to produce the spectra.
                 Default is 1. Used for noise estimation.
        h_fornoise : [-] how many height levels (the most far ones)
                 should be used to estimate a first guess of the noise floor.
                 Default=32.
        n_min_stat : [-] minimal number of power measurements to be used to
                     compute the spectrum moments, default = 5

    '''

    # Get information about inputs
    sz  = spec_in.shape
    n_fft = len(vel_array)
    if sz[1] != n_fft:
        raise ValueError('Dimension mismatch between spec_in and vel_array')

    num_hei = sz[0] # Number of heights
    # Get noise level estimate
    noise = get_noise(spec_in[-h_for_noise:,].ravel(),num_incoh) # Noise floor first guess at
    # longer ranges

    noise = max([noise, 1E-20])

    # Allocate output
    spec_out_pow = spec_in + np.nan
    power = np.zeros(num_hei) + np.nan # [mW] usually

    # Moments
    m1_dop = power.copy(); m2_dop = power.copy()
    m3_dop = power.copy(); m4_dop = power.copy() # [m/s]

    # Noise-level: power-density of the noise
    # Noise floor: noise level integrated over the area where the signal is
    noise_level = m1_dop.copy()

    # A vectorized version of noise estimation is now available
    noise_level = get_noise_vectorized(spec_in, num_incoh)
    noise_level[noise_level < noise] = noise
    
    # I noticed that Jacopo does not seem to subtract the noise from the power
    # at least on old FFT scans from Payerne
    spec_out_pow = spec_in  - noise_level[:,None]
    spec_out_pow[spec_out_pow < 0] = 0. # Noise-removed power

    # Get total power and total noise-floor
    ncondi = np.sum(spec_out_pow == 0., axis = 1)
    power = np.sum(spec_out_pow, axis = 1)
    noise_floor = ncondi * noise_level

    # Index of signal max
    max_pos = np.argmax(spec_out_pow, axis = 1)

    spec_in = spec_out_pow.copy()
    # Set the area of the spectrum where the signal is, in order to calculate its moments
    if notch:
        # Find the leftermost elements at the left of the max which
        # are larger than 0 : ll and the rightmost elements at the right of the max
        # which are larger than 0 : rr

        idx = np.tile(np.arange(n_fft),(num_hei,1))

        rr = np.argmax((spec_out_pow <= 0) * (idx > max_pos[:, None]), axis=1)
        rr[rr == 0] = n_fft # rr = 0 happens when there is no match
        ll = n_fft - np.argmax(((spec_out_pow <= 0) *
                                (idx < max_pos[:, None]))[:,::-1], axis=1)
        ll[ll == n_fft] = 0 # rr = 0 happens when there is no match

        # Isolate around the peak and calculate the moments
        nelem_peak = rr - ll + 1
        
        invalid = np.logical_or(idx < ll[:,None],idx > rr[:,None])

        spec_in[invalid] = 0
        
        # If not enough points, assign zero to whole spectrum
        spec_in[nelem_peak<n_min_stat, :] = 0
        
        vel_in = np.tile(vel_array,(num_hei,1))
        vel_in[invalid] = 0

    else: # No notch
        vel_in = vel_array

    pwr = np.sum(spec_in, axis = 1) # M0
    weights = spec_in / pwr[:,None]

    with np.errstate(divide='ignore'): # Ignore divide by zero warnings
        # M1 [m/s]
        m1_dop = np.sum(vel_in * weights, axis=1)

        # M2 [m/s] sigma
        m2_dop = np.sqrt(np.sum(weights * (vel_in - m1_dop[:,None])**2,axis=1))
        # M3 [-] skewness
        m3_dop = ((np.sum(weights * (vel_in - m1_dop[:,None])**3,axis=1))
                        / m2_dop[:,None] ** 3)
        # M4 [-] kurtosis
        m4_dop = ((np.sum(weights * (vel_in - m1_dop[:,None]**4),axis=1))
                        / m2_dop[:,None]**4)

    snr = 10 * np.log10(power/noise_floor) # [-]

    # Create output structure
    params = {'m1_dop':m1_dop,'m2_dop':m2_dop,'m3_dop':m3_dop,'m4_dop':m4_dop,\
              'noise_level':noise_level,'noise_floor':noise_floor,'power':power,\
              'snr':snr}

    return params
