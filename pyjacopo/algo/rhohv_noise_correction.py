#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:27:52 2017

@author: wolfensb
"""

import numpy as np

def rhohv_noise_correction(rhohv, snr_h, zdr, noise_h, noise_v):
    """
    Corrects RhoHV for noise according to eq. 6 in Gourley et al. 2006.
    This correction should only be performed if noise has not been subtracted
    from the signal during the moments computation.

    Inputs:
        rhohv: original rhohv
        snr_h: snr in linear units signal / noise, at hor. pol.
        zdr: diff. reflectivity in dB
        noise_h: noise measurements in power units mW at hor. pol.
        noise_v: noise measurements in power units mW at vert. pol.
    Outputs:
        rhohv_corr: rhohv corrected for noise
    """
    zdr_lin = np.ma.power(10., 0.1*zdr)
    alpha = np.ma.power(10., 0.1*(noise_h - noise_v))

    rhohv_corr = rhohv*np.ma.sqrt((1.+1./snr_h)*(1. + zdr_lin/(alpha*snr_h)))
    rhohv_corr[rhohv_corr > 1.] = 1.

    return rhohv_corr