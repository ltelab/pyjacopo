#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:41:11 2017

@author: wolfensb
"""

import numpy as np
from scipy import interpolate


def phidp_kdp_profile(rres, psidp_in, phidp_kdp_struc):
    
    """
    Generic method for phidp and kdp estimation

    Parameters
    ----------
    rres : float
        Range resolution in meters.
    psidp_in : ndarray
        Total differential phase measurements.
    windsize : int, optional
        Length of the rang range derivative window (in meters)
    band : char, optional
        Radar frequency band string. Accepted "X", "C", "S" (capital
        or not). It is used to set default boundaries for expected
        values of Kdp
    n_iter : int, optional
        Ǹumber of iterations of the method. Default is 10.
    interp : bool, optional
        If set all the nans are interpolated.The advantage is that less data 
        are lost (the iterations in fact are "eating the edges") but some
        non-linear errors may be introduced
        
    Returns
    -------
    kdp_calc : ndarray
        Retrieved specific differential profile
    phidp_rec,: ndarray
        Retrieved differential phase profile

    """
    
    pk_s = phidp_kdp_struc
    
    phidp_method = pk_s['phidp_method']
    kdp_method = pk_s['kdp_method']    
    rad_freq = pk_s['rad_freq']
    
    if rad_freq < 4:
        band = 'S'
    elif rad_freq < 8:
        band = 'C'
    elif rad_freq < 12:
        band = 'X'
            
    if phidp_method == 'MULTISTEP': # Vulpiani
        windsize = pk_s['phidp_windsize']

        n_iter = pk_s['phidp_niterations'] 
        kdp, phidp = _kdp_vulpiani_profile(rres,psidp_in,windsize, band,
                                           n_iter)
    
    if kdp_method == 'MULTISTEP': # Vulpiani
        
        # Check if kdp_method = phidp_method and same parameters
        same = False
        if phidp_method == kdp_method:
            if pk_s['phidp_niterations'] == pk_s['kdp_niterations'] \
            and pk_s['phidpwindsize'] == pk_s['phidpwindsize'] :
                same = True
        
        if not same:
            windsize = pk_s['kdp_windsize']
            n_iter = pk_s['kdp_niterations'] 
            kdp, _ = _kdp_vulpiani_profile(rres,psidp_in,windsize, band,
                                               n_iter)      
            
    out = {}
    out['phidp'] = phidp
    out['kdp'] = kdp         
    return out
    
def _kdp_vulpiani_profile(rres,psidp_in, windsize = 10,band = 'X', n_iter = 10, interp = False):
    
    """
    Estimates Kdp with the Vulpiani method for a single profile of psidp measurements

    Parameters
    ----------
    rres : float
        Range resolution in meters.
    psidp_in : ndarray
        Total differential phase measurements.
    windsize : int, optional
        Length of the rang range derivative window (in meters)
    band : char, optional
        Radar frequency band string. Accepted "X", "C", "S" (capital
        or not). It is used to set default boundaries for expected
        values of Kdp
    n_iter : int, optional
        Ǹumber of iterations of the method. Default is 10.
    interp : bool, optional
        If set all the nans are interpolated.The advantage is that less data 
        are lost (the iterations in fact are "eating the edges") but some
        non-linear errors may be introduced
        
    Returns
    -------
    kdp_calc : ndarray
        Retrieved specific differential profile
    phidp_rec,: ndarray
        Retrieved differential phase profile

    """

    if not np.isfinite(psidp_in).any(): # Check if psidp has at least one finite value
        return psidp_in,psidp_in, psidp_in # Return the NaNs...
    
    l = int(windsize/rres)
    l += l%2 # l needs to be even
    
    #Thresholds in kdp calculation
    if band == 'X':   
        th1 = -2.
        th2 = 25.  
    elif band == 'C':   
        th1 = -0.5
        th2 = 15.
    elif band == 'S':   
        th1 = -0.5
        th2 = 10.
    else:   
        print('Unexpected value set for the band keyword ')
        print(band)
        return None
    
    psidp = psidp_in
    nn = len(psidp_in)
   
    #Get information of valid and non valid points in psidp the new psidp
    nonan = np.where(np.isfinite(psidp))[0]
    nan =  np.where(np.isnan(psidp))[0]
    if interp:
        ranged = np.arange(0,nn)
        psidp_interp = psidp
        # interpolate
        if len(nan):
            interp = interpolate.interp1d(ranged[nonan],psidp[nonan],kind='zero',
                                          bounds_error=False, fill_value=np.nan)
            psidp_interp[nan] = interp(ranged[nan])
            
        psidp = psidp_interp
        
    psidp = np.ma.filled(psidp, np.nan)
    kdp_calc = np.zeros([nn]) * np.nan
    
    #Loop over range profile and iteration
    for ii in range(0, n_iter):
        # In the core of the profile
        kdp_calc[int(l/2):nn - int(l / 2)] = (psidp[l:nn] - psidp[0:nn-l]) / (2. * l * rres/1000.)
            
        # In the beginnning of the profile: use all the available data on the RHS
        kdp_calc[0:int(l/2)] = (psidp[l] - psidp[0]) / (2. * l * rres)        
        
        # In the end of the profile: use the  LHS available data
        kdp_calc[nn - int(l/2):] = (psidp[nn - 1] - psidp[nn - l - 1]) / (2. * l * rres/1000.)             
                
        #apply thresholds
        kdp_calc[kdp_calc <= th1 ] = th1
        kdp_calc[kdp_calc >= th2 ] = th2

        #Erase first and last gate
        kdp_calc[0] = np.nan
        kdp_calc[nn - 1] = np.nan
    
    kdp_calc = np.ma.masked_array(kdp_calc, mask = np.isnan(kdp_calc))
    #Reconstruct Phidp from Kdp
    phidp_rec = np.cumsum(kdp_calc) * 2. * rres/1000.

    #Censor Kdp where Psidp was not defined
    if len(nan):   
        kdp_calc[nan] = np.nan
        
    return kdp_calc, phidp_rec

