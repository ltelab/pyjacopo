#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:16:48 2017

@author: wolfensb
"""

import numpy as np

def _attenuation_correction_hbordan(rres,zh,apar = 0.0001825191, bpar = 0.7222082,\
                                    r0 = 0):
    '''
    def _attenuation_correction_hbordan(rres,zh,apar = 0.0001825191, bpar = 0.7222082,\
                                    r0 = 0)
    
    PURPOSE:
        Hitschfeld and Bordan attenuation correction for reflectivity Z
        For complete explanation and more complex implementation of the attenuation
        correction scheme see the article:

        W. Hitshfeld and J. Bordan, "Errors inherent in the
        radar measurement of rainfall at attenuating
        wavelengths",J.Metcorol.,11.58-67.1954
        
        NOTES:
        1) The DSD dependent parameters refer to the following power laws:
        Ah=apar*Zh^bpar    : relation between attenuation and non-attenuated reflectivity
        Ah is in [db/km] while Zh in linear scale
        2) There is no quality check in the routine (melting layer, etc)
        3) With respect to the original paper, the calibration is assumed to be
        perfect

    INPUTS:
        rres: range gate spacing in [m]
        zh : radial of ZH observations in [dBZ]
        apar : intercept parameter of the k = f(z) relation
        bpar : power parameter of the k = f(z) relation
        r0 : Index of the first valid gate to consider (default 0) 
        
    OUTPUTS:
        output_corr: structure with two fields "pia", the path-integrated
                     attenuation [dBZ] and "zh_corr" the corrected refl.
    '''
    
    # Convert rres to km
    rres /= 1000
    
    # Initialize output
    output_corr = {}
    zh_corr = np.zeros(zh.shape) + np.nan
    pia = np.zeros(zh.shape) + np.nan   

    # Linearize Zh
    zh_lin = 10 ** (0.1* zh)
    
    num_1 = -1*0.2*bpar*apar*np.log(10.)*rres #Invariant factor 1
    integral = np.cumsum(zh_lin[r0:]**bpar) * num_1
    den_1 = (1.+integral)**(1./bpar)
    zh_corr[r0:] = zh_lin[r0:] / den_1
    
    valids = np.isfinite(zh_corr)
    zh_corr[valids] = 10 * np.log10(zh_corr[valids])
    pia[valids] = zh_corr[valids] - zh[valids] # log scale
    
    output_corr = {'pia':pia,'zh_corr':zh_corr}
    return output_corr
    
def _attenuation_correction_phidp(rres,zh,zdr,phidp, phi_param_method = 'MFrance',\
                                  zphi = False):
    
    
    # Set parameterization
    if phi_param_method == 'Ryzhkov':
        attcoeffZh=0.25; attcoeffZdr=0.033
    else:
        attcoeffZh=0.28; attcoeffZdr=0.04

    if zphi:
         bpar = 0.64884 # (for conversion from phidp to piavol)
         gpar =  0.31916  # gamma coeff for specific differential attenuation
         cpar = 0.15917   # alpha
         dpar = 1.0804 # beta
         delta_phitotmin = 2  # minimum total differential phase shift [Deg]    
    
    # Convert rres to km
    rres /= 1000
    
    # Define output
    ah = np.zeros(zh.shape)
    output_corr = {} # output dictionary
    
    # Put zh in linear units
    zhlin = 10**(zh/10.)
    # Remove Phidp < 0 and not valid phidp
    phidp[phidp < 0] = np.nan
    
    # Fill gaps ensuring monoticity
    phidp = np.maximum.accumulate(phidp)
    
    # ZPHI method
    if zphi:
        zhlinb = zhlin**bpar
        coeffint = 0.46*bpar*rres
        coeffcpia=0.1*bpar
        coeffpia=2.* rres
    				
        delta_phitot = np.max(phidp)   
        piatot = gpar*delta_phitot
        cpia = 10**(coeffcpia*piatot)-1.			
        intrtot = coeffint * np.sum(zhlinb, 2)
         
        # Reverse zhlinb and make the total
        intr = coeffint * np.cumsum(zhlinb)[::-1]
        
        if delta_phitot > delta_phitotmin:
            ah = zhlinb * cpia/intrtot + cpia*intr
        
        pia = coeffpia * np.sum(ah)
    
        if len(zdr):
            adp = cpar * ah ** dpar
            pida = coeffpia * np.sum(adp)
            
    else:
        pia = attcoeffZh * phidp
        
        # Increments of PIA (no loop this way)
        pia_shift = np.diff(pia)
        pia_shift = np.insert(pia_shift,0,0)
        
        denominator = 2 * rres
        
        # Fill ahvol
        ah = pia_shift / denominator
        # Fill adpvol
        
        if len(zdr):
            pida = attcoeffZdr*phidp
            pida_shift = np.diff(pia_shift)
            pida_shift = np.insert(pia_shift,0,0)
            adp = pida_shift/denominator

    # Generate attenuation-corrected output
    zh_corr = zh + pia
            
    if len(zdr):  
        zdr_corr = zdr + pida
        
        output_corr['adp'] = adp
        output_corr['pida'] = pida   
        output_corr['zdr_corr'] = zdr_corr
    
    output_corr['ah'] = ah
    output_corr['pia'] = pia
    output_corr['zh_corr'] = zh_corr

    return output_corr
    
def corr_att(rres,zh,zdr,phidp,attcorr_struct):
    method = attcorr_struct['method']
    
    if method == 'ZPHI':
        output_corr = _attenuation_correction_phidp(rres,zh,zdr,phidp,True)
    elif method == 'PHILINEAR':
        output_corr = _attenuation_correction_phidp(rres,zh,zdr,phidp)        
    elif method == 'HBORDAN':
        output_corr = _attenuation_correction_hbordan(rres,zh)        
    return output_corr
