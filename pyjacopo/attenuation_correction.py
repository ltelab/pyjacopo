#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

'''
Attenuation correction (for MXPol).

Two algorithm are implemented, all of them based on the Z-PHI algorithm described
in Bringi et al. 2001: "Correcting C-Band Radar Reflectivity and Differential Reflectivity 
Data for Rain Attenuation: A Self-Consistent Method With Constraints"

At the end of the script there is a wrapper, corr_att, which can receive in
input a struct called attcorr_struct, used to choose a method.

In all the explanation, we use the notation for RHI scans (range, elevation),
but the function  should be compatible with PPI scans, as long as the
arrays containing the fields have "range" as their first dimension and
"azimuth" (instead of elevation) as the second one. This has not been tested yet.

When a melting layer field is provided in the input, the calculation of specific 
attenuation is only performed using range gates below melting layer top.

References:
    V. N. Bringi, T. D. Keenan, and V. Chandrasekar. 2001: "Correcting C-Band Radar Reflectivity 
    and Differential Reflectivity Data for Rain Attenuation: A Self-Consistent Method With 
    Constraints", IEEE
    
    S.G. Park, V.N. Bringi, V. Chandrasekar, M. Marki and K. Iwanami, 2005: "Correction of Radar 
    Reflectivity and Differential Reflectivity for RainAttenuation atX Band. Part I: Theoretical 
    and Empirical Basis", JAOT
    
    A. Ryzhkov, M. Diederich, P. Zhang, C. Simmer, 2014: "Potential Utilization of Specific 
    Attenuation for Rainfall Estimation, Mitigation of Partial Beam Blockage and Radar 
    Networking.", JAOT
    

FORMULAS:
    --------------------------
    0) BASIC FORMULAS [Park et al. 2005]
    --------------------------
    Specific attenuation:
        Ah = alpha * Kdp
        Ah = a * Z^b
    Specific differential attenuation:
        Adp = gamma * Ah^d
    
    --------------------------
    1) REFLECTIVITY CORRECTION
    --------------------------
    The specific attenuation is computed using:
    (Bringi et al., 2001, formula 14)

                    (Zh[r])^b * (10^(0.1*b*alpha*Delta_Phidp(r0, rm)) - 1)
        Ah[r] = ---------------------------------------------------------------
                I(r0, rm) + (10^(0.1*b*alpha*Delta_Phidp(r0, rm)) - 1)*I(r, rm)

    Where:
    - Ah is the radial profile of the attenuation
    - r is the range along the beam
    - b is an exponent in the relation A_h = a*Z^b
    - alpha is the prefactor in the relation A_h = alpha*K_dp
    - Delta_Phidp(r0, rm) = Phidp(rm) - Phidp(r0)
    - I(r0, rm) = 0.46 * b * integral_from_r0_to_rm(Zh(s)^b ds) (Eq. 15 in B2001)
    - I(r, rm) = 0.46 * b * integral_from_r_to_rm(Zh(s)^b ds)
    - Zh is the uncorrected linear reflectivity field.

    From Ah the path integrated attenuation is computed as:
        Pia[r] = 2.0 * integral_from_r0_to_r(Ah[s] ds)
        
    The corrected reflectivity is:
        10*log10(ZhCorr[r]) = 10*log10(Zh[r]) + Pia[r]

    In the 'ZPHI' algorithm, a fixed value is used for alpha.
    
    In the 'ZPHI_SELF_CONSISTENT' version, an optimal value of alpha is
    determined as proposed in Bringi et al. 2001:
        An estimated Phidp is calculated:
            Phidp_cal[r, alpha] = 2 * integral_from_r0_to_r((Ah[s, alpha] / alpha) ds)

        The optimal alpha is the value that leads to a minimum difference
        between Phidp_cal and Phidp along the whole beam:

        Error_Phidp[alpha] = sum_from_r0_to_rm(abs(Phidp_cal[ri, alpha] - Phidp[ri]))
        
    ---------------------------------------
    2) DIFFERENTIAL REFLECTIVITY CORRECTION
    ---------------------------------------
    In the 'ZPHI' algorithm, the specific differential attenuation is computed using:
        Adp[r] = gamma * Ah[r]^d

        Where:
        - d is a constant close to unity at X band [Park et Al., 2005].
        - gamma is an a-priori fixed value
    
    In the 'ZPHI_SELF_CONSISTENT' version, horizontal and vertical attenuation (Ah, Av) are computed
    using the method described above. Then, the specific differential attenuation is:
        Adp[r] = Ah[r] - Av[r]

    The path integrated differential attenuation is computed as:
        Pida[r] = 2.0 * integral_from_r0_to_r(Adp[r] dr)

    The corrected differential reflectivity is:
        ZdrCorr[r] = Zdr[r] + Pida[r]



'''
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
# Defining a "fill" value in place of NaN
FVALUE = -9999.

# FIXED PARAMETERS (X-band)

# b value:
# It's a rather "constant" value for a given frequency, not depending a lot from temperature.
# In Park et al. 2005 the mean value is 0.78
# In MeteoSwiss pyart it is 0.64884
B = 0.64884

# alpha coefficient used for ZPHI method (or as default for ZPHI-SELF-CONSISTENT)
# In MeteoSwiss pyart (from Ryzhkov 2014) it is 0.31916 
# In Park et al. 2005 the mean value is 0.315 (Andsager relation for drop shapes)
ALPHA = 0.31916

# gamma coefficient for specific differential attenuation
# In Park et al. 2005 (Andsager relation for drop shapes)
GAMMA = 0.128

# d coefficient for specific differential attenuation
# In Park et al. 2005 (Andsager relation for drop shapes)
D = 1.156

# minimum total differential phase shift [Deg]
MIN_DELTA_PHITOT = 1. #2.

def _attenuation_correction_zphi(rres, zh, zdr, phidp, ml=None, below_ml=1, in_ml = 3, above_ml=5,elevation=None):
    '''
    PURPOSE:
        Implementation of the attenuation correction algorithm with fixed a-priori values for all parameters

    INPUTS:
        rres:  np.array, 1D(range), range gate spacing in [m]
        zh :   np.array, 2D(elevation, range), ZH observations in [dBZ]
        zdr:   np.array, 2D(elevation, range), ZDR observations in [dB]
        phidp: np.array, 2D(elevation, range), Phidp
        ml : np.array, 2D(elevation, range), indicator of position vs. melting layer 
                                                (1 = below, 3 = in, 5 = above)

    OUTPUTS:
        output_corr: structure with various fields. It contains:
                    - ah -> specific attenuation data [dBZ/km]
                    - pia -> path-integrated attenuation [dBZ]
                    - zh_corr -> corrected reflectivity [dBZ]
                    If Zdr is provided. it contains also:
                    - adp -> specific differential attenuation data [dB/km]
                    - pida -> the path-integrated differential attenuation [dB]
                    - zdr_corr -> the corrected differential reflectivity [dB]
    '''

    n_rgates = rres.shape[0]
    n_elev = zh.shape[0]
    rres = rres/1000 
    
    # Checking if the melting layer is provided: if not provided, the full profile is used
    if ml is None:
        idx_ml = np.ones(n_elev, dtype='int') * n_rgates - 1
    else:
        idx_ml = ml_to_index(ml)

    # Define output
    ah = np.zeros(zh.shape)
    output_corr = {}  # output dictionary

    # Put zh in linear units, and undefined to 0
    zh_lin = np.power(10., zh / 10.)
    zh_lin[np.logical_not(np.isfinite(zh))] = 0.
    zh_lin[zh < -998.] = 0.
    zh_lin_b = np.power(zh_lin, B) # Zh^b, to be used later
    
    # Fill gaps in Phidp for ensuring monoticity
    # Note: when np.maximum.accumulate meets "Nan", it gives Nan as a result.
    # -> eliminate "NaN" and "inf" values, saving their position and the position values < 0 in an array
    idx_to_exclude = np.logical_not(np.isfinite(phidp))
    phidp[idx_to_exclude] = FVALUE
    phidp_acc = np.maximum.accumulate(phidp,axis=1) # monotonical Phidp along range
    
    # Define containers
    delta_phitot = np.zeros(n_elev, dtype='float32') # Delta_Phidp(r0, rm)
    cpia = np.zeros(n_elev, dtype='float32') # cpia = 10^(0.1*b*alpha*Delta_Phidp(r0, rm)) - 1
    ah = np.zeros(zh.shape, dtype='float32')  # specific att. horiz.
    pia = np.zeros(zh.shape, dtype='float32')  # path-integrated ah horiz. pol
    adp = np.zeros(zh.shape, dtype='float32')  # specific differential att.
    pida = np.zeros(zh.shape, dtype='float32')  # path-integrated differential attenuation

    # loop over all elevations, for computing the attenuation for each beam.
    for i_el in range(n_elev):
        
        # For each beam, define the first and last index along r, depending on ML height and Phi_dp vector.
        min_r = 0
        max_r = 0
        for j_r in np.arange(n_rgates - 1, 2, -1):
            # We exclude isolated gates, so we check three gates
            if (phidp[i_el, j_r - 2] > FVALUE) and (phidp[i_el, j_r - 1] > FVALUE) and (phidp[i_el,j_r] > FVALUE):
                max_r = j_r
                break
        for j_r in np.arange(0, n_rgates - 1, 1):
            if (phidp[i_el, j_r] > FVALUE):
                min_r = j_r
                break

        max_r = min(max_r, idx_ml[i_el]) #modify r_max according to ML height
#         print(max_r)
        if min_r == 0 and max_r == 0: # skip beam if Phi_dp is full of FVALUES
            continue
            
        # Computing Zh**b (zero outside the range min_r, max_r)
        zh_lin_b_beam = np.zeros(n_rgates, dtype='float32')
        zh_lin_b_beam[min_r:max_r] = zh_lin_b[i_el,min_r:max_r]

        # Filling the Delta_Phi_tot array for the current beam
        delta_phitot[i_el] = phidp_acc[i_el, max_r] - phidp_acc[i_el, min_r]

        # Computing the term: 10^(0.1*b*alpha*Delta_Phidp(r0, rm)) - 1
        cpia[i_el] = np.power(10., 0.1*B*ALPHA*delta_phitot[i_el]) -1 

        # I(r0, rm) = 0.46 * b * integral_from_r0_to_rm(Zh(r)^b dr)
        intrtot = 0.46 * B * np.trapz(zh_lin_b_beam[min_r:max_r], x=rres[min_r:max_r])

        # I(r, rm)
        intr = np.zeros(n_rgates, dtype='float32') #container
        intr[min_r:max_r] = 0.46 * B * integrate.cumulative_trapezoid(zh_lin_b_beam[min_r:max_r][::-1],
                                                                      x = rres[min_r:max_r], initial = 0)[::-1]
            
        if delta_phitot[i_el] < MIN_DELTA_PHITOT or intrtot < 0.:
            # If alpha * Delta_Phidp(r0, rm) is lower than the threshold fixed at the beginning, attenuation is 0.
            # Check for (intrtot > 0.) in the if to ensure we don't divide by 0.
            continue
        else:
            # Computing the specific attenuation for the beam:
            ah_beam = np.zeros(n_rgates, dtype='float32')
            ah_beam[min_r:max_r] = np.divide(zh_lin_b_beam[min_r:max_r] * cpia[i_el], 
                                             intrtot + intr[min_r:max_r] * cpia[i_el])
            ah[i_el,min_r:max_r] = ah_beam[min_r:max_r] # saving it in the full attenuation matrix
        
#         Those few lines are just to see what the reconstructed phidp profile looks like
#         phidp_com = np.zeros(n_rgates, dtype='float32') 
#         phidp_com[min_r:max_r] = 2. * integrate.cumulative_trapezoid(ah_beam[min_r:max_r]/ALPHA,
#                                                                     x = rres[min_r:max_r], initial = 0, axis=0)
#         fig = plt.figure()
#         plt.plot(rres[min_r:max_r], phidp_com[min_r:max_r])
#         plt.plot(rres[min_r:max_r], phidp_acc[i_el,min_r:max_r])
#         if not (elevation is None):
#             plt.title(elevation[i_el])
#         plt.show()
            
    # Computing the path integrated attenuation
    pia[:,min_r:] = 2 * integrate.cumulative_trapezoid(ah[:,min_r:], x=rres[min_r:],initial=0,axis=1)
    zh_corr = zh + pia # attenuation-corrected Zh
    
    # If the Zdr array has been provided, we compute also its correction
    if len(zdr):
        adp = GAMMA * np.power(ah, D)
        pida[i_el,min_r:] = 2 * integrate.cumulative_trapezoid(adp[i_el,min_r:], x = rres[min_r:], initial = 0)

        zdr_corr = zdr + pida

        # Finally, we fill the output dictionary
        output_corr['adp'] = adp
        output_corr['pida'] = pida
        output_corr['zdr_corr'] = zdr_corr

    output_corr['ah'] = ah
    output_corr['pia'] = pia
    output_corr['zh_corr'] = zh_corr

    return output_corr


def _attenuation_correction_zphi_self_consistent(rres, zh, zv, zdr, phidp,
                                            ml=None,elevation=None):
    '''
    PURPOSE:
        Implementation of the attenuation correction algorithm with choice of optimal alpha.
        
    INPUTS:
        rres:  np.array, 1D(range), range gate spacing in [m]
        zh :   np.array, 2D(range, elevation), ZH observations in [dBZ]
        zdr:   np.array, 2D(range, elevation), ZDR observations in [dBZ]
        phidp: np.array, 2D(range, elevation), Phidp
        ml_r : np.array, 1D(elevation), distance in m on the beam
                        at which there is the melting layer.

    OUTPUTS:
        output_corr: structure with various fields. It contains:
                    - ah -> specific attenuation data [dBZ/km]
                    - pia -> the path-integrated attenuation [dBZ]
                    - zh_corr -> the corrected reflectivity [dBZ]
                    - av -> specific vertical attenuation data [dBZ/km]
                    - piav -> the path-integrated vertical attenuation [dBZ]
                    - zv_corr -> the corrected vertical reflectivity [dBZ]
                    If Zdr is provided, it contains also:
                    - adp -> specific differential attenuation data [dB/km]
                    - pida -> the path-integrated differential attenuation [dB]
                    - zdr_corr -> the corrected differential reflectivity [dB]
    '''

    # PARAMETERS
    # Min. and max. alpha to test, when deciding optimal value (Liu et al, 2006)
    alpha_min = 0.1
    alpha_max = 0.8
    alpha_step = 0.02  # The step used for trying all the alphas
    alpha_default = ALPHA  # Default value, form Jacopo's script
    alpha_vec = np.arange(alpha_min, alpha_max, alpha_step)
    num_alphas = alpha_vec.shape[0]

    n_rgates = rres.shape[0]
    n_elev = zh.shape[0]
    
    rres = rres/1000 
    
    # Checking if the ml height is provided: if not provided, the full profile is used
    if ml is None:
        idx_ml = np.ones(n_elev, dtype='int') * n_rgates
    else:
        idx_ml = ml_to_index(ml)

    # Define output
    ah = np.zeros(zh.shape)
    output_corr = {}  # output dictionary

    # Put zh in linear units and set to zero the "excluded" values
    zh_lin = np.power(10., zh / 10.)
    zh_lin[np.logical_not(np.isfinite(zh))] = 0.
    zh_lin[zh < -998.] = 0.

    # And do the same for zv
    if zv is None:
        # In case it was not provided, we compute zv from zh and zdr
        zdr_lin = np.power(10., zdr / 10.)
        # We have to exclude invalid values of zdr and zh
        pos_defined = np.logical_and(np.isfinite(zdr_lin), np.isfinite(zh_lin))
        # And also the position in which they are too low and we risk to get super-high value when dividing
        pos_defined[zdr_lin < 1e-6] = False
        pos_defined[zh_lin < 1e-6] = False

        # In invalid position, we put a zero
        zv_lin = np.zeros(zh_lin.shape, dtype='float32')
        zv_lin[pos_defined] = np.divide(zh_lin[pos_defined], zdr_lin[pos_defined])
        # We compute also the logarithmic form
        zv = np.zeros(zh.shape, dtype='float32')
        zv[pos_defined] = 10. * np.log10(zv_lin[pos_defined])
    else:
        # Otherwise we proceed as usual, putting to zero the "excluded" values
        zv_lin = np.power(10., zv / 10.)
        zv_lin[np.logical_not(np.isfinite(zv))] = 0.
        zv_lin[zv < -998.] = 0.

    # Fill gaps in Phidp for ensuring monoticity
    # Note: when np.maximum.accumulate meets "Nan", it gives Nan as a result.
    # -> eliminate "NaN" and "inf" values, saving their position and the position values < 0 in an array
    idx_to_exclude = np.logical_not(np.isfinite(phidp))
    phidp[idx_to_exclude] = FVALUE
    # And this is the monotonic Phi_dp
    phidp_acc = np.maximum.accumulate(phidp, axis=1)
    phidp_acc = phidp
    # Let's assign some variables to quantities we well use later:
    zh_lin_b = np.power(zh_lin, B) # Zh^b
    zv_lin_b = np.power(zv_lin, B) # Zv^b


    # Define containers
    delta_phitot = np.zeros(n_elev, dtype='float32') # Delta_Phidp(r0, rm)
    cpia = np.zeros((n_elev,num_alphas), dtype='float32') # cpia = 10^(0.1*b*alpha*Delta_Phidp(r0, rm)) - 1
    ah = np.zeros(zh.shape, dtype='float32')  # specific att. horiz.
    pia = np.zeros(zh.shape, dtype='float32')  # path-integrated ah horiz. pol
    av = np.zeros(zh.shape, dtype='float32')  # specific att. vert. pol.
    piav = np.zeros(zh.shape, dtype='float32')  # path-integrated ah vert. pol.
    adp = np.zeros(zh.shape, dtype='float32')  # specific differential att.
    pida = np.zeros(zh.shape, dtype='float32')  # path-integrated differential attenuation

 
    # loop over all elevations, for computing the attenuation for each beam.
    for i_el in range(n_elev):
        
        # For each beam, define the first and last index along r, depending on ML height and Phi_dp vector.
        min_r = 0
        max_r = 0
        for j_r in np.arange(n_rgates - 1, 2, -1):
            # We exclude isolated gates, so we check three gates
            if (phidp[i_el, j_r - 2] > FVALUE) and (phidp[i_el, j_r - 1] > FVALUE) and (phidp[i_el,j_r] > FVALUE):
                max_r = j_r
                break
        for j_r in np.arange(0, n_rgates - 1, 1):
            if (phidp[i_el, j_r] > FVALUE):
                min_r = j_r
                break

        max_r = min(max_r, idx_ml[i_el]) #modify r_max according to ML height

        if min_r == 0 and max_r == 0: # skip beam if Phi_dp is full of FVALUES
            continue
            

        # Computing Zh**b and Zv**b (it's zero outside the range min_r, max_r)
        zh_lin_b_beam = np.zeros(n_rgates, dtype='float32')
        zv_lin_b_beam = np.zeros(n_rgates, dtype='float32')
        zh_lin_b_beam[min_r:max_r] = zh_lin_b[i_el,min_r:max_r]
        zv_lin_b_beam[min_r:max_r] = zv_lin_b[i_el,min_r:max_r]
        
        # Filling the Delta_Phi_tot array for the current beam
        delta_phitot[i_el] = phidp_acc[i_el, max_r] - phidp_acc[i_el, min_r]


        # Compute 10^(0.1*b*alpha*Delta_Phidp(r0, rm)) - 1 for all alpha values
        cpia[i_el,:] = np.power(10.,0.1*B*alpha_vec*delta_phitot[i_el]) -1 
        
        # I(r0, rm) = 0.46 * b * integral_from_r0_to_rm(Zh(r)^b dr)
        intrtot = 0.46 * B * np.trapz(zh_lin_b_beam[min_r:max_r], x=rres[min_r:max_r])
        intrtot_v = 0.46 * B  * np.trapz(zv_lin_b_beam[min_r:max_r],x=rres[min_r:max_r])

        # I(r, rm) and Iv(r, rm) (Note that the cumulative integral is "backwards" because the integral is from r to rm):
        intr = np.zeros(rres.shape[0], dtype='float32')
        intr[min_r:max_r] = 0.46 * B * integrate.cumulative_trapezoid(zh_lin_b_beam[min_r:max_r][::-1],
                                                                      x = rres[min_r:max_r], initial = 0)[::-1]
        intr_v = np.zeros(rres.shape[0], dtype='float32')
        intr_v[min_r:max_r] = 0.46 * B * integrate.cumulative_trapezoid(zv_lin_b_beam[min_r:max_r][::-1],
                                                                      x = rres[min_r:max_r], initial = 0)[::-1]

        if delta_phitot[i_el] < MIN_DELTA_PHITOT:# or intrtot < 0. or intrtot_v < 0.:
            # If alpha * Delta_Phidp(r0, rm) is lower than the threshold fixed at the beginning, attenuation is 0.
            # Check for (intrtot > 0.) in the if to ensure we don't divide by 0.
            print('continue')
            continue

#     else:
            # Containers for error and intermediate computations of att and Phidp
        error = np.full(num_alphas, 9e7, dtype='float32') #for the error
        error_v = np.full(num_alphas, 9e7, dtype='float32')

        ah_beam = np.zeros([n_rgates, num_alphas], dtype='float32') 
        phidp_com = np.zeros([n_rgates, num_alphas], dtype='float32') 
        av_beam = np.zeros([n_rgates, num_alphas], dtype='float32')
        phidp_com_v = np.zeros([n_rgates, num_alphas], dtype='float32')

        # compute specific attenuation (h and v) for all alpha values
        ah_beam[min_r:max_r,:] = np.divide(zh_lin_b_beam[min_r:max_r,None] * cpia[i_el:i_el+1,:],
                                          intrtot + intr[min_r:max_r,None] * cpia[i_el:i_el+1,:])
        av_beam[min_r:max_r, :] = np.divide(zv_lin_b_beam[min_r:max_r,None] * cpia[i_el:i_el+1, :],
                        intrtot_v + intr_v[min_r:max_r,None] * cpia[i_el:i_el+1, :])

        phidp_com[min_r:max_r, :] = 2. * integrate.cumulative_trapezoid(ah_beam[min_r:max_r, :]/alpha_vec,
                                                                        x = rres[min_r:max_r], initial = 0, 
                                                                        axis=0)
        phidp_com_v[min_r:max_r, :] = 2 * integrate.cumulative_trapezoid(av_beam[min_r:max_r, :]/alpha_vec,
                                                                         x = rres[min_r:max_r], initial = 0, 
                                                                         axis=0)

        error = np.nansum(np.abs(phidp_com[min_r:max_r, :] - phidp_acc[i_el,min_r:max_r, None]),axis=0)
        error_v = np.nansum(np.abs(phidp_com_v[min_r:max_r, :] - phidp_acc[i_el,min_r:max_r, None]),axis=0)

#             Those few lines are just to see what the reconstructed phidp profile looks like
        fig = plt.figure()
        plt.plot(rres[min_r:max_r], phidp_com[min_r:max_r])
        plt.plot(rres[min_r:max_r], phidp_acc[i_el,min_r:max_r],'k',linewidth=2, label='true')
        if not(elevation is None):
            plt.title('el = %.1f'%elevation[i_el])
        plt.xlabel('range (km)')
        plt.ylabel('Phidp (deg)')
#             plt.ylim(0,min(10,np.max(phidp_com)))
        plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
        plt.show()

        chosen_alpha_idx = np.nanargmin(error)
        if error[chosen_alpha_idx] > 0. and error[chosen_alpha_idx] < 9e7:
            chosen_alpha = alpha_vec[chosen_alpha_idx]
        else:
            # If, for some reason, the error has remained = NaN, we use the alpha closest to the default one
            chosen_alpha = alpha_default
            chosen_alpha_idx = np.nanargmin(np.abs(alpha_vec - alpha_default))

        # Vertical
        chosen_alpha_idx_v = np.nanargmin(error_v)
        if error_v[chosen_alpha_idx_v] > 0. and error_v[chosen_alpha_idx_v] < 9e7:
            chosen_alpha_v = alpha_vec[chosen_alpha_idx_v]
        else:
            chosen_alpha_v = alpha_default
            chosen_alpha_idx_v = np.nanargmin(np.abs(alpha_vec - alpha_default))

        ah[i_el, min_r:max_r] = ah_beam[min_r:max_r, chosen_alpha_idx]
        av[i_el, min_r:max_r] = av_beam[min_r:max_r, chosen_alpha_idx_v]

        print(chosen_alpha, chosen_alpha_v)
            
    # Computing the path integrated attenuation
    pia[:,min_r:] = 2 * integrate.cumulative_trapezoid(ah[:,min_r:], x=rres[min_r:],initial=0,axis=1)
    piav[:,min_r:] = 2 * integrate.cumulative_trapezoid(av[:,min_r:], x=rres[min_r:],initial=0,axis=1)
    zh_corr = zh + pia
    zv_corr = zv + piav

    # If the Zdr array has been provided, we compute also its correction
    if len(zdr):
        # specific differential attenuation
        adp[:, :] = ah[i_el, :] - av[:, :]
        pida[:,min_r:] = 2 * integrate.cumulative_trapezoid(adp[:,min_r:], x=rres[min_r:],initial=0,axis=1)

        zdr_corr = zdr + pida
        zdr_corr[zv_lin == 0.] = FVALUE
        zdr_corr[zv_lin == 0.] = FVALUE

        # Finally, we fill the output dictionary
        output_corr['adp'] = adp
        output_corr['pida'] = pida
        output_corr['zdr_corr'] = zdr_corr

    output_corr['ah'] = ah
    output_corr['pia'] = pia
    output_corr['zh_corr'] = zh_corr

    output_corr['av'] = av
    output_corr['piav'] = piav
    output_corr['zv_corr'] = zv_corr

    return output_corr


def slicing_at_nan(a):
    # A small wrapper around numpy functions, for slicing an array at nans.
    # It returns slices for all the contiguous non-nan entries of the array
    return np.ma.clump_unmasked(np.ma.masked_invalid(a))

def ml_to_index(ml):
    idx_ml = np.ones(ml.shape[0], dtype='int') * ml.shape[1]
    for i in range(ml.shape[0]):
        imlb = np.where(ml[i,:]==1)[0]
        if len(imlb)>0:
            idx_ml[i] = imlb[-1]#int(len(imlb)/2)]
            print(idx_ml[i])
    return idx_ml


def corr_att(rres, zh, zdr, phidp, method='ZPHI',
             zv=None, ml=None, rhohv=None, elevation=None):
    '''
    This function calls the attenuation correction function depending
    on the attcorr_struct provided.

    INPUTS:
        rres: 
        attcorr_struct: contains 'method' field
        rres:  np.array, 1D(range), range in [m]
        zh :   np.array, 2D(range, elevation), ZH observations in [dBZ]
        zdr:   np.array, 2D(range, elevation), ZDR observations in [dBZ]
        phidp: np.array, 2D(range, elevation), Phidp
        ml_r : np.array, 1D(elevation), distance in m on the beam
                        at which there is the melting layer.
    In input, attcorr_struct should contain at least one field, called 'method',
    whose value is a string containing the name of the correction to use.
    For example:
    attcorr_struct = {'method': 'ZPHI'}
    (This is the chosen default value)

    Note that the "method" string is case insensitive, since we apply "upper"
    when reading it.

    For now, only the ZPHI and ZPHI-LIU2006 methods are implemented.
    For more information on them, refer to the description at the beginning
    of the two functions.
    '''

    # Extracting the method string from the attcorr_struct
#     method = attcorr_struct['method'].upper()

    # Defining an error code
    ERR = {'ah': np.full(zh.shape, np.nan, dtype='float32'),
            'pia': np.full(zh.shape, np.nan, dtype='float32'),
            'zh_corr': np.full(zh.shape, np.nan, dtype='float32'),
            'adp': np.full(zh.shape, np.nan, dtype='float32'),
            'pida': np.full(zh.shape, np.nan, dtype='float32'),
            'zdr_corr': np.full(zh.shape, np.nan, dtype='float32')}

    zh = zh
    zdr = zdr
    phidp = phidp
    
    if method == 'ZPHI':
        # The basic Z-Phi algorithm, with fixed parameters.
        output_corr = _attenuation_correction_zphi(rres, zh, zdr, phidp,ml=ml, elevation=elevation)
    elif method == 'ZPHI_SELF_CONSISTENT':
        # The Z-Phi variation, with optimal choosing of alpha and
        # the Z_dr correction described in Liu et al. 2006 (see description)
        output_corr = _attenuation_correction_zphi_self_consistent(rres, zh, zv, zdr, phidp,
                                                            ml=ml,elevation=elevation)
    elif method == 'ZPHI-MOD':
        # The basic Z-Phi algorithm, in the PyART implementation
        output_corr = _attenuation_correction_mod(rres, zh, zv, zdr, phidp,
                                                            ml_r=ml_r)

    return output_corr
