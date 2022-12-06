# -*- coding: utf-8 -*-

'''
Defines the default values for all user inputs

When the name of a key is prepended by a *, it means that the field is optional
i.e. if the user does not enter this field, no default will be set

For example in PROCESSING_STANDARD_DEFAULT, phidp_kdp is optional,
meaning that if the user does not give anything for this key, NaN will be assigned
to the phidp_kdp key in the final parsed dictionary
'''

import numpy as np

# Flag to indicate mandatory fields
MANDATORY = -9999.9


###############################################################################
# 1) PRODUCTS DEFAULT
# Note that some fields are used only for certain types of products
PRODUCT_DEFAULT = {}

PRODUCT_DEFAULT['PPI_IMAGE'] = {
    'variables': MANDATORY, \
    'continuous_cb': False,\
}

PRODUCT_DEFAULT['RHI_IMAGE'] = {
    'variables': MANDATORY, \
    'continuous_cb': False,\
}

PRODUCT_DEFAULT['PPI_RHI_IMAGE'] = {
    'variables': MANDATORY, \
    'continuous_cb': False,\
}

PRODUCT_DEFAULT['PPI_RHI_3D_COMP'] = {
    'variables': MANDATORY, \
    'continuous_cb': False,\
    'views': 1,\
    'nanval': np.nan,\
}

PRODUCT_DEFAULT['TIMEHEIGHT_IMAGE'] = {
    'variables': MANDATORY, \
    'continuous_cb': False,\
    'res': [1000,600],\
    'rebin_seconds': 30,\
}

PRODUCT_DEFAULT['SPECTROGRAM_IMAGE'] = {
    'variables': MANDATORY, \
    'continuous_cb': False,\
    'stacked_spectra': False,\
}

PRODUCT_DEFAULT['NETCDF_POLAR'] = {
    'add_fields': [],\
    'nanval': np.nan,\
}

PRODUCT_DEFAULT['NETCDF_POLAR_SPECTRA'] = {
    'add_fields': [],\
    'nanval': np.nan,\
}

PRODUCT_DEFAULT['NETCDF_PROFILE_SPECTRA'] = {
    'nanval': np.nan,\
}

PRODUCT_DEFAULT['NETCDF_TIMEHEIGHT_PROFILE'] = {
    'nanval': np.nan,\
}

###############################################################################
# 2) PROCESSING MODULES DEFAULT

# 2.1) Kdp estimation
PHIDP_KDP_DEFAULT = {
    'phidp_method' : 'MEDIAN',\
    'kdp_method' : 'DERIVATION',\
    'int_kdp' : 0,\
    'unwrap' : 1,\
    'jitter': 2000,\
    'offset_calc' : 0,\
    'offset_ngates': 5,\
    'point_int_remove': 1,\
    'snr_th': 10,\
    'rhohv_th': 0.6,\
    'phidp_niterations': 10,\
    'kdp_niterations': 3,\
    # DEFAULT for rcov and pcov are provided in appropriate part of the code
    'rcov_file': None,\
    'pcov_file': None,\
    'phidp_windsize': 1000,\
    'phidp_windsize_s': 1000,\
    'phidp_windsize_l': 2000, \
    'phidp_zh_min': 30,\
    'kdp_windsize':1000,\
    'kdp_windsize_s': 1000,\
    'kdp_windsize_l': 3000,\
    'kdp_windsize_i': 1500,\
    'kdp_zh_min': 30,\
    'kdp_zh_min_2': 45
}

# 2.2) Clutter filter
CLUTTER_FILTER_DEFAULT = {
    'method': 'NOTCH',\
    'pre_filter': 0,\
    'vel_th': 0.2,\
    'use_mag': 1,\
}

# 2.3) Attenuation correction
ATTENUATION_CORRECTION_DEFAULT = {
    'method': 'ZPHI',\
}

# 3.4) TODO - Hydrometeor classification
HYDROMETEOR_CLASSIFICATION_DEFAULT = {}

###############################################################################
# 3) PROCESSING DEFAULT

# 3.1) PPI/RHI/SectorScan data
PROCESSING_STANDARD_DEFAULT = {
    'czdr_apply': False,\
    'snr_pow': 5,\
    'snr_phase': 10,\
    'rhohv_th': 0.5,\
    'discard_spectra': True,\
    'discard_signal': True,\
    '*mask_out': {'snr':0,'rhohv':0.6},\
    '*phidp_kdp': PHIDP_KDP_DEFAULT,\
    '*att_corr': ATTENUATION_CORRECTION_DEFAULT,\
    'rhohv_snr_correction': False,\
}

# 3.2) Spectral data
PROCESSING_SPECTRA_DEFAULT={
    'notch_spectra': True,\
    'dealias_spectra': True,\
    'time_average': 200,\
    'rhohv_th': 0.5,\
    'snr_th': -10,\
    '*clutter_filter': CLUTTER_FILTER_DEFAULT,\
    # DSD retrieval not in IDL framework yet...
    '*dsd_retrieval': {},\
}


###############################################################################
# 4) DATASETS DEFAULT

DATASET_DEFAULT = {}

DATASET_DEFAULT['PPI'] = {
    'angles': MANDATORY,\
    'nang_max': 500,\
    'nang_min': 10,\
    'ang_tol': 0.3,\
    'range_min': 300,\
    'processing': PROCESSING_STANDARD_DEFAULT,\
    'products': PRODUCT_DEFAULT,\
    'image': True
}

DATASET_DEFAULT['RHI'] = {
    'angles': MANDATORY,\
    'nang_max': 250,\
    'nang_min': 10,\
    'ang_tol': 0.3,\
    'range_min': 300,\
    'processing': PROCESSING_STANDARD_DEFAULT,\
    'products': PRODUCT_DEFAULT,\
    'image': True
}

DATASET_DEFAULT['SECTOR_SCAN'] = {
    'angles': MANDATORY,\
    'nang_max': 500,\
    'nang_min': 10,\
    'ang_tol': 0.3,\
    'range_min': 300,\
    'processing': PROCESSING_STANDARD_DEFAULT,\
    'products': PRODUCT_DEFAULT,\
    'image': True
}

DATASET_DEFAULT['WINDOW'] = {
     # nang_max default is to be adjusted at runtime (nb of records in raw file)
    'nang_max': None,\
    'nang_min': 30,\
    'min_az': -361,\
    'min_el': -1,\
    'processing': PROCESSING_STANDARD_DEFAULT,\
    'products': PRODUCT_DEFAULT,\
}

DATASET_DEFAULT['PROFILE'] = {
    'angles': MANDATORY,\
    'nang_max': 500,\
    'nang_min': 10,\
    'ang_tol': 0.3,\
    'range_min': 300,\
    'processing': PROCESSING_STANDARD_DEFAULT,\
    'products': PRODUCT_DEFAULT,\
    'image': True
}

DATASET_DEFAULT['PROFILE_NO_PEDESTAL'] = {
    'angles': MANDATORY,\
    'az_angles': None,\
    'nang_max': 500,\
    'nang_min': 10,\
    'ang_tol': 0.3,\
    'range_min': 300,\
    'processing': PROCESSING_STANDARD_DEFAULT,\
    'products': PRODUCT_DEFAULT,\
    'image': True
}

DATASET_DEFAULT['SPECTRA_PROFILE'] = {
    'input_complement': '*.nc',\
    'processing': PROCESSING_SPECTRA_DEFAULT,\
    'products': PRODUCT_DEFAULT,
}

# TODO: SPECTRA_RHI_DATA

DATASET_DEFAULT['TIMEHEIGHT'] = {
    'input_complement': '*.nc',\
    'products': PRODUCT_DEFAULT,
}

DATASET_DEFAULT['TIME_SUMMARY_HEIGHT'] = {
     'input_complement': '*.nc',\
     'variables': ['ZH'],\
     'max_ele': 50,\
     'stats': 'MEAN',\
     'nmin_stats': 10,\
     # Note that I changed xmin (ymin) and xmax (ymax) in IDL to x_range and
     # y_range, seemed more logical to me
     'x_range': [-15,15],\
     'y_range': [0,10],\
     'xy_res': [150,75],\
     'products': PRODUCT_DEFAULT,\
}

DATASET_DEFAULT['NETCDF_MXPOL'] = {
     'zdr_cal': {'czdr_apply': True},\
     'products': PRODUCT_DEFAULT,\
     'att_corr': ATTENUATION_CORRECTION_DEFAULT,\
     'hydro_class': HYDROMETEOR_CLASSIFICATION_DEFAULT,\
}

DATASET_DEFAULT['NETCDF_MXPOL_SPECTRA'] = {
     'zdr_cal': {'czdr_apply': True},\
     'products': PRODUCT_DEFAULT,\
     'att_corr': ATTENUATION_CORRECTION_DEFAULT,\
     'hydro_class': HYDROMETEOR_CLASSIFICATION_DEFAULT,\
}

DATASET_DEFAULT['NETCDF_PPI_RHI_COMPOSITE'] = {
     'ppi_angles': MANDATORY,\
     'rhi_angles' : MANDATORY,\
     'products': PRODUCT_DEFAULT,\
     'att_corr': ATTENUATION_CORRECTION_DEFAULT,\
     'hydro_class': HYDROMETEOR_CLASSIFICATION_DEFAULT,\
}


###############################################################################
# 5) CONFIGURATION FILES DEFAULT
CONFIG_DEFAULT={}

# 5.1) MAIN
CONFIG_DEFAULT['main']= {
    'campaign_name': MANDATORY,\
    'radar_name': 'MXPol',\
    'datapath_raw': MANDATORY,\
    'org_datapath_raw': 'YYYY/MM/DD',\
    'datapath_proc': None,\
    'datapath_proc2': None,\
    'save_img_path': None,\
    'save_img_path_2': None,\
}

# 5.2) LOCATION
CONFIG_DEFAULT['location'] = {
    'radar_position': {'latitude': 0, 'longitude': 0, 'altitude': 0,\
                       'hemisphere': 'N'},\
    'az_offset': 0,\
    'el_offset': 0,\
    'radar_beamwidth': 1.45,\
    'radar_beamwidth_tol': 0.2,\
    'radar_frequency': 9.41,\
    'radar_calibration_constant': 10.3,\
    'zdr_offset': 0,\
    'h_channel': 0,\
    'v_channel': 1,\
    'fft2i_bug': True,\
}

# 5.3) PRODUCTS
CONFIG_DEFAULT['products']= {
    'datasets': DATASET_DEFAULT,\
    'ppi_image_config': {'dx': 75,'dy': 75,'xmin': -40, 'xmax': 40,\
                            'ymin': -40,'ymax': -40},\

    'rhi_image_config': {'dx': 75,'dy': 75,'xmin': 5,'ymin': 40,'ymax': 40,\
                        'changeside': False },\
    'sector_scan_image_config': {'dx': 75,'dy': 75,'xmin': 5,'ymin': 40,\
                              'ymax': 40},\
    'time_height_image_config' : {'ymin': 0.5, 'ymax': 10},\
    'spectrogram_image_config': {'ymin': 0.5, 'ymax': 10},\
    'save_img': False,\
    'img_format': 'eps',\
    'convert_format': 'png',\
    'parallel': False
}
