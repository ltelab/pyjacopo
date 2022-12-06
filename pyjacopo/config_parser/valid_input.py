# -*- coding: utf-8 -*-
import numpy as np
import builtins


# The Range class extends the behaviour of python range() to floats
class Range(object):
    def __init__(self,x,y):
        if type(x) != type(y):
            raise ValueError('range bounds are not of the same type!')
        if y <= x:
            raise ValueError('Lowe bound is larger than upper bound!')
        self.x = x
        self.y = y
        self.type = type(x)
    def __contains__(self,z):
        return type(z) == self.type and z <= self.y and z >= self.x

def generic_type(value):
    type_ = type(value)
    if type_ == np.int64:
        type_ = int
    elif type_ == np.float or type_ == np.float64 or type_ == np.float128:
        type_ = float
    elif type_ == np.str_:
        type_ = str
    return type_

# The TypeList class is used to check if a given array or list is of appropriate
# dimension and type

class TypeList(object):
    def __init__(self,type_,dim = []):
        # Note that dim = [], is used to indicate an arbitrary length
        if type(type_) != builtins.type:
            raise(ValueError('Specified type is invalid!'))
        if any([d<0 for d in dim]):
            raise(ValueError('Specified dimension is invalid (<0)!'))

        self.type_ = type_
        self.dim = dim
    def __eq__(self, array):
        flag = False
        try:
            array = np.array(array)
            dim = array.shape
            # Check if dimensions are ok
            if len(self.dim): # Check only if dim != []
                flag = all([d1 == d2 for d1,d2 in zip(dim,self.dim)])
            else:
                flag = True
            # Check if all types are ok
            flag *= all([generic_type(v) == self.type_ for v in array.ravel()])
        except:
            pass
        return flag


'''
 Standard

 The following assumptions are made:
 If the valid value given below is :
 a type :  the input must be of that type (ex int, int,float, \
                                             list, np.ndarray...)
 a value: the input must be that value
 a list:  the input must match with the elements in the list (i.e. be of the \
          right type or the right value)
 '-reg-' followed by a string: the input must be a string that matches with
                               the corresponding regex
  Range(x,y) : the input must of the type x and y and be in the range between
               x and y
  TypeList(type,dim):  the input must be a list (or array) with the appropriate
                       dimensions (dim) and type (type), if no dimension is pro
                       vided, the input is allowed to have any arbitrary dim

 Note that when floats are needed, int will be accepted as well, i.e.
 both 2.0 and 2 are valid. This is however NOT reciprocal.
'''



###############################################################################
# 1) PRODUCTS VALID
# Note that some fields are used only for certain types of products

PRODUCT_VALID = {}

PRODUCT_VALID['PPI_IMAGE'] = {
    'variables': TypeList(str), \
    'continuous_cb':  [0,1],\
    'vmin': TypeList(float),\
    'vmax': TypeList(float),\
}

PRODUCT_VALID['RHI_IMAGE'] = {
    'variables': TypeList(str), \
    'continuous_cb':  [0,1],\
}

PRODUCT_VALID['PPI_RHI_IMAGE'] = {
    'variables': TypeList(str), \
    'continuous_cb':  [0,1],\
}

PRODUCT_VALID['PPI_RHI_3D_COMP'] = {
    'variables': TypeList(str), \
    'continuous_cb':  [0,1],\
    'views': [1,2,3,4],\
}

PRODUCT_VALID['TIMEHEIGHT_IMAGE'] = {
    'variables': TypeList(str), \
    'continuous_cb':  [0,1],\
    'res': TypeList(int,[2]),\
    'rebin_seconds':  [0,1],\
}

PRODUCT_VALID['SPECTROGRAM_IMAGE'] = {
    'variables': TypeList(str), \
    'continuous_cb':  [0,1],\
    'stacked_spectra':  [0,1],\
}

PRODUCT_VALID['NETCDF_POLAR'] = {
    'add_fields': TypeList(str),\
    'nanval': float,\
}

PRODUCT_VALID['NETCDF_POLAR_SPECTRA'] = {
    'add_fields': TypeList(str),\
    'nanval': float,\
}

PRODUCT_VALID['NETCDF_PROFILE_SPECTRA'] = {
}

PRODUCT_VALID['NETCDF_TIMEHEIGHT_PROFILE'] = {
}

###############################################################################
# 2) PROCESSING MODULES VALID

# 2.1) Kdp estimation
PHIDP_KDP_VALID = {
    'phidp_method' : ['MEDIAN','NONE','KF','MEDIAN2W','MULTISTEP','HUBBER1995_MOD'],\
    'kdp_method' : ['DERIVATION','DERIVATION2','LINFIT','LINFIT2','LINFIT3','MULTISTEP',\
    'LEASTSQUARE','LEASTSQUARE2','KF'],\
    'int_kdp' : int,\
    'unwrap' : [0,1],\
    'jitter': float,\
    'offset_calc' : int,\
    'offset_ngates': int,\
    'point_int_remove': int,\
    'snr_th': float,\
    'rhohv_th': float,\
    'phidp_niterations': int,\
    'kdp_niterations': int,\
    # VALID for rcov and pcov are provided in appropriate part of the code
    'rcov_file': [list,np.ndarray],\
    'pcov_file': [list,np.ndarray],\
    'phidp_windsize': float,\
    'phidp_windsize_s': float,\
    'phidp_windsize_l': float, \
    'phidp_zh_min': float,\
    'kdp_windsize':float,\
    'kdp_windsize_s': float,\
    'kdp_windsize_l': float,\
    'kdp_windsize_i': float,\
    'kdp_zh_min': float,\
    'kdp_zh_min_2': float
}

# 2.2) Clutter filter
CLUTTER_FILTER_VALID = {
    'method': ['NOTCH'],\
    'pre_filter': int,\
    'vel_th': float,\
    'use_mag': int,\
}

# 2.3) Attenuation correction
ATTENUATION_CORRECTION_VALID = {
    'method': ['ZPHI','PHILINEAR','HBORDAN'],\
}

# 2.4) TODO - Hydrometeor classification
HYDROMETEOR_CLASSIFICATION_VALID = {}

###############################################################################
# 3) PROCESSING VALID

# 3.1) PPI/RHI/SectorScan data
PROCESSING_STANDARD_VALID={
    'czdr_apply': int,\
    'snr_pow': float,\
    'snr_phase': float,\
    'rhohv_th':  [0,1],\
    'discard_spectra': int,\
    'discard_signal': int,\
    'mask_out': {'snr':float,'rhohv':float},\
    'phidp_kdp': PHIDP_KDP_VALID,\
    'att_corr': ATTENUATION_CORRECTION_VALID,\
    'rhohv_snr_correction': [0,1],\
}


# 3.2) Spectral data
PROCESSING_SPECTRA_VALID={
    'notch_spectra': int,\
    'dealias_spectra': int,\
    'time_average': float,\
    'rhohv_th': float,\
    'snr_th': float,\
    'clutter_filter': CLUTTER_FILTER_VALID,\
    # DSD retrieval not in IDL framework yet...
    'dsd_retrieval': dict,\
}



###############################################################################
# 4) DATASETS VALID

DATASET_VALID = {}

DATASET_VALID['PPI'] = {
    'angles': list,\
    'nang_max': [float,int],\
    'nang_min': [float,int],\
    'ang_tol': [float,int],\
    'range_min': [float,int],\
    'processing': PROCESSING_STANDARD_VALID,\
    'products': PRODUCT_VALID,\
    'image': int
}

DATASET_VALID['RHI'] = {
    'angles': list,\
    'nang_max': float,\
    'nang_min': float,\
    'ang_tol': float,\
    'range_min': float,\
    'processing': PROCESSING_STANDARD_VALID,\
    'products': PRODUCT_VALID,\
    'image': int
}

DATASET_VALID['SECTOR_SCAN'] = {
    'angles': list,\
    'nang_max': float,\
    'nang_min': float,\
    'ang_tol': float,\
    'range_min': float,\
    'processing': PROCESSING_STANDARD_VALID,\
    'products': PRODUCT_VALID,\
    'image': int
}

DATASET_VALID['WINDOW'] = {
    'ds_type' : str,\
     # nang_max default is to be adjusted at runtime (nb of records in raw file)
    'nang_max': float,\
    'nang_min': float,\
    'min_az': float,\
    'min_el': float,\
    'processing': PROCESSING_STANDARD_VALID,\
    'products': PRODUCT_VALID,\
}

DATASET_VALID['PROFILE'] = {
    'angles': list,\
    'nang_max': float,\
    'nang_min': float,\
    'ang_tol': float,\
    'range_min': float,\
    'processing': PROCESSING_STANDARD_VALID,\
    'products': PRODUCT_VALID,\
    'image': True
}

DATASET_VALID['PROFILE_NO_PEDESTAL'] = {
    'angles': list,\
    'az_angles': TypeList(float),\
    'nang_max': float,\
    'nang_min': float,\
    'ang_tol': float,\
    'range_min': float,\
    'processing': PROCESSING_STANDARD_VALID,\
    'products': PRODUCT_VALID,\
    'image': True
}

DATASET_VALID['SPECTRA_PROFILE']  = {
    'input_complement': str,\
    'processing': PROCESSING_SPECTRA_VALID,\
    'products': PRODUCT_VALID,
}

# TODO: SPECTRA_RHI_DATA

DATASET_VALID['TIMEHEIGHT']  = {
    'input_complement': str,\
    'products': PRODUCT_VALID,
}

DATASET_VALID['TIME_SUMMARY_HEIGHT']  = {
     'input_complement': str,\
     'variables': list,\
     'max_ele': float,\
     'stats':['MEAN','STDDEV','MIN','MAX','-reg-Q[0-9]{3}'],\
     'nmin_stats': int,\
     # Note that I changed xmin (ymin) and xmax (ymax) in IDL to x_range and
     # y_range, seemed more logical to me
     'x_range': TypeList(float),\
     'y_range': TypeList(float),\
     'xy_res': TypeList(float),\
     'products': PRODUCT_VALID,\
}

DATASET_VALID['NETCDF_MXPOL']  = {
     'zdr_cal': {'czdr_apply': True},\
     'products': PRODUCT_VALID,\
     'att_corr': ATTENUATION_CORRECTION_VALID,\
     'hydro_class': dict,\
}

DATASET_VALID['NETCDF_MXPOL_SPECTRA']  = {
     'zdr_cal': {'czdr_apply': int},\
     'products': PRODUCT_VALID,\
     'att_corr': ATTENUATION_CORRECTION_VALID,\
     'hydro_class': dict,\
}

DATASET_VALID['NETCDF_PPI_RHI_COMPOSITE']  = {
     'ppi_angles': list,\
     'rhi_angles' : list,\
     'products': PRODUCT_VALID,\
     'att_corr': ATTENUATION_CORRECTION_VALID,\
     'hydro_class': dict,\
}



###############################################################################
# 5) CONFIGURATION FILES VALID
CONFIG_VALID={}

# 5.1) MAIN
CONFIG_VALID['main']= {
    'campaign_name': str,\
    'radar_name': str,\
    'datapath_raw': str,\
    'org_datapath_raw': ['YYYY/MM/DD','YYYY/DD/MM'],\
    'datapath_proc': str,\
    'datapath_proc2': str,\
    'save_img_path': str,\
    'save_img_path_2': str,\
}

# 5.2) LOCATION
CONFIG_VALID['location'] = {
    'radar_position': {'latitude': float, 'longitude': float, 'altitude': float,\
                       'hemisphere': ['N','S']},\
    'az_offset': float,\
    'el_offset': float,\
    'radar_beamwidth': float,\
    'radar_beamwidth_tol': float,\
    'radar_frequency': float,\
    'radar_calibration_constant': float,\
    'zdr_offset': float,\
    'h_channel': int,\
    'v_channel': int,\
    'fft2i_bug': int,\
}

# 5.3) PRODUCTS
CONFIG_VALID['products']= {
    'datasets': DATASET_VALID,\
    'ppi_image_config': {'dx': float,'dy': float,'xmin': float, 'xmax': float,\
                            'ymin': float,'ymax': float},\

    'rhi_image_config': {'dx': float,'dy': float,'xmin': float,'ymin': float,'ymax': float,\
                        'changeside': int },\
    'sector_scan_image_config': {'dx': float,'dy': float,'xmin': float,'ymin': float,\
                              'ymax': float},\
    'time_height_image_config' : {'ymin': float, 'ymax': float},\
    'spectrogram_image_config': {'ymin': float, 'ymax': float},\
    'save_img': False,\
    'img_format': ['eps','jpg','png'],\
    'convert_format': ['png','eps','jpg'],\
    'parallel': int
}



