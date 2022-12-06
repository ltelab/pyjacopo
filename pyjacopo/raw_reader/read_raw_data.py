"""
Created on Mon Jan  9 11:33:45 2017

@author: wolfensb
"""
import struct
import numpy as np
import time

from .file_structure_info import HEADER, RECORD_HEADER, RECORD_DATA, \
                                BYTE_SIZES, SCAN_TYPES

# not used ??
# NP_TYPES = {'f':np.float,'d':np.float64}


def _get_header(ba):
    # Read the file byte by byte
    values = {}
    read_pos = 0
    for i in range(len(HEADER['names'])):
        len_val = HEADER['len'][i]
        type_var = HEADER['type'][i]
        ffmt = '<%d%s'%(len_val,type_var)

        offset = len_val * BYTE_SIZES[type_var]
        val = struct.unpack_from(ffmt,ba[read_pos:read_pos+offset])

        if len(val) == 1:
            val = val[0]
        else:
            val = np.array(val)
        values[HEADER['names'][i]] = val
        read_pos += offset

    #convert the null terminated bytes to string
    values['site']=values['site'].decode().rstrip('\0')

    return values, ba[read_pos:]

def _get_record_header(ba):
    values = {}
    read_pos = 0
    for i in range(len(RECORD_HEADER['names'])):

        len_val = RECORD_HEADER['len'][i]

        type_var = RECORD_HEADER['type'][i]
        ffmt = '<%d%s'%(len_val,type_var)

        offset = len_val * BYTE_SIZES[type_var]
        val = struct.unpack_from(ffmt,ba[read_pos:read_pos+offset])

        if len(val) == 1:
            val = val[0]
        else:
            val = np.array(val)

        values[RECORD_HEADER['names'][i]] = val

        read_pos += offset

    return values, ba[read_pos:]

def _get_record_data(ba,head):
    values = {}
    read_pos = 0
    if head['servmode'] == 0:
        if head['sumpower'] == 1:
            mode = 'PPSP'
        else:
            mode = 'PP'
    elif head['servmode'] == 1:
        if head['sumpower'] == 1:
            mode = 'DPPSP'
        else:
            mode = 'DPP'
    elif head['servmode'] == 2:
        mode = 'FFT'
    elif head['servmode'] == 3:
        mode = 'FFT2'
    elif head['servmode'] == 4:
        mode = 'FFT2I'
    record_data_mode = RECORD_DATA[mode]

    for i in range(len(record_data_mode['names'])):
        multi_dim = False # Flag to know if array is multidimensional

        len_val = record_data_mode['len'][i]

        if type(len_val) is list: # len_val is a list
            multi_dim = True
            dim = []
            for el in len_val:
                if type(el) == str:
                    dim.append(head[el])
                else:
                    dim.append(el)
            dim = np.array(dim)
            len_val = np.prod(dim)


        type_var = record_data_mode['type'][i]
        
        if type_var != 'cx':
            ffmt = '<%d%s'%(len_val,type_var)
        else:
            # Complex numbers are not a struct type so we handle them as pair of floats
            ffmt = '<%df'%(len_val*2)

        offset = len_val * BYTE_SIZES[type_var]
        val = struct.unpack_from(ffmt,ba[read_pos:read_pos+offset])

        val = np.array(val)
        if type_var == 'cx':
            val.dtype = complex

        if multi_dim:
            val = np.squeeze(np.reshape(val,dim))

        values[record_data_mode['names'][i]] = val
        read_pos += offset

    return values, ba[read_pos:]


def _split_at_tol(records, config = None):

    options = {}
    if not config:
        options['RHI'] = {'ang_tol':0.3,'min_nb':10}
        options['PPI'] = {'ang_tol':0.3,'min_nb':10}
    else:
        # use user provided values
        options['RHI'] = {'ang_tol':config['products']['datasets']['RHI']['ang_tol'],
               'min_nb':config['products']['datasets']['RHI']['nang_min']}
        options['PPI'] = {'ang_tol':config['products']['datasets']['PPI']['ang_tol'],
               'min_nb':config['products']['datasets']['PPI']['nang_min']}

    if 'RHI' in records.keys():
        all_az = np.array([rr['az'] for rr in records['RHI']['header']])
        az_diff = np.abs(np.diff(all_az))

        all_splits = np.where(az_diff > options['RHI']['ang_tol'])[0]

        if len(all_splits):
            all_splits += 1
            all_splits = np.insert(all_splits,0,0)
            all_splits = np.append(all_splits,len(all_az))

            idx_valid_split = 0
            for idx in range(len(all_splits)-1):
                len_split = all_splits[idx+1] - all_splits[idx]
                if len_split > options['RHI']['min_nb']:
                    idx_valid_split += 1
                    # Add new category
                    cat_name = 'RHI'+str(idx_valid_split)
                    # Split
                    records[cat_name] = {}
                    records[cat_name]['header'] = records['RHI']['header'][all_splits[idx]:all_splits[idx+1]]
                    records[cat_name]['data'] = records['RHI']['data'][all_splits[idx]:all_splits[idx+1]]
            # Remove original key
            records.pop('RHI')

            if idx_valid_split == 1: # If only one key 'RHI1', rename it to 'RHI'
                records['RHI'] = records.pop('RHI1')

    # Same for PPI, ok it's not the best coding feel free to improve
    if 'PPI' in records.keys():
        all_elev = np.array([rr['el'] for rr in records['PPI']['header']])
        elev_diff = np.abs(np.diff(all_elev))

        all_splits = np.where(elev_diff > options['PPI']['ang_tol'])[0]

        if len(all_splits):
            all_splits += 1
            all_splits = np.insert(all_splits,0,0)
            all_splits = np.append(all_splits,len(all_elev))

            idx_valid_split = 0

            for idx in range(len(all_splits)-1):
                len_split = all_splits[idx+1] - all_splits[idx] + 1

                if len_split > options['PPI']['min_nb']:
                    idx_valid_split += 1
                    # Add new category
                    cat_name = 'PPI'+str(idx_valid_split)
                    # Split
                    records[cat_name] = {}
                    records[cat_name]['header'] = records['PPI']['header'][all_splits[idx]:all_splits[idx+1]]
                    records[cat_name]['data'] = records['PPI']['data'][all_splits[idx]:all_splits[idx+1]]
            # Remove original key
            records.pop('PPI')

            if idx_valid_split == 1: # If only one key 'PPI', rename it to 'PPI1'
                records['PPI'] = records.pop('PPI1')

    return records

def read_raw_data(filename,  config = None,verbose=False):

    '''
    def read_raw_data(filename, config)
    INPUTS:
        filename : name of the raw data (.dat) file to be read
        ang_tol : angular tolerance for the separation of different PPI (on elevation)
                  or different RHI (on azimuth)
        config: user config as obtained by parse, if not present defaults will
            be used
    '''

    with open(filename, 'rb') as f:
        ba = memoryview(bytearray(f.read())) # bytearray

    ''' Read the binary file'''
    if verbose:
        print('Reading raw data file...')
    head,rem = _get_header(ba)

    all_records = {}
    while len(rem):
        rh, rem = _get_record_header(rem)
        rd, rem = _get_record_data(rem,head)

        scan_type = SCAN_TYPES[rh['scan_type']]

        if scan_type not in all_records.keys():
            all_records[scan_type] = {'header':[],'data':[]}

        # Append to all_records
        all_records[scan_type]['header'].append(rh)
        all_records[scan_type]['data'].append(rd)
    if verbose:
        print('Reading finished...')
    all_records = _split_at_tol(all_records, config)

    return head, all_records


if __name__ == '__main__':
    start_time = time.time()
    head, all_records = read_raw_data('/ltedata/HYMEX/SOP_2012/Radar/Raw_data/2012/09/26/XPOL-20120926-075336.dat')
    print("--- %s seconds ---" % (time.time() - start_time))
#    aa = records['RHI1']['data'][20]['power-spectra']
#    a = np.reshape(aa,[4,528,64])
#    b = np.nansum(a[2],axis=1)
#    import matplotlib.pyplot as plt
#    plt.plot(b)
#
