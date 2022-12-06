# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:13:37 2017

@author: wolfensb
"""

import copy
import yaml
import builtins
import re
import numpy as np

from .valid_input import Range, TypeList, CONFIG_VALID
from .default_input import MANDATORY, CONFIG_DEFAULT

def check_validity(input_value,valid_value):
    flag_valid = False
    if type(input_value) == list:
        # If input is a list, check all elements in the list
        flag_valid = all([check_validity(i,valid_value) for i in input_value])
    else:
        # Check if valid value is a type
        if type(valid_value) == builtins.type:
            flag_valid = type(input_value) == valid_value
        # Check if valid value isa Range
        elif type(valid_value) == Range:
            flag_valid = input_value in valid_value
        elif type(valid_value) == list:
            if type(valid_value[0]) == builtins.type:
                # case 1 list of types
                flag_valid = type(input_value) in valid_value
            else:
                # case 2 list of values
                flag_valid = input_value in valid_value
        # Check if valid value is a string with a regex
        elif type(valid_value) == str and valid_value[0:5] == '-reg-':
            # See if input matches regex (the \Z is used to match end of string)
            if re.match(valid_value[5:]+'\Z',input_value):
                flag_valid = True
        # Last possibility is TypeList
        elif type(valid_value) == TypeList:
            flag_valid = valid_value == input_value
        else:
            # Last possibility is that valid_value is a single value
            flag_valid = valid_value == input_value
    return flag_valid


'''
parse_config(input_values, default_values, valid_values, history = '')

description: recursively parses a structure (either given as input or read
             from a yaml file), assigns default values if needed and check
             validity of given values

inputs:
    input_values: either a filename or a structure to be parsed
    default_values: dictionary of default values for the structure or file to
                    parse (defined in defaults_inputs.py)
    valid_values:   dictionary of valid values for the structure or file to
                    parse (defined in valid_input.py)
'''

def parse_config(input_values, default_values = CONFIG_DEFAULT, 
                 valid_values = CONFIG_VALID,verbose=False):
    if verbose:
        print('Reading provided configuration file')
    
    parsed, hist = rec_parse(input_values,default_values,valid_values,verbose=verbose)
    if verbose:
        print('Reading finished')
    return parsed

def rec_parse(input_values, default_values, valid_values, history = '',verbose=False):
    '''history is used to backtrace the full name of the key, it
       it should always be left to default , when first called'''

    # First we need to create a local copy of the history
    local_history = history

    # Check if input is a file name (a string)
    if type(input_values) == str and history == '':
        try:
            input_values = yaml.safe_load(open(input_values,'r'))
        except:
            raise IOError('Could not read yaml file from provided filename')
    parsed_inputs = copy.deepcopy(input_values)

    # Start parsing
    # if the given value is a dictionary, recursively parse
    if type(default_values) == dict:

        for k in default_values.keys():
            optional = False
            if '*' in k:
                optional = True
                k = k.replace('*','')
            try:
                new_history = local_history + '/' +k

                if optional:
                    defaults = default_values['*'+k]
                else:
                    defaults = default_values[k]

                parsed, history = rec_parse(input_values[k],defaults,
                                  valid_values[k],new_history)
                parsed_inputs[k] = parsed
            except:
                # Assign defaults
                if not optional:
                    # Assign the default only if non optional input
                    parsed_inputs[k] = default_values[k]
                else:
                    # Assign NaN
                    parsed_inputs[k] = np.nan
            ''' Here we handle the special case of the datasets and products
            --> we remove all datasets/products  that were not explictely given
            by the users i.e. we remove the defaults for all datasets not
            wanted by user'''

            if k == 'datasets' or k == 'products':
                topop = [k2 for k2 in parsed_inputs[k].keys() if \
                    k2 not in input_values[k].keys()]

                for k2 in topop:
                    parsed_inputs[k].pop(k2)



    # if the given input is not a dict, check the validity
    else:
        # Now check for the validity of the given value
        if not check_validity(input_values,valid_values):
            if verbose:
                print('Invalid value for key : '+ history)
            if default_values == MANDATORY:
                raise ValueError('Input "' +history+'" is mandatory, you '+
                                 ' have to provide a valid value!')
            else:
#                print('Assigning default value: '+ str(default_values))
                parsed_inputs = default_values

    return parsed_inputs, local_history
