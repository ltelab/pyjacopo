import pyart
import numpy as np
from pyjacopo import read_cfradial, write_cfradial
import glob
import scipy.io as sio
import datetime
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.facecolor']='white'
from ml_detection import detect_ml

"""
This script computes the hydrometeor classification with demixing (Besic et al. 2018 doi:10.5194/amt-9-4425-2016) on the ICE GENESIS dataset.
Most can be re-used for other datasets but some parts might be irrelevant.
NB this requires the meteoswiss version of pyart (pyart_mch)
"""

SNR_THR = -10.
RHOHV_THR = 0.6
ZDR_BIAS = 0.


# Load centroids once and for all, needed for hydrometeor classification
centroids = sio.loadmat('/ltedata/Prog_com/Lib_python/hydro_classif/MXPol_centroids/Centroids_sqeuc_averaged_X_c2.mat')['Centroid']
dummy_centroids = [99999, 99999, 99999, 99999, 99999]
centroids = np.insert(centroids,5,dummy_centroids,0)
# Then permute the order to make it corresponds to Jordi
perm = [1,0,2,4,3,5,6,8,7]
centroids = centroids[perm]  


def classify_hm(fname,date,rain_flag=0):
    """
    Function to run hydrometeor classification with demixing (Besic et al. 2018) using the implementation in pyart-mch
    """
	
    radar = read_cfradial(fname)

    # Define iso0 depending on the rain flag
    if rain_flag: 
        # if it rains, define iso0 as the top of the melting layer (detected using Wolfensberger and Berne 2018, see ml_detection.py script)
        ml_out, ml_obj, iso0_dict = detect_ml(radar)
        ml_obj.fields['MLHeight']['data'].data
        iso0 = np.nanmean(ml_obj.fields['MLHeight']['data'][:,1])
    else: 
        # if it snows, set the iso0 below the ground
        iso0 = -10
        
    # Create the field containing height over iso0
    height_over_iso0 = radar.get_gate_lat_lon_alt(0)[2] - iso0 # relative height above iso0, + means above, - below
    height_over_iso0_dic = {'long_name':'height above the zero degree celsius isotherm', 'units':'m','valid_min': -30000, 'valid_max': 30000, 'data': height_over_iso0}
    radar.add_field('height_over_iso0',height_over_iso0_dic)

    # Set values below SNR and Rhohv threshold to nans
    for v in radar.fields.keys()-{'sCC','sPowH','sPowV'}:
        V = radar.get_field(0, v).data
        V[np.logical_and(radar.get_field(0, 'SNRh')<SNR_THR, radar.get_field(0, 'Rhohv')<RHOHV_THR)] = np.nan              
        if v == 'Zdr' and ZDR_BIAS:
            V = V - ZDR_BIAS
        radar.add_field_like(v, v, V, replace_existing=True)
        
    # Create empty container for entropy
    radar.add_field('hydroclass_entropy', {'long_name':'Hydroclass entropy', 'units': '','valid_min':-1, 'valid_max':1, 'data':radar.fields['Zdr']['data']*np.nan})

    # Semisupervised classification from PyArt - MCH
    hydro_classif = pyart.retrieve.hydroclass_semisupervised(radar, refl_field='Zh', zdr_field='Zdr', rhv_field='Rhohv', 
                                                            kdp_field='Kdp', iso0_field='height_over_iso0', entropy_field='hydroclass_entropy', compute_entropy=True,
                                                            output_distances=True, mass_centers=centroids, temp_ref='height_over_iso0')
    
    
    # Add hydrometeor centroids to metadata
    radar.metadata['CentroidsAG'] = centroids[0,:]
    radar.metadata['CentroidsCR'] = centroids[1,:]
    radar.metadata['CentroidsLR'] = centroids[2,:]
    radar.metadata['CentroidsRP'] = centroids[3,:]
    radar.metadata['CentroidsRN'] = centroids[4,:]
    radar.metadata['CentroidsVI'] = centroids[5,:]
    radar.metadata['CentroidsWS'] = centroids[6,:]
    radar.metadata['CentroidsMH'] = centroids[7,:]
    radar.metadata['CentroidsIH'] = centroids[8,:]
    radar.metadata['CentroidsColumnNames'] = 'ZH, ZDR, KDP, RhoHV, height_above_iso0'

    # Information on the method
    radar.metadata['ClassificationMethod'] = 'Besic et al. AMT 2016 with demixing of Besic et al. AMT 2018'

    # Adding the hydrometeor classification to the radar object
    for i,key in enumerate(hydro_classif.keys()):
        radar.add_field(key,hydro_classif[key])

    for key in ['proportion_AG', 'proportion_CR', 'proportion_LR', 'proportion_RP', 'proportion_RN', 'proportion_VI', 'proportion_WS', 'proportion_MH', 'proportion_IH/HDG']:
        radar.fields[key]['units']='%'
    
    return radar


def plot_classification(radar,savepath='tmp',xlim=(-20,20),ylim=(0,10)):
    """ small function to plot the results - dominant hydrometeor class """
    import matplotlib as mpl
    radar.fields['hydro']['data'] = radar.fields['hydro']['data'].astype('float32')
    radar.fields['hydro']['data'][radar.fields['hydro']['data']>200.] = np.nan
    
    fig,ax = plt.subplots()
    display = pyart.graph.RadarDisplay(radar)
    cmap = mpl.colors.ListedColormap(["blue", "deepskyblue", "green", "gold", "orange", "red", 'magenta', 'purple','brown'])
    norm = mpl.colors.BoundaryNorm(np.arange(2,12), cmap.N) 
    cbarticks = np.arange(2,12)+.5
    legend = ['AG', 'CR', 'LR', 'RP', 'RN', 'VI', 'WS', 'MH', 'IH'] # AG: aggregates, CR: crystals, LR: light rain, RP: rimed particles, RN: rain, VI: vertically-aligned ice, WS: wet snow, MH: melting hail, IH: ice hail
    display.plot('hydro',cmap=cmap,norm=norm,ticks=cbarticks,ticklabs=legendVI)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    fig.savefig(savepath,dpi=300,bbox_inches='tight')
    

def find_rain_flag(rain_events_csv, date):    
    """
    Tailored for ICE GENESIS where we had a list of rain and snow events. Used to set height of iso0 below the ground when we know it is snowing.
    """
    with open(rain_events_csv) as f:
        lines = f.read().splitlines()

    rain_events = []
    for l in lines:
        lstart,lend = l.split(',')
        dtstart = datetime.datetime.strptime(lstart,'%Y%m%d%H%M')
        dtend = datetime.datetime.strptime(lend,'%Y%m%d%H%M')
        rain_events.append((dtstart,dtend))

    rain_flag = 0
    for r_e in rain_events:
        if ((date>r_e[0]) & (date<r_e[1])):
            rain_flag=1
            print('RAIN')
            break


if __name__=='__main__':
    
    for fname in sorted(glob.glob('/ltenas8/users/anneclaire/ICEGENESIS_2021/MXPol/Proc_data_v2/*.nc')):

        try:
            name = fname.split('/')[-1].split('_')[0]
            savepath = '/ltenas8/users/anneclaire/ICEGENESIS_2021/hydro_classif/figures_v3/'+name+'_hm_classif'
            outname = '/ltenas8/users/anneclaire/ICEGENESIS_2021/hydro_classif/cfradial_v3/'+name+'_hm_classif.nc'
            if os.path.exists(outname):
                print(outname, ' exists')
                continue

            date = datetime.datetime.strptime(name,'XPOL-%Y%m%d-%H%M%S')

            rain_flag = find_rain_flag('/ltenas8/users/anneclaire/ICEGENESIS_2021/rain_events.csv',date)
            radar_w_hm = classify_hm(fname,date,rain_flag=rain_flag)
            plot_classification(radar_w_hm,xlim=(-15,1),ylim=(0,6),savepath=savepath)

            # there is a conflict with some fields having a '_FillValue' fields and others not, so we do:
            for key in radar_w_hm.fields.keys():
                if '_FillValue' in radar_w_hm.fields[key].keys():
                    del radar_w_hm.fields[key]['_FillValue']

            write_cfradial(outname,radar_w_hm)
            print(outname, ' created')
        except Exception as e:
            print(outname,e)
