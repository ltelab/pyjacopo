# from pyart.retrieve.ml import detect_ml
import pyart
from copy import deepcopy
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline, pchip
from scipy.interpolate import RegularGridInterpolator
from pyart.config import get_metadata

MAXTHICKNESS_ML = 1000
MAXHEIGHT_ML = 6000.
MINHEIGHT_ML = 100.
LOWMLBOUND = 0.7
UPMLBOUND = 1.3
SIZEFILT_M = 75
ZH_IM_BOUNDS = (10, 60)
RHOHV_IM_BOUNDS = (0.75, 1)
RHOHV_VALID_BOUNDS = (0.6, 1)
KE = 4 / 3.  # Constant in the 4/3 earth radius model
# Two extreme earth radius
R_EARTH_MAX = 6378.1370 * 1000
R_EARTH_MIN = 6356.7523 * 1000

def polar_to_cartesian(radar_sweep, field_name, cart_res=25,
                       max_range=None, mapping=None):

    KE = 4 / 3.  # Constant in the 4/3 earth radius model
    # Two extreme earth radius
    R_EARTH_MAX = 6378.1370 * 1000
    R_EARTH_MIN = 6356.7523 * 1000
    
    # Get data to be interpolated
    pol_data = radar_sweep.get_field(0, field_name)
    is_ppi = False

    if mapping:
        # Check if mapping is usable:
        if is_ppi != mapping['is_ppi']:
            print('Input mapping does not correspond to given scan type, ignoring it')
            mapping = None
        elif mapping['dim_pol'] != pol_data.shape:
            print('Input mapping does not correspond to dimensions of given field'
                  ', ignoring it')
            mapping = None
        else:
            cart_res = mapping['res']
            max_range = mapping['max_range']

    # Get distances of radar data
    r = radar_sweep.range['data']

    if max_range is None:
        max_range = np.max(r)

    # Cut data at max_range
    pol_data_cut = deepcopy(pol_data[:, r < max_range])
    r = r[r < max_range]

    # Set masked pixels to nan
    pol_data_cut.filled(np.nan)

    # One specificity of using the kd-tree is that we need to pad the array
    # with nans at large ranges and angles smaller and larger
    pol_data_cut = np.pad(pol_data_cut, pad_width=((1, 1), (0, 1)),
                          mode='constant', constant_values=np.nan)

    # Get angles of radar data
    if is_ppi:
        theta = radar_sweep.azimuth['data']
    else:
        theta = radar_sweep.elevation['data']

    # We need to pad theta and r as well
    theta = np.hstack([np.min(theta) - 0.1, theta, np.max(theta) + 0.1])
    r = np.hstack([r, np.max(r) + 0.1])

    r_grid_p, theta_grid_p = np.meshgrid(r, theta)

    # Generate regular cartesian grid
    if is_ppi:
        x_vec = np.arange(-max_range - cart_res,
                          max_range + cart_res, cart_res)
        y_vec = np.arange(-max_range - cart_res,
                          max_range + cart_res, cart_res)
    else:
        x_vec = np.arange(min(
            [(max_range-cart_res)*np.cos(np.radians(np.max(theta))), 0]),
                          max_range+cart_res, cart_res)

        y_vec = np.arange(0, max_range + cart_res, cart_res)

    x_grid_c, y_grid_c = np.meshgrid(x_vec, y_vec)

    if is_ppi:
        theta_grid_c = np.degrees(np.arctan2(-x_grid_c, -y_grid_c) + np.pi)
        r_grid_c = (np.sqrt(x_grid_c**2 + y_grid_c**2))
    else:
        theta_grid_c = np.degrees(-(np.arctan2(x_grid_c,
                                               y_grid_c) - np.pi / 2))
        E = pyart.map.get_earth_radius(radar_sweep.latitude['data'])
        r_grid_c = (np.sqrt((E * KE * np.sin(np.radians(theta_grid_c)))**2 +
                            2 * E * KE * y_grid_c + y_grid_c ** 2)
                    - E * KE * np.sin(np.radians(theta_grid_c)))

    if not mapping:
        # Kd-tree construction and query
        kdtree = spatial.cKDTree(np.vstack((r_grid_p.ravel(),
                                            theta_grid_p.ravel())).T)
        _, mapping_idx = kdtree.query(np.vstack((r_grid_c.ravel(),
                                                 theta_grid_c.ravel())).T, k=1)

        mapping = {'idx': mapping_idx, 'max_range': max_range, 'res': cart_res,
                   'is_ppi': is_ppi, 'dim_pol': pol_data.shape}

    cart_data = pol_data_cut.ravel()[mapping['idx']]
    cart_data = np.reshape(cart_data, x_grid_c.shape)

    return (x_vec, y_vec), cart_data, mapping

def get_earth_radius(latitude):
    """
    Computes the earth radius for a given latitude

    Parameters
    ----------
    latitude: latitude in degrees (WGS84)

    Returns
    -------
    earth_radius : the radius of the earth at the given latitude
    """
    a = R_EARTH_MAX
    b = R_EARTH_MIN
    num = ((a ** 2 * np.cos(latitude)) ** 2 +
           (b ** 2 * np.sin(latitude)) ** 2)
    den = ((a * np.cos(latitude)) ** 2 +
           (b * np.sin(latitude)) ** 2)

    earth_radius = np.sqrt(num / den)

    return earth_radius


def r_to_h(earth_radius, gate_range, gate_theta):
    '''
    Computes the height of radar gates knowing the earth radius at the given
    latitude and the range and elevation angle of the radar gate.

    Inputs:
        earth_radius : the radius of the earth for a given latitude in m.

        gate_range : the range of the gate(s) in m.

        gate_theta : elevation angle of the gate(s) in degrees.

    Outputs:
        height : the height above ground of all specified radar gates
    '''

    height = ((gate_range**2 + (KE * earth_radius)**2 +
               2 * gate_range * KE * earth_radius *
               np.sin(np.deg2rad(gate_theta)))**(0.5) - KE * earth_radius)

    return height

def remap_to_polar(radar_sweep, x, bottom_ml, top_ml, tol=1.5, interp=True):
    '''
    This routine converts the ML in Cartesian coordinates back to polar
    coordinates.

    Inputs:
        radar_sweep : Radar
            A pyart radar instance containing the radar data in polar
            coordinates for a single sweep
        x: array of floats
            The horizontal distance in Cartesian coordinates.
        bottom_ml: array of floats
            Bottom of the ML detected in Cartesian coordinates.
        top_ml: array of floats
            Top of the ML detected on Cartesian coordinates.
        tol : float, optional
            Angular tolerance in degrees that is used when mapping elevation
            angles computed on the Cartesian image to the original angles in
            the polar data.
        interp : bool, optional
            Whether or not to interpolate the ML in polar coordinates (fill holes)

    Outputs:
        (theta, r) : tuple of elevation angle and range corresponding to the
                     polar coordinates
        (bottom_ml, top_ml) : tuple of ml bottom and top ranges for every
                              elevation angle theta
        map_ml_pol : a binary map of the ML in polar coordinates
    '''
    # This routine converts the ML in cartesian coordinates back to polar
    # coordinates

    # Get ranges of radar data
    r = radar_sweep.range['data']
    dr = r[1]-r[0]

    # Get angles of radar data
    theta = radar_sweep.elevation['data']

    # Vectors to store the heights of the ML top and bottom and matrix for the
    # map
    map_ml_pol = np.zeros((len(theta), len(r)))
    bottom_ml_pol = np.zeros(len(map_ml_pol)) + np.nan
    top_ml_pol = np.zeros(len(map_ml_pol)) + np.nan

    if np.sum(np.isfinite(bottom_ml)) > 0:
         # Convert cartesian to polar

        # Get ranges of all pixels located at the top and bottom of cartesian
        # ML
        theta_bottom_ml = np.degrees(-(np.arctan2(x, bottom_ml) - np.pi / 2))
        E = get_earth_radius(radar_sweep.latitude['data'])  # Earth radius
        r_bottom_ml = (np.sqrt((E * KE * np.sin(np.radians(theta_bottom_ml)))**2 +
                               2 * E * KE * bottom_ml + bottom_ml ** 2)
                       - E * KE * np.sin(np.radians(theta_bottom_ml)))

        theta_top_ml = np.degrees(- (np.arctan2(x, top_ml) - np.pi / 2))
        E = get_earth_radius(radar_sweep.latitude['data'])  # Earth radius
        r_top_ml = (np.sqrt((E * KE * np.sin(np.radians(theta_top_ml))) ** 2 +
                            2 * E * KE * top_ml + top_ml ** 2) -
                    E * KE * np.sin(np.radians(theta_top_ml)))

        idx_r_bottom = np.zeros((len(theta))) * np.nan
        idx_r_top = np.zeros((len(theta))) * np.nan

        for i, t in enumerate(theta):
            # Find the pixel at the bottom of the ML with the closest angle
            # to theta
            idx_bot = np.nanargmin(np.abs(theta_bottom_ml - t))

            if np.abs(theta_bottom_ml[idx_bot] - t) < tol:
                # Same with pixel at top of ml
                idx_top = np.nanargmin(np.abs(theta_top_ml - t))
                if np.abs(theta_top_ml[idx_top] - t) < tol:

                    r_bottom = r_bottom_ml[idx_bot]
                    r_top = r_top_ml[idx_top]

                    idx_aux = np.where(r >= r_bottom)[0]
                    if idx_aux.size > 0:
                        idx_r_bottom[i] = idx_aux[0]

                    idx_aux = np.where(r >= r_top)[0]
                    if idx_aux.size > 0:
                        idx_r_top[i] = idx_aux[0]
        if interp:
            if np.sum(np.isfinite(idx_r_bottom)) >= 4:
                idx_valid = np.where(np.isfinite(idx_r_bottom))[0]
                idx_nan = np.where(np.isnan(idx_r_bottom))[0]
                bottom_ml_fill = InterpolatedUnivariateSpline(
                    idx_valid, idx_r_bottom[idx_valid], ext=1)(idx_nan)
                bottom_ml_fill[bottom_ml_fill == 0] = -9999
                idx_r_bottom[idx_nan] = bottom_ml_fill

            if np.sum(np.isfinite(idx_r_top)) >= 4:
                idx_valid = np.where(np.isfinite(idx_r_top))[0]
                idx_nan = np.where(np.isnan(idx_r_top))[0]
                top_ml_fill = InterpolatedUnivariateSpline(
                    idx_valid, idx_r_top[idx_valid], ext=1)(idx_nan)
                top_ml_fill[top_ml_fill == 0] = -9999
                idx_r_top[idx_nan] = top_ml_fill
        else:
            idx_r_bottom[np.isnan(idx_r_bottom)] = -9999
            idx_r_top[np.isnan(idx_r_top)] = -9999

        idx_r_bottom = idx_r_bottom.astype(int)
        idx_r_top = idx_r_top.astype(int)

        for i in range(len(map_ml_pol)):
            if idx_r_bottom[i] != -9999 and idx_r_top[i] != -9999:
                r_bottom_interp = min([len(r), idx_r_bottom[i]])*dr
                bottom_ml_pol[i] = r_to_h(E, r_bottom_interp, theta[i])

                r_top_interp = min([len(r), idx_r_top[i]])*dr
                top_ml_pol[i] = r_to_h(E, r_top_interp, theta[i])

                # check that data has plausible values
                if (bottom_ml_pol[i] > MAXHEIGHT_ML or
                        bottom_ml_pol[i] < MINHEIGHT_ML or
                        top_ml_pol[i] > MAXHEIGHT_ML or
                        top_ml_pol[i] < MINHEIGHT_ML or
                        bottom_ml_pol[i] >= top_ml_pol[i]):
                    bottom_ml_pol[i] = np.nan
                    top_ml_pol[i] = np.nan
                else:
                    map_ml_pol[i, 0:idx_r_bottom[i]] = 1
                    map_ml_pol[i, idx_r_bottom[i]:idx_r_top[i]] = 3
                    map_ml_pol[i, idx_r_top[i]:] = 5

    return (theta, r), (bottom_ml_pol, top_ml_pol), map_ml_pol



def create_ml_obj(radar, ml_pos_field='melting_layer_height'):
    """
    Creates a radar-like object that will be used to contain the melting layer
    top and bottom

    Parameters
    ----------
    radar : Radar
        Radar object
    ml_pos_field : str
        Name of the melting layer height field

    Returns
    -------
    ml_obj : radar-like object
        A radar-like object containing the field melting layer height with
        the bottom (at range position 0) and top (at range position one) of
        the melting layer at each ray

    """
    ml_obj = deepcopy(radar)

    # modify original metadata
    ml_obj.range['data'] = np.array([0, 1], dtype='float64')
    ml_obj.ngates = 2

    ml_obj.gate_x = np.zeros((ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_y = np.zeros((ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_z = np.zeros((ml_obj.nrays, ml_obj.ngates), dtype=float)

    ml_obj.gate_longitude = np.zeros(
        (ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_latitude = np.zeros(
        (ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_altitude = np.zeros(
        (ml_obj.nrays, ml_obj.ngates), dtype=float)

    # Create field
    ml_obj.fields = dict()
    ml_dict = get_metadata(ml_pos_field)
    ml_dict['data'] = np.ma.masked_all((ml_obj.nrays, ml_obj.ngates))
    ml_obj.add_field(ml_pos_field, ml_dict)

    return ml_obj

def compute_iso0(radar, ml_top, iso0_field='height_over_iso0'):
    """
    Estimates the distance respect to the freezing level of each range gate
    using the melting layer top as a proxy

    Parameters
    ----------
    radar : Radar
        Radar object
    ml_top : 1D array
        The height of the melting layer at each ray
    iso0_field : str
        Name of the iso0 field.

    Returns
    -------
    iso0_dict : dict
        A dictionary containing the distance respect to the melting layer
        and metadata

    """
    iso0_data = np.ma.masked_all((radar.nrays, radar.ngates))
    for ind_ray in range(radar.nrays):
        iso0_data[ind_ray, :] = (
            radar.gate_altitude['data'][ind_ray, :]-ml_top[ind_ray])

    iso0_dict = get_metadata(iso0_field)
    iso0_dict['data'] = iso0_data

    return iso0_dict

def detect_ml(radar_rhi, refl_field = 'Zh', rhohv_field = 'Rhohv', max_range=15000, detect_threshold=0.02,
              interp_holes=False, max_length_holes=250, check_min_length=True, fill_value=-9999.):
    
    coords_c, refl_field_c, mapping = polar_to_cartesian(
            radar_rhi, refl_field, max_range=max_range)
    coords_c, rhohv_field_c, _ = polar_to_cartesian(
         radar_rhi, rhohv_field, mapping=mapping)
    cart_res = mapping['res']

    # Get Zh and Rhohv images
    refl_im = pyart.retrieve.ml._normalize_image(refl_field_c, *ZH_IM_BOUNDS)
    rhohv_im = pyart.retrieve.ml._normalize_image(rhohv_field_c, *RHOHV_IM_BOUNDS)

    # Combine images
    comb_im = (1 - rhohv_im) * refl_im
    comb_im[np.isnan(comb_im)] = 0.

    # Get vertical gradient
    size_filt = np.floor(SIZEFILT_M / cart_res).astype(int)
    gradient = pyart.retrieve.ml._gradient_2D(pyart.retrieve.ml._mean_filter(comb_im, (size_filt, size_filt)))
    gradient_z = gradient['Gy']
    gradient_z[np.isnan(rhohv_field_c)] = np.nan

    bottom_ml, top_ml = pyart.retrieve.ml._process_map_ml(
        gradient_z, rhohv_field_c, detect_threshold, *RHOHV_VALID_BOUNDS)

    # Restrict gradient using conditions on medians
    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)

    if not np.isnan(median_bot_height):
        gradient_z[0:np.floor(LOWMLBOUND *
                              median_bot_height).astype(int), :] = np.nan
    if not np.isnan(median_top_height):
        gradient_z[np.floor(UPMLBOUND *
                            median_top_height).astype(int):, :] = np.nan

    # Identify top and bottom of ML with restricted gradient
    bottom_ml, top_ml = pyart.retrieve.ml._process_map_ml(
        gradient_z, rhohv_field_c, detect_threshold, *RHOHV_VALID_BOUNDS)
    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)

    thickness = top_ml - bottom_ml
    bad_pixels = ~np.isnan(thickness)
    bad_pixels[bad_pixels] &= (
        bad_pixels[bad_pixels] > MAXTHICKNESS_ML/cart_res)
    top_ml[bad_pixels] = np.nan
    bottom_ml[bad_pixels] = np.nan
    top_ml[np.isnan(bottom_ml)] = np.nan
    bottom_ml[np.isnan(top_ml)] = np.nan

    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)

    mid_ml = (median_top_height + median_bot_height) / 2

    # Check if ML is valid
    # 1) check if median_bot_height and median_top_height are defined
    if np.isnan(median_bot_height + median_top_height):
        invalid_ml = True
    else:
        invalid_ml = False
        # 2) Check how many values in the data are defined at the height of the
        # ML
        line_val = rhohv_field_c[np.int(mid_ml), :]

        # Check if ML is long enough
        if check_min_length:
            # the condition is that the ml is at least half as
            # long as the length of valid data at the ml height
            if np.logical_and(sum(np.isfinite(top_ml)) < 0.5,
                              sum(np.isfinite(line_val))):
                invalid_ml = True
            
    map_ml = np.zeros(gradient_z.shape)

    # 1 = below ML, 3 = in ML, 5 =  above ML
    mdata_ml = {'BELOW':1, 'INSIDE':3, 'ABOVE':5}
    # If ML is invalid, just fill top_ml and bottom_ml with NaNs
    if invalid_ml: 
        top_ml = np.nan * np.zeros((gradient_z.shape[1]))
        bottom_ml = np.nan * np.zeros((gradient_z.shape[1]))
    else:
        for j in range(0, len(top_ml) - 1):
            if(not np.isnan(top_ml[j]) and not np.isnan(bottom_ml[j])):
                map_ml[np.int(top_ml[j]):, j] = mdata_ml['BELOW']
                map_ml[np.int(bottom_ml[j]):np.int(top_ml[j]), j] = mdata_ml['INSIDE']
                map_ml[0:np.int(bottom_ml[j]), j] = mdata_ml['ABOVE']

    # create dictionary of output ml

    # Cartesian coordinates
    ml_cart = {}
    ml_cart['data'] = np.array(map_ml)
    ml_cart['x'] = coords_c[0]
    ml_cart['z'] = coords_c[1]

    ml_cart['bottom_ml'] = np.array((bottom_ml) * cart_res)
    ml_cart['top_ml'] = np.array((top_ml) * cart_res)

    # Polar coordinates
    (theta, r), (bottom_ml, top_ml), map_ml = remap_to_polar(
        radar_rhi, ml_cart['x'], ml_cart['bottom_ml'], ml_cart['top_ml'],
        interp=True)
    map_ml = np.ma.array(map_ml, mask=map_ml == 0, fill_value=fill_value)
    bottom_ml = np.ma.masked_invalid(bottom_ml)
    top_ml = np.ma.masked_invalid(top_ml)

    ml_pol = {}
    ml_pol['data'] = map_ml
    ml_pol['theta'] = theta
    ml_pol['range'] = r
    ml_pol['bottom_ml'] = bottom_ml
    ml_pol['top_ml'] = top_ml

    output = {}
    output['ml_cart'] = ml_cart
    output['ml_pol'] = ml_pol
    output['ml_exists'] = not invalid_ml
    
    
    all_ml = [output]
    ml_field = 'ML'
    ml_pos_field = 'MLHeight'
    
    ml_dict = get_metadata(ml_field)
    ml_dict.update({'_FillValue': 0})
    ml_obj = create_ml_obj(radar_rhi, ml_pos_field)

    ml_data = np.ma.masked_all(
        (radar_rhi.nrays, radar_rhi.ngates), dtype=np.uint8)
    for sweep in range(radar_rhi.nsweeps):
        sweep_start = radar_rhi.sweep_start_ray_index['data'][sweep]
        sweep_end = radar_rhi.sweep_end_ray_index['data'][sweep]
        ml_obj.fields[ml_pos_field]['data'][sweep_start:sweep_end+1, 0] = (
            all_ml[sweep]['ml_pol']['bottom_ml'])
        ml_obj.fields[ml_pos_field]['data'][sweep_start:sweep_end+1, 1] = (
            all_ml[sweep]['ml_pol']['top_ml'])
        ml_data[sweep_start:sweep_end+1, :] = all_ml[sweep]['ml_pol']['data']
    ml_dict['data'] = ml_data

    valid_values = ml_obj.fields[ml_pos_field]['data'][:, 1].compressed()

    get_iso0 = True
    iso0_field='iso0'
    # get the iso0
    iso0_dict = None
    if get_iso0:
        iso0_dict = compute_iso0(
            radar_rhi, ml_obj.fields[ml_pos_field]['data'][:, 1],
            iso0_field=iso0_field)
    
    return output, ml_obj, iso0_dict



if __name__ =='__main__':
    
    radar = pyart.io.read_cfradial('/ltenas8/users/anneclaire/ICEGENESIS_2021/ICEGENESIS/Proc_data_v0_clutter_filter_no_thres/XPOL-20210128-090449_FFT_RHI.nc')
    output = detect_ml(radar)
    
    fig, ax = plt.subplots()
    im = ax.pcolormesh(output['ml_cart']['x'],output['ml_cart']['z'],output['ml_cart']['data'])
    plt.colorbar(im)
    ax.plot(output['ml_cart']['x'],output['ml_cart']['top_ml'],'-r')
    ax.plot(output['ml_cart']['x'],output['ml_cart']['bottom_ml'],'-r')
    ax.plot(output['ml_cart']['x'],.5*(output['ml_cart']['top_ml']+output['ml_cart']['bottom_ml']),'-r')

    ax.set_ylim(0,3000)
    fig.savefig('test.png')
