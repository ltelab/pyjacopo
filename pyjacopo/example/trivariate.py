#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:37:31 2021

@author: ghiggi
"""
import matplotlib
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata

#-----------------------------------------------------------------------------.
#### Utils #####
### Pcolor as RGB 
### - Plot RGB with pcolormesh() 
# RGB image (non-uniform coordinate)
def plot_rgb_pcolormesh(ax, da, x, y, rasterized=True):
    im_rgb = da.values
    colorTuple = im_rgb.reshape((im_rgb.shape[0]*im_rgb.shape[1],im_rgb.shape[2]))
    im = ax.pcolormesh(da[x].values,
                       da[y].values, 
                       im_rgb[:,:,1], # dummy to work ...
                       color = colorTuple,
                       linewidth=0,rasterized=rasterized)
    im.set_array(None)
    return im
    
    
def get_trivariate_x(y1=None, y2=None, y3=None):
    """Get the x position in the unit traingle."""
    h = np.sqrt(1 - (0.5**2))
    if (y1 is None and (y2 is not None and y3 is not None)):
        y1 = 1 - y2 - y3 
        x = (1-y3)*h/np.cos(np.deg2rad(30)) - y1*h/np.tan(np.deg2rad(60))
        return x
    elif (y2 is None and (y1 is not None and y3 is not None)):
        x = (1-y3)*h/np.cos(np.deg2rad(30)) - y1*h/np.tan(np.deg2rad(60))
        return x
    elif (y3 is None and (y1 is not None and y2 is not None)):
        y3 = 1 - y1 - y2 
        x = (1-y3)*h/np.cos(np.deg2rad(30)) - y1*h/np.tan(np.deg2rad(60))
        return x
    else:
        raise ValueError("Provide 2 arguments")

def get_trivariate_y(y1):
    """Get the y position in the unit traingle."""
    h = np.sqrt(1 - (0.5**2))
    return y1*h

def vrange(x, dropna=True):
    if dropna: 
        v = [np.nanmin(x), np.nanmax(x)]  
    else: 
        v = [np.min(x), np.max(x)]
    return v

#----------------------------------------------------------------------------.
# ################################
#### Trivariate color palette ####
# ################################
def trivariate_colpal(name="1", clockwise_rotation="0", mirror_y=False):
    """Return a trivariate color palette.
    
    Mirror after rotation.
    """
    ##------------------------------------------------------------------------.
    # Checks 
    if not isinstance(name, str): 
        raise TypeError("'name' must be a string")
    if name not in ["1","2","3","4","5","6","7"]:
        raise ValueError("'name must be a string integer between 1 and 7")
    if not isinstance(clockwise_rotation, str): 
        raise TypeError("'clockwise_rotation' must be a string")
    if clockwise_rotation not in ["0","90","180"]:
        raise ValueError("Valid 'clockwise_rotation': '0', '90', '180'")
    ##------------------------------------------------------------------------.
    # Define dictionary of nice trivariate color palettes
    # --> Order: "bottom left","left center","top","right center","bottom right","bottom center","center"
    tricols_dict = {"1": np.array([50,200,0,    
                                   0,230,230,  
                                   40,40,230,    
                                   200,0,255,   
                                   200,0,50, 
                                   255,255,0,
                                   200,200,200]).reshape(7,3),
                    "2": np.array([0,230,0,    
                                   0,230,230,  
                                   0,0,230,    
                                   230,0,230,  
                                   230,0,0,    
                                   230,230,0,  
                                   200,200,200]).reshape(7,3),
                    "3": np.array([204,204,0, # bottom left
                                   0,153,51,  # left center
                                   0,0,204,   # top
                                   204,0,204, # right center
                                   204,0,0,   # bottom right
                                   204,102,0, # bottom center
                                   200,200,200]).reshape(7,3),
                    "4": np.array([63, 200,0,
                                   20,200,200,
                                   0,100,255,
                                   255,100,0,
                                   230,0,100,
                                   240,240,30,
                                   220,220,150]).reshape(7,3),
                    "5": np.array([63, 200,0,
                                   20,210,210,
                                   0,100,255,
                                   200,50, 100,
                                   255,100,0,
                                   240,240,30,
                                   220,220,150]).reshape(7,3),
                    "6": np.array([63, 200,200,
                                   0,200,154,
                                   0,150,250,
                                   230,230,92,
                                   250,120,0,
                                   153,204,0,
                                   220,220,150]).reshape(7,3),
                    "7": np.array([63, 200,0,
                                   31,150,125,
                                   0,100,250,
                                   230,230,92,
                                   250,100,0,
                                   153,204,0,
                                   220,200,150]).reshape(7,3)}
    ##------------------------------------------------------------------------.
    # Retrieve trivariate color palette 
    tricols = tricols_dict[name]
    ##------------------------------------------------------------------------.
    # Rotate colors
    if clockwise_rotation == "90":
        tricols = tricols[[2,3,4,5,0,1,6],:]
    if clockwise_rotation == "180":
        tricols = tricols[[4,5,0,1,2,3,6],:]
    ##------------------------------------------------------------------------.  
    # Mirror colors 
    if mirror_y:
        tricols = tricols[[4,3,2,1,0,5,6],:]
    ##------------------------------------------------------------------------.
    # Return trivariate color palette
    return tricols

#-----------------------------------------------------------------------------.
# ###############################
### Trivariate color mapping ####
# ###############################
def _set_trivariate_colors_1D(y1, y2, y3, 
                              tri_colpal = trivariate_colpal("1"),   
                              NA_fill = True, 
                              NA_color = [255,255,255], 
                              max_rgb = 1):
    # y1: left, y2: right, y3: bottom
    # ------------------------------------------------------------------------.
    ############## -
    ## Checks ####
    ############## -
    # Check y1, y2, y3 
    if not isinstance(y1, (list,tuple,np.ndarray)):
        raise TypeError("'y1' must be a list, tuple or np.ndarray.")
    if not isinstance(y2, (list,tuple,np.ndarray)):
        raise TypeError("'y2' must be a list, tuple or np.ndarray.")
    if not isinstance(y3, (list,tuple,np.ndarray)):
        raise TypeError("'y3' must be a list, tuple or np.ndarray.")
    # - Checks that the variable have same length  
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 must have same length.")
    if len(y1) != len(y3):
        raise ValueError("y1 and y3 must have same length.")
    # - Check that there is values (fine is all nan)
    if len(y1) == 0:
        raise ValueError("y1 and y2 must be of at least length 1.")
    # - Ensure to be numpy array
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    ##------------------------------------------------------------------------.
    # Check max_rgb 
    if not isinstance(max_rgb, int):
        raise TypeError("'max_rgb' must be an integer.")
    if max_rgb not in [1, 255]:
        raise ValueError("'max_rgb' must be either 1 or 255.")
    ##------------------------------------------------------------------------.
    # Check NA_color 
    # - If matplotlib color (return a 0-1 rgb)
    if isinstance(NA_color, str):
        if not matplotlib.colors.is_color_like(NA_color):
            raise ValueError("Not a valid matplotlib color.")
        NA_color = matplotlib.colors.to_rgba_array(NA_color) # 0-1
    # - If RGB vector 
    elif isinstance(NA_color, (list,tuple,np.ndarray)) and len(NA_color) in [3, 4]:
        NA_color = np.array(NA_color) 
        if np.any(NA_color < 0): 
            raise ValueError("NA_color RGB values must be positive and between 0-1 or 0-255.")
        if np.any(NA_color > 255): 
            raise ValueError("NA_color RGB values must be between 0-1 or 0-255.")
        if np.any(NA_color > 1):
            NA_color = NA_color/255 
        NA_color = NA_color # 0 -1 
    else:
        raise ValueError("'NA_color' not valid. Provide rgb vector or matplotlib color.")
    ##------------------------------------------------------------------------.
    ## Check tri_colpal 
    if np.any(tri_colpal < 0):
        raise ValueError("RGB colors of the trivariate color palette must be  between 0-1 or 0-255")
    if np.any(tri_colpal > 255):
        raise ValueError("RGB colors of the trivariate color palette must be between 0-1 or 0-255")
    if np.any(tri_colpal > 1):
        tri_colpal = tri_colpal/255
    if tri_colpal.shape[0] not in [3, 4, 6, 7]:
        raise ValueError("The trivariate color palette require specification of 3, 4, 6 or 7 colors.")
   
    ##------------------------------------------------------------------------.
    ## - Check values are in the interval 0-1 or 0-100
    r1 = vrange(y1, dropna=True)
    r2 = vrange(y2, dropna=True)
    r3 = vrange(y3, dropna=True)
    if min(r1) < 0: 
        raise ValueError("y1 must be between 0-1 or 0-100")
    if min(r2) < 0: 
        raise ValueError("y2 must be between 0-1 or 0-100")  
    if min(r2) < 0: 
        raise ValueError("y3 must be between 0-1 or 0-100")
    if max(r1) > 100:
        raise ValueError("y1 must be between 0-1 or 0-100")
    if max(r2) > 100: 
        raise ValueError("y2 must be between 0-1 or 0-100")
    if max(r3) > 100: 
        raise ValueError("y3 must be between 0-1 or 0-100")
    ##------------------------------------------------------------------------.
    # If any value is larger than 1, assume percentage [0-100]  
    # --> Normalize to 0-1 
    if (max(r1) > 1 or max(r2) > 1 or max(r3) > 1):
        y1 = y1/100
        y2 = y2/100
        y3 = y3/100
    ##------------------------------------------------------------------------.
    # Check that values sum up to 1 (or 100)  
    tot = y1 + y2 + y3
    tot = tot[~np.isnan(tot)] 
    if np.any(tot < 0.98):
        raise ValueError("The sum of the three variables must be equal to 1 or 100.")
    ##------------------------------------------------------------------------.
    ###############################################################
    ## Specifiy the color position on the equilateral triangle ####
    ###############################################################
    # Define height of the unit equilateral triangle 
    h = np.sqrt(1 - (0.5**2))
    # Determine the color position in the equilateral triangle   
    # --> Assuming x=[0-1], y=[0,h]
    n_cols = tri_colpal.shape[0]
    if n_cols == 7: 
        # 7:  bottom left, left center, top , right center, bottom right, bottom center, triangle center
        coords = {"x": [0, 1-np.cos(np.deg2rad(30))*h,0.5, np.cos(np.deg2rad(30))*h, 1, 0.5, 0.5],
                  "y": [0, h-np.sin(np.deg2rad(30))*h,  h, np.sin(np.deg2rad(30))*h, 0, 0  , h/3]}
    elif n_cols == 6:  
        # 6: bottom left, left center, top , right center, bottom right, bottom center
        coords = {"x": [0, 1-np.cos(np.deg2rad(30))*h, 0.5, np.cos(np.deg2rad(30))*h ,1, 0.5],
                  "y": [0, h-np.sin(np.deg2rad(30))*h,   h, np.sin(np.deg2rad(30))*h, 0, 0  ]} 
    elif n_cols == 4: 
        # 4: bottom left, top, bottom right, center
        coords = {"x": [0, 0.5, 1, 0.5],
                  "y": [0, h  , 0, 1/3*h]}
    elif n_cols == 3:
        # 3: bottom left, top, bottom right, 
        coords = {"x": [0,0.5, 1],
                  "y": [0,  h, 0]}
    else:
        raise ValueError("The trivariate color palette require 3, 4, 6 or 7 colors.")
    #---------------------------------------------------------------------------.
    #############################
    ## Initialize NaN colors ####
    #############################
    # Initialize output 
    output_RGB = np.zeros((len(y1),3))*np.nan
    ## Remove NA  
    idx_NA = np.logical_or(np.isnan(y1),np.isnan(y2), np.isnan(y3))
    y1 = y1[~idx_NA]
    y2 = y2[~idx_NA]
    y3 = y3[~idx_NA]
    ######################################## 
    ## Retrieve colors for each channel ####
    ########################################
    # Project the requested coordinates on the equilateral triangle 
    x_req = get_trivariate_x(y1=y1, y3=y3)   
    y_req = get_trivariate_y(y1=y1) 
    # Interpolate each single color channel in the unit equilateral triangle 
    for i in range(3):
        # Retrieve the channel colors for each observations (x_req, y_req)
        val = griddata(points = np.column_stack((coords["x"], coords["y"])),
                       values = tri_colpal[:,i],
                       xi = np.column_stack((x_req, y_req)),
                       method = "cubic") # linear
        val[val < 0] = NA_color[i]  # if extrapolating 
        val[val > 1] = NA_color[i]  # if extrapolating 
        output_RGB[~idx_NA,i] = val   
    #-------------------------------------------------------------------------.
    # Infill nan with NA_color 
    if NA_fill:  
        output_RGB[idx_NA, :] = NA_color
    #-------------------------------------------------------------------------.
    # Rescale RGB to 0-255
    if max_rgb == 255:
        output_RGB = output_RGB*255
    #-------------------------------------------------------------------------.
    # Return colors
    return output_RGB 

##----------------------------------------------------------------------------.
def set_trivariate_colors(y1,y2,y3, 
                          tri_colpal = trivariate_colpal("1"),   
                          NA_fill = True, 
                          NA_color = [255,255,255], 
                          max_rgb = 1):
    # ------------------------------------------------------------------------.
    # Retrieve array shape
    shape_arr = y1.shape
    # Check  size 
    if not np.array_equal(np.array(y1.shape), np.array(y2.shape)):
        raise ValueError("y1 and y2 does not have the same shape.")
    if not np.array_equal(np.array(y1.shape), np.array(y3.shape)):
        raise ValueError("y1 and y3 does not have the same shape.")
    # Reshape to 1D 
    y1 = y1.ravel()
    y2 = y2.ravel()
    y3 = y3.ravel()
    ##------------------------------------------------------------------------.
    # Retrieve color array  
    tri_cols = _set_trivariate_colors_1D(y1=y1, y2=y2, y3=y3, 
                                         tri_colpal=tri_colpal, 
                                         NA_fill=NA_fill,
                                         NA_color=NA_color,
                                         max_rgb=max_rgb)
    ##------------------------------------------------------------------------.
    # Reshape color array 
    reshape_dim = list(shape_arr) + [3]
    output_RGB = tri_cols.reshape(*reshape_dim)
    ##------------------------------------------------------------------------.
    return output_RGB

#-----------------------------------------------------------------------------.
# ##########################
### Trivariate colorbar ####
# ##########################
def _add_oblique_axis(ax,  
                      ticks_angle = None,
                      ticks_type = "0-1",
                      ticks_interval = 0.2,
                      ticks_linewidth = 0.5,
                      ticks_length = 0.02,
                      ticks_spacing = 0.2,
                      ticks_fontdict = None,
                      ticks_color = "black",
                      axis_side = "left",  
                      axis_color = "black",
                      axis_linewidth = 1,
                      axis_linestyle = "-"):
    ##------------------------------------------------------------------------.    
    ## Checks 
    if not isinstance(axis_side, str):
        raise TypeError("'axis_side' must be a string.")
    if axis_side not in ["left", "right", "bottom"]:
        raise ValueError("'axis_side' must be either 'left', 'right' or 'bottom'.")
    if not isinstance(ticks_type, str):
        raise TypeError("'ticks' must be a string.")
    if ticks_type not in ["0-1", "0-100"]:
        raise ValueError("'ticks_type' must be either '0-1' or '0-100'.")
    if ticks_angle is None: 
        ticks_angle_dict = {'left': 180,
                            'right': 60,
                            'bottom': -60}
        ticks_angle = ticks_angle_dict[axis_side]
    ##------------------------------------------------------------------------.
    # Define horizontal alignment dictionary for tick labels  
    horizontalalignment_dict = {'left': 'left',
                                'right': 'right', # center 
                                'bottom': 'right'}
    ##------------------------------------------------------------------------.
    # Define ticks and ticklabels
    if ticks_type == "0-100":
        if ticks_interval > 50 or ticks_interval <= 0: 
            raise ValueError("Provide a 'ticks_interval' between 0 and 50.")
        ticks = np.arange(0,100, step=ticks_interval)
        ticks = np.append(ticks, 100)
        ticks_labels = ticks
        ticks = ticks/100 # for plotting assume 0-1
    else: 
        if ticks_interval > 0.5 or ticks_interval <= 0:  
            raise ValueError("Provide a 'ticks_interval' between 0 and 0.5.")
        ticks = np.arange(0,1, step=ticks_interval)
        ticks = np.append(ticks, 1)
        ticks_labels = ticks
    ##------------------------------------------------------------------------.
    # Get position of ticks 
    n_ticks = ticks.shape[0]
    if axis_side == "left":
        ticks_pos_x = get_trivariate_x(y1 = ticks, y2 = np.zeros(n_ticks))
        ticks_pos_y = get_trivariate_y(y1 = ticks) 
    elif axis_side == "right":
        ticks_pos_x = get_trivariate_x(y2 = ticks, y3 = np.zeros(n_ticks))
        ticks_pos_y = get_trivariate_y(y1 = 1 - ticks) 
    else: # bottom 
        ticks_pos_x = get_trivariate_x(y3 = ticks, y1 = np.zeros(n_ticks)) 
        ticks_pos_y = get_trivariate_y(y1 = np.zeros(n_ticks))
    ##------------------------------------------------------------------------.  
    # Outer position of the ticks
    ticks_out_x = ticks_pos_x + ticks_length*np.cos(np.deg2rad(ticks_angle))
    ticks_out_y = ticks_pos_y + ticks_length*np.sin(np.deg2rad(ticks_angle))
    ##------------------------------------------------------------------------.
    # Ticks coords 
    ticks_line_xcoords = np.column_stack((ticks_pos_x, ticks_out_x))
    ticks_line_ycoords = np.column_stack((ticks_pos_y, ticks_out_y)) 
    ##------------------------------------------------------------------------.
    # Position of tick labels 
    if axis_side in ["left, bottom"]:
        ticks_text_x = ticks_pos_x + (ticks_length + ticks_spacing)*np.cos(np.deg2rad(ticks_angle))
        ticks_text_y = ticks_pos_y + (ticks_length + ticks_spacing)*np.sin(np.deg2rad(ticks_angle))
    else:
        ticks_text_x = ticks_pos_x + (ticks_length + ticks_spacing)*np.cos(np.deg2rad(ticks_angle))
        ticks_text_y = ticks_pos_y + (ticks_length + ticks_spacing/2)*np.sin(np.deg2rad(ticks_angle))
    ##------------------------------------------------------------------------.
    # Draw axis 
    ax.plot(ticks_pos_x[[0,-1]], 
            ticks_pos_y[[0,-1]],
            color = axis_color,
            linewidth = axis_linewidth,
            ls = axis_linestyle)
    ##------------------------------------------------------------------------.
    ## Draw ticks
    for i in range(ticks_line_xcoords.shape[0]):
        ax.plot(ticks_line_xcoords[i,],
                ticks_line_ycoords[i,], 
                color = ticks_color,
                linewidth = ticks_linewidth,
                ls = "-")
    ##------------------------------------------------------------------------.
    # Draw tick labels  
    for i in range(len(ticks_labels)):
        ax.text(ticks_text_x[i],ticks_text_y[i], ticks_labels[i],
                horizontalalignment = horizontalalignment_dict[axis_side],
                fontdict=ticks_fontdict, color=ticks_color)
        
    ##------------------------------------------------------------------------.
    # Return the axis 
    return ax 
 
def _add_triangle_axis(ax,  
                       ticks_type = "0-1",
                       ticks_interval = 0.2,
                       ticks_linewidth = 0.5,
                       ticks_length = 0.02,
                       ticks_spacing = 0.2,
                       ticks_fontdict = None,
                       ticks_color = "black",
                       axis_side = "left",  
                       axis_color = "black",
                       axis_linewidth = 1,
                       axis_linestyle = "-",
                       xlabel="xlab",
                       ylabel="ylab",
                       zlabel="zlab", 
                       labels_color = "black",
                       labels_spacing = 0.1, 
                       labels_fontdict = None):             
    ##------------------------------------------------------------------------.
    # Checks                   
    if not isinstance(ticks_type, str):
        raise TypeError("'ticks' must be a string.")
    if ticks_type not in ["0-1", "0-100"]:
        raise ValueError("'ticks_type' must be either '0-1' or '0-100'.")
    ##------------------------------------------------------------------------.
    # Add axis  
    for axis_side in ["left", "right", "bottom"]:
        ax = _add_oblique_axis(ax=ax, 
                               axis_side=axis_side, 
                               ticks_angle = None, 
                               ticks_type = ticks_type,
                               ticks_interval = ticks_interval,
                               ticks_linewidth = ticks_linewidth,
                               ticks_length = ticks_length,
                               ticks_spacing = ticks_spacing,
                               ticks_fontdict = ticks_fontdict,
                               ticks_color = ticks_color,
                               axis_color = axis_color,
                               axis_linestyle = axis_linestyle,
                               axis_linewidth = axis_linewidth)                      
    ##------------------------------------------------------------------------.
    # Draw axis labels   
    # - Left
    ax.text(0.05 - labels_spacing, 0.5, s = xlabel, fontdict=labels_fontdict, color=labels_color)
    # - Right
    ax.text(0.95 + labels_spacing, 0.5, s = ylabel, fontdict=labels_fontdict, color=labels_color)
    # - Bottom
    ax.text(0.5, -0.15 - labels_spacing/2, s = zlabel, fontdict=labels_fontdict, color=labels_color)
    ##------------------------------------------------------------------------.
    # Return axis
    return ax

def _add_gridded_triangles(ax, interval=0.1, color="white", linewidth=0.5, linestyle="-"):
    if interval > 1:
        interval = interval/100
    ##------------------------------------------------------------------------.
    ax_seq = np.arange(0, 1, step = interval)
    ax_seq = np.append(ax_seq, 1)
    zero_arr = np.zeros(ax_seq.shape[0])
    # Draw lines for each side 
    for side in ['left', 'right', 'bottom']: 
        if side == "left":
            combo_start = np.column_stack((ax_seq, zero_arr, ax_seq[::-1]))
            combo_end = np.column_stack((ax_seq, ax_seq[::-1], zero_arr))
      
        if side == "right":
            combo_start = np.column_stack((ax_seq[::-1], ax_seq, zero_arr))
            combo_end = np.column_stack((zero_arr, ax_seq, ax_seq[::-1]))
      
        if side == "bottom":
            combo_start = np.column_stack((zero_arr,ax_seq[::-1],ax_seq))
            combo_end = np.column_stack((ax_seq[::-1], zero_arr, ax_seq))
      
        x_start = get_trivariate_x(y1=combo_start[:,0], y3=combo_start[:,2])
        x_end = get_trivariate_x(y1=combo_end[:,0], y3=combo_end[:,2])
        y_start = get_trivariate_y(y1=combo_start[:,0])
        y_end = get_trivariate_y(y1=combo_end[:,0])
        xcoords = np.column_stack((x_start, x_end))
        ycoords = np.column_stack((y_start, y_end))
        # Plot lines 
        for i in range(xcoords.shape[0]):
            ax.plot(xcoords[i,],
                    ycoords[i,], 
                    color = color,
                    linewidth = linewidth,
                    linestyle = linestyle)
    ##------------------------------------------------------------------------.
    # Return axis
    return ax

def _get_triangle_colorbar_image(tri_colpal = trivariate_colpal("1"), 
                                 NA_fill = True,
                                 NA_color = [255, 255, 255],
                                 ns = 1000):
    """Retrieve RGB image displaying the trivariate colorbar.""" 
    ##-------------------------------------------------------------------------.
    h = int(np.round(np.sqrt(ns**2-(ns/2)**2)))      # number of pixels along y (for triangle)
    v_y = np.concatenate((np.ones(int(ns-h))*np.nan, # y without and with the triangle
                          np.linspace(1,0,num=h)))   # top triangle (1), bottom triangle(0)
    x, y = np.meshgrid(np.arange(1,ns+1), np.arange(1, ns+1))
    #--------------------------------------------------------------------------.
    # Here we define which side is (y1,y2,y3)
    left_mesh = np.repeat(v_y,ns).reshape(ns, ns) # left 
    bottom_mesh = 1 - np.flipud((x + (y/np.tan(np.deg2rad(60))))*np.cos(np.deg2rad(30)))/h  # bottom
    right_mesh = np.fliplr(bottom_mesh)  # right 
    triangle_mesh = np.stack((left_mesh, right_mesh, bottom_mesh), axis=2) 
    #-------------------------------------------------------------------------.
    # Identify values outside triangle and mask it 
    mask_out_triangle = np.sum(triangle_mesh > 0, axis=2) < 3 
    mask_out_triangle[np.isnan(mask_out_triangle)] = True
    # plt.imshow(mask_out_triangle)  
    triangle_mesh[:,:,0][mask_out_triangle] = np.nan
    triangle_mesh[:,:,1][mask_out_triangle] = np.nan
    triangle_mesh[:,:,2][mask_out_triangle] = np.nan
    #--------------------------------------------------------------------------.
    # Normalize value between 0 and 1 
    proportions_arr = np.divide(triangle_mesh, np.expand_dims(np.sum(triangle_mesh, axis=2),axis=2))
    # Retrieve color matrix 
    triangle_RGB = set_trivariate_colors(y1=proportions_arr[:,:,0],
                                         y2=proportions_arr[:,:,1],
                                         y3=proportions_arr[:,:,2],
                                         tri_colpal=tri_colpal,
                                         NA_fill = NA_fill,
                                         NA_color = NA_color,
                                         max_rgb = 1)
    return triangle_RGB

def plot_trivariate_colorbar(tri_colpal = trivariate_colpal("5"),
                             xlabel = "xlab", ylabel = "ylab", zlabel = "zlab",
                             ax = None,
                             display_axis = False,
                             xlim = [-0.2,1.2],
                             ylim = [-0.2,1.1],
                             ticks_type = "0-100",
                             ticks_interval = 20,
                             ticks_linewidth = 1,
                             ticks_length = 0.03,
                             ticks_spacing = 0.12,
                             ticks_fontdict = None,
                             ticks_color = "black",
                             axis_color = "black",
                             axis_linewidth = 1,
                             axis_linestyle = "-",
                             grid_interval = 0.1,
                             grid_color = "black",
                             grid_linewidth = 0.2,
                             grid_linestyle= "-",
                             labels_color = "black",
                             labels_spacing = 0.05, 
                             labels_fontdict = None):                             
                       
    ##------------------------------------------------------------------------.
    # Check xlim and ylim 
    # - Must include 0-1 range 
    if len(xlim) != 2: 
        raise ValueError("xlim must have length 2.")
    if len(ylim) != 2: 
        raise ValueError("ylim must have length 2.")
    if xlim[0] > 0: 
        raise ValueError("xlim should be smaller than 0.")
    if ylim[0] > 0: 
        raise ValueError("ylim should be smaller than 0.")
    if xlim[1] < 1: 
        raise ValueError("xlim should be larger than 1.")
    if ylim[1] < 1: 
        raise ValueError("ylim should be larger than 1.")
    ##------------------------------------------------------------------------.
    # Initialize figure if ax not provided 
    if ax is None: 
        fig, ax = plt.subplots(1,1)
    ##------------------------------------------------------------------------.
    ## Retrieve the RGB image with the triangle colorbar   
    triangle_RGB = _get_triangle_colorbar_image(tri_colpal = tri_colpal, ns=1000)
    ## Plot the RGB image with the triangle colorbar 
    # - Set axis to be between 0 and 1 --> extent = [0,1,0,1] = [left, right, bottom, up]
    ax.imshow(triangle_RGB, aspect="equal", extent=[0,1,0,1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ##------------------------------------------------------------------------.
    # - Show if display axis
    if not display_axis:
        ax.set_axis_off()
    ##------------------------------------------------------------------------.
    # - Add diagonal lines 
    ax = _add_gridded_triangles(ax=ax,
                                interval = grid_interval,
                                color = grid_color, 
                                linewidth = grid_linewidth, 
                                linestyle = grid_linestyle)
    ##------------------------------------------------------------------------.
    ## Add axis 
    ax = _add_triangle_axis(ax=ax, 
                            ticks_type = ticks_type,
                            ticks_interval = ticks_interval,                           
                            ticks_linewidth = ticks_linewidth,
                            ticks_length = ticks_length,
                            ticks_spacing = ticks_spacing,
                            ticks_fontdict = ticks_fontdict,
                            ticks_color = ticks_color,
                            axis_color = axis_color,
                            axis_linewidth = axis_linewidth,
                            axis_linestyle = axis_linestyle,              
                            xlabel = xlabel,
                            ylabel = ylabel,
                            zlabel = zlabel, 
                            labels_color = labels_color,
                            labels_spacing = labels_spacing, 
                            labels_fontdict = labels_fontdict)
    return ax 

def display_trivariate_colpals(clockwise_rotation="0", 
                               mirror_y=False):
    if not isinstance(clockwise_rotation, str): 
        raise TypeError("'clockwise_rotation' must be a string")
    if clockwise_rotation not in ["0","90","180"]:
        raise ValueError("Valid 'clockwise_rotation': '0', '90', '180'")
        
    fig, axes = plt.subplots(2,3, figsize=(12,6))
    for i, ax in enumerate(axes.ravel()):
        tri_colpal = trivariate_colpal(str(i+1),
                                     clockwise_rotation=clockwise_rotation,
                                     mirror_y=mirror_y)
        ax = plot_trivariate_colorbar(tri_colpal = tri_colpal,
                                      xlabel = "RP", ylabel = "AG", zlabel = "CR",
                                      ax = ax)
        ax.set_title('Name ' + '"' + str(i) + '"')
    return fig

    