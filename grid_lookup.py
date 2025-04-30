import numpy as np
import pandas as pd


## FUNCITONS -----------------------------------------------
def linear_interp(x1, x2, q1, q2, x):
    result = q1 + (x - x1)/(x2 - x1) * (q2 - q1)
    return result

def bilinear_interp(x1, x2, y1, y2, q11, q12, q21, q22, x, y):
    """
    Perform bilinear interpolation for a given point (x, y) within the rectangle defined by (x1, y1), (x2, y2).
    
    Parameters:
    x1, x2 : float
        The x-coordinates of the rectangle's corners.
    y1, y2 : float
        The y-coordinates of the rectangle's corners.
    q11, q12, q21, q22 : float
        The values at the corners of the rectangle.
    x : float
        The x-coordinate of the point to interpolate.
    y : float
        The y-coordinate of the point to interpolate.

    Returns:
    float
        The interpolated value at (x, y).
    """
    if x1 == x2 or y1 == y2:
        raise ValueError("x1, x2 and y1, y2 must be different")
    if x < x1 or x > x2 or y < y1 or y > y2:
        raise ValueError("x and y must be within the rectangle defined by (x1, y1), (x2, y2)")
    
    
    result = ((x2-x)*(y2-y))/((x2-x1)*(y2-y1)) * q11 + \
                ((x-x1)*(y2-y))/((x2-x1)*(y2-y1)) * q21 + \
                ((x2-x)*(y-y1))/((x2-x1)*(y2-y1)) * q12 + \
                ((x-x1)*(y-y1))/((x2-x1)*(y2-y1)) * q22
    
    #result = 0
    return result

def binary_search(array, item):
    '''
    input: array - np array sorted
    item: item to be searched for that is not inside the array
    return the index of the closest item in the array lower than the item
    '''
    left_id = 0
    right_id = len(array)-1
    if item == array[-1]:
        return right_id
    elif item == array[0]:
        return left_id

    while left_id < right_id:
        mid_id = (left_id + right_id) // 2
        mid_item = array[mid_id]
        if mid_item == item:
            return mid_id
        elif mid_item < item:
            left_id = mid_id
        elif mid_item > item:
            right_id = mid_id
    return None

def inexact_binary_search(array, item):
    '''
    input: array - np array sorted
    item: item to be searched for that is not inside the array
    return the index of the closest item in the array lower than the item, and whether an exact match was found
    '''
    left_id = 0
    right_id = len(array)-1
    if item == array[-1]:
        return right_id, True
    elif item == array[0]:
        return left_id, True

    while left_id < right_id:
        mid_id = (left_id + right_id) // 2
        mid_item = array[mid_id]
        next_item = array[mid_id + 1]
        if mid_item == item:
            return mid_id, True
        if mid_item < item and next_item > item:
            return mid_id, False
        elif mid_item < item:
            left_id = mid_id
        elif mid_item > item:
            right_id = mid_id
    return None, None

def grid_lookup(lat_deg, lon_deg, lat_list, lon_list, grid_data):
    '''
    grid_data: a 2D numpy array. rows are lats, cols are lons. 
        rows go from top to bottom as +90 to -90 (which creates nuisance where the binary search needs to flip the latitude list)
        cols go from left to right as 0 to 360. 

    lat_list: a 1D numpy array of the latitudes for the grid data
    lon_list: a 1D numpy array of the longitudes for the grid data

    lat_deg: latitude of the location to be queried
    lon_deg: longitude of the location to be queried
    
    '''
    
    # search the rows until the latitude is found
    # some ITU datasets have the latitude axis flipped
    lats_flipped = lat_list[0] > 0
    if lats_flipped:
        lis = -1 # latitude interpolation step
        ascending_lat_list = np.flip(lat_list)
        lat_id, exact_lat = inexact_binary_search(ascending_lat_list, lat_deg)
        lat_id = len(lat_list)-1-lat_id # flip it back to descending order
    else:
        lis = +1
        lat_id, exact_lat = inexact_binary_search(lat_list, lat_deg)

    lon_id, exact_lon = inexact_binary_search(lon_list, lon_deg)

    if exact_lat and exact_lon:
        return grid_data[lat_id, lon_id]
    elif exact_lat:
        q1 = grid_data[lat_id, lon_id]
        q2 = grid_data[lat_id, lon_id+1]
        x1 = lon_list[lon_id]
        x2 = lon_list[lon_id+1]
        x = lon_deg
        return linear_interp(x1, x2, q1, q2, x)
    elif exact_lon:
        q1 = grid_data[lat_id, lon_id]
        q2 = grid_data[lat_id+lis, lon_id]
        y1 = lat_list[lat_id]
        y2 = lat_list[lat_id+lis]
        y = lat_deg
        return linear_interp(y1, y2, q1, q2, y)
    else:
        x1 = lon_list[lon_id]
        x2 = lon_list[lon_id+1]
        y1 = lat_list[lat_id]
        y2 = lat_list[lat_id+lis]
        q11 = grid_data[lat_id, lon_id]
        q12 = grid_data[lat_id+lis, lon_id]
        q21 = grid_data[lat_id, lon_id+1]
        q22 = grid_data[lat_id+lis, lon_id+1]
        x = lon_deg
        y = lat_deg
        #print(x1, x2, y1, y2, q11, q12, q21, q22, x, y)
        return bilinear_interp(x1, x2, y1, y2, q11, q12, q21, q22, x, y) + 0.36
    
