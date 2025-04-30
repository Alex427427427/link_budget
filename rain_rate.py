'''
Inputs: 
    desired probability of exceedance 
    location on earth
Outputs: 
    rainfall rate exceeded for that desired probability of exceedance
'''

import numpy as np
import pandas as pd
import grid_lookup as gl
import scipy.integrate as it

month_indices = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
N_days_monthly = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

## ITU DATA READ
# T = temperature in K
lat_grid_T = pd.read_csv("atmos_data\\ITU-R P1510 mean temp\\LAT_T.TXT", sep="\\s+", header=None).to_numpy() # Y
lon_grid_T = pd.read_csv("atmos_data\\ITU-R P1510 mean temp\\LON_T.TXT", sep="\\s+", header=None).to_numpy() # X
lat_list_T = lat_grid_T[:,0].flatten()
lon_list_T = lon_grid_T[0]
# TR = total rainfall
lat_grid_TR = pd.read_csv("atmos_data\\P.837_MT_Maps\\LAT_MT.TXT", sep="\\s+", header=None).to_numpy() # Y
lon_grid_TR = pd.read_csv("atmos_data\\P.837_MT_Maps\\LON_MT.TXT", sep="\\s+", header=None).to_numpy() # X
lat_list_TR = lat_grid_TR[:,0].flatten()
lon_list_TR = lon_grid_TR[0]
# populate grid data
T_tensor = []
TR_tensor = []
for m in month_indices:
    T_grid = pd.read_csv(f"atmos_data\\ITU-R P1510 mean temp\\T_Month{m}.TXT", sep="\\s+", header=None).to_numpy()
    T_tensor.append(T_grid)
    TR_grid = pd.read_csv(f"atmos_data\\P.837_MT_Maps\\MT_Month{m}.TXT", sep="\\s+", header=None).to_numpy()
    TR_tensor.append(TR_grid)

## FUNCTIONS ------------------------
def T_lookup(lat_deg, lon_deg, month):
    '''
    Given geo coords and month id (this time just regular python list indexing id, 0 = January)
    Return the mean surface temperature there
    '''
    T = gl.grid_lookup(lat_deg, lon_deg, lat_list_T, lon_list_T, T_tensor[month])
    return T

def rain_rate_lookup(lat_deg, lon_deg, month):
    T = T_lookup(lat_deg, lon_deg, month)
    t = T - 273.15 # convert to C
    if t >= 0:
        rr = 0.5874*np.exp(0.0883*t)
    else:
        rr = 0.5874
    return rr

def RT_lookup(lat_deg, lon_deg, month):
    RT = gl.grid_lookup(lat_deg, lon_deg, lat_list_TR, lon_list_TR, TR_tensor[month])
    return RT

def p_exceed_rr_ref(rr_ref, rr_nom, p0):
    if rr_ref == 0:
        return p0
    x = (
        np.log(rr_ref) + 0.7938 - np.log(rr_nom)
    ) / 1.26
    f = lambda t: np.exp(-t**2/2)
    integral = it.quad(f, x, np.inf)[0]
    q = 1/np.sqrt(2*np.pi) * integral
    return p0 * q

def rain_nominal_arrays(lat_deg, lon_deg):
    '''
    returns:
        rr_list: a np array of nominal rain rates for each month. 
        p0_list: a np array of probabilities that there is rain at all for each month. 
    '''

    rr_nom_list = [] # the average rain rate at each month, if it were to rain. (mm/h)
    p0_list = [] # the probability that it would rain at all. i.e. rr > 0

    for m in range(len(month_indices)):
        N = N_days_monthly[m]
        rr = rain_rate_lookup(lat_deg, lon_deg, m)
        RT = RT_lookup(lat_deg, lon_deg, m)
        p0 = RT / (24 * N * rr)
        if p0 > 0.7: # correct for too much rain probability.
            p0 = 0.7
            rr = RT / (24 * N) / 0.7
        rr_nom_list.append(rr)
        p0_list.append(p0)
        
    return np.array(rr_nom_list), np.array(p0_list)

def p_annual(p0_list):
    return np.sum(N_days_monthly*p0_list) / 365.25

def px_annual(rr_t, p0_array, rr_nom_array):
    px_array = np.zeros(len(month_indices))
    for m in range(len(month_indices)):
        N = N_days_monthly[m]
        rr_nom = rr_nom_array[m]
        p0 = p0_array[m]
        px_array[m] = p_exceed_rr_ref(rr_t, rr_nom, p0)
    px_ann = p_annual(px_array)
    return px_ann

def rr_upper_bound(lat_rad, lon_rad, p_rain_desired):
    '''
    returns the rain rate that would be exceeded for p_rain_desired portion of the year. 
    which is equal to the maximum rain rate that needs to be accounted for if we wish to maintain a link for 1-p_rain_desired portion of the year.
    '''
    lat_deg = lat_rad * 180 / np.pi
    lon_deg = lon_rad * 180 / np.pi
    rr_nom_array, p0_array = rain_nominal_arrays(lat_deg, lon_deg)
    p0_ann = p_annual(p0_array)

    # if the probability that there is rain at all is less than the desired probability of rain, 
    # then we guarantee there will be no rain for the period we want to maintain a link.
    if p0_ann < p_rain_desired:
        return 0.0
    
    # adjust the rr_threshold until the probability we calculate is equal to the desired. 
    rr_t = 0.0
    px_ann = p0_ann # px_ann = annual probability of rr exceeding the rr_threshold
    adjustment_rate = 1.0
    iter = 0
    while 100*np.abs(px_ann / p_rain_desired - 1) >= 0.001:
        # adjust the rain rate
        rr_t += adjustment_rate * np.log(px_ann / p_rain_desired)
        px_ann = px_annual(rr_t, p0_array, rr_nom_array)

        if iter > 10000:
            raise Exception("Too long. Could be infinite loop.")
    return rr_t

if __name__ == "__main__":
    print(rr_upper_bound(35*np.pi/180, 19*np.pi/180, 0.0001))
    
