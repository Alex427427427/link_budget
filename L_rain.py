'''
Created by alexander Li on 2025/04/29
'''
import numpy as np
import pandas as pd
import grid_lookup as gl

## DATA -------------------------------------------
# rain coeffs
rain_coeffs = {
    "k_h": {
        "a": [-5.33980, -0.35351, -0.23789, -0.94158],
        "b": [-0.10008, 1.26970, 0.86036, 0.64552],
        "c": [1.13098, 0.45400, 0.15354, 0.16817],
        "mk": -0.18961,
        "ck": 0.71147
    },
    "k_v": {
        "a": [-3.80595, -3.44965, -0.39902, 0.50167],
        "b": [0.56934, -0.22911, 0.73042, 1.07319],
        "c": [0.81061, 0.51059, 0.11899, 0.27195],
        "mk": -0.16398,
        "ck": 0.63297
    },
    "alpha_h": {
        "a": [-0.14318, 0.29591, 0.32177, -5.37610, 16.1721],
        "b": [1.82442, 0.77564, 0.63773, -0.96230, -3.29980],
        "c": [-0.55187, 0.19822, 0.13164, 1.47828, 3.43990],
        "ma": 0.67849,
        "ca": -1.95537
    },
    "alpha_v": {
        "a": [-0.07771, 0.56727, -0.20238, -48.2991, 48.5833],
        "b": [2.33840, 0.95545, 1.14520, 0.791669, 0.791459],
        "c": [-0.76284, 0.54039, 0.26809, 0.116226, 0.116479],
        "ma": -0.053739,
        "ca": 0.83433
    }
}
# rain heights
lat_grid_rain_height = pd.read_csv("atmos_data\\rain_height_itu\\Lat.txt", sep="\\s+", header=None).to_numpy() # Y
lon_grid_rain_height = pd.read_csv("atmos_data\\rain_height_itu\\Lon.txt", sep="\\s+", header=None).to_numpy() # X
h0_grid = pd.read_csv("atmos_data\\rain_height_itu\\h0.txt", sep="\\s+", header=None).to_numpy() # Z
lat_list_rain_height = lat_grid_rain_height[:,0].flatten()
lon_list_rain_height = lon_grid_rain_height[0]


## FUNCITONS -----------------------------------------------
def k_calc(f, polarisation):
    '''
    calculate a parameter k in the rain attenuation model
    f: frequency in GHz
    polarisation: str, either 'h' for horizontal or 'v' for vertical polarisation 
    '''
    if polarisation not in ['h', 'v']:
        raise ValueError("Polarisation must be either 'h' or 'v'")
    if f < 0:
        raise ValueError("Frequency must be positive")
    # read coefficients from the dictionary
    if polarisation == 'h':
        a_list = np.array(rain_coeffs["k_h"]["a"])
        b_list = np.array(rain_coeffs["k_h"]["b"])
        c_list = np.array(rain_coeffs["k_h"]["c"])
        mk = rain_coeffs["k_h"]["mk"]
        ck = rain_coeffs["k_h"]["ck"]
    elif polarisation == 'v':
        a_list = np.array(rain_coeffs["k_v"]["a"])
        b_list = np.array(rain_coeffs["k_v"]["b"])
        c_list = np.array(rain_coeffs["k_v"]["c"])
        mk = rain_coeffs["k_v"]["mk"]
        ck = rain_coeffs["k_v"]["ck"]
    # calculate
    l10k = ck + mk*np.log10(f) + np.sum(
        a_list*np.exp(-((np.log10(f)-b_list)/c_list)**2)
    )
    k = 10**l10k
    return k

def alpha_calc(f, polarisation):
    '''
    calculate a parameter alpha in the rain attenuation model
    f: frequency in GHz
    polarisation: str, either 'h' for horizontal or 'v' for vertical polarisation 
    '''
    if polarisation not in ['h', 'v']:
        raise ValueError("Polarisation must be either 'h' or 'v'")
    if f < 0:
        raise ValueError("Frequency must be positive")
    # read coefficients from the dictionary
    if polarisation == 'h':
        a_list = np.array(rain_coeffs["alpha_h"]["a"])
        b_list = np.array(rain_coeffs["alpha_h"]["b"])
        c_list = np.array(rain_coeffs["alpha_h"]["c"])
        ma = rain_coeffs["alpha_h"]["ma"]
        ca = rain_coeffs["alpha_h"]["ca"]
    elif polarisation == 'v':
        a_list = np.array(rain_coeffs["alpha_v"]["a"])
        b_list = np.array(rain_coeffs["alpha_v"]["b"])
        c_list = np.array(rain_coeffs["alpha_v"]["c"])
        ma = rain_coeffs["alpha_v"]["ma"]
        ca = rain_coeffs["alpha_v"]["ca"]
    # calculate
    alpha = ca + ma*np.log10(f) + np.sum(
        a_list*np.exp(-((np.log10(f)-b_list)/c_list)**2)
    )
    return alpha

def rain_spec_atten(f, rr, elev_angle=np.pi/2, p_angle=np.pi/4):
    """
    Calculate the specific attenuation due to rain for a given frequency and rain rate.
    
    Parameters:
    f : float
        Frequency in GHz.
    rr : float
        Rain rate in mm/h.
    p_angle : float
        Polarization angle in radians (default is pi/4 for circular polarization). Must be between 0 and pi/2.
    elev_angle : float
        Elevation angle in radians (default is pi/2 for vertical polarization). Must be between 0 and pi/2.

    Returns:
    float
        Specific attenuation in dB/km.
    """
    if f < 0 or rr < 0:
        raise ValueError("Frequency and rain rate must be positive")
    if p_angle < 0 or p_angle > np.pi/2:
        raise ValueError("Polarization angle must be between 0 and pi/2")
    if elev_angle < 0 or elev_angle > np.pi/2:
        raise ValueError("Elevation angle must be between 0 and pi/2")
    if elev_angle == 0:
        raise ValueError("Elevation angle must be greater than 0")
    
    kh = k_calc(f, 'h')
    kv = k_calc(f, 'v')
    alpha_h = alpha_calc(f, 'h')
    alpha_v = alpha_calc(f, 'v')
    k = (kh + kv + (kh-kv)*np.cos(elev_angle)**2 * np.cos(2*p_angle))/2
    alpha = (kh*alpha_h + kv*alpha_v + (kh*alpha_h - kv*alpha_v)*np.cos(elev_angle)**2 * np.cos(2*p_angle))/(2*k)
    # Calculate specific attenuation
    gamma = k * rr**alpha
    return gamma

def rain_total_atten(f, rr, gnd_lat_rad, gnd_lon_rad, elev_angle=np.pi/2, p_angle=np.pi/4):
    if elev_angle <= 0.0:
        raise ValueError("elevation angle must be positive.")
    gnd_lat_deg = gnd_lat_rad * 180.0 / np.pi
    gnd_lon_deg = gnd_lon_rad * 180.0 / np.pi

    gamma = rain_spec_atten(f, rr, elev_angle, p_angle)
    rain_height = gl.grid_lookup(gnd_lat_deg, gnd_lon_deg, lat_list_rain_height, lon_list_rain_height, h0_grid)
    slant_path_length = rain_height / np.sin(elev_angle)
    
    return gamma*slant_path_length


