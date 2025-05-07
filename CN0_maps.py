import main_calc as lb
import numpy as np
import kml_gain_query as kgq
import modcod_select as ms
import matplotlib.pyplot as plt
import numpy.ma as ma

'''
to produce an Eb/N0 map for the forward and return links for fixed ground station, varying user terminal. 
adaptive modcod select.
'''
# constants
C_FREE_SPACE = 299792458.0 # m/s
R_GEO = 42164000 # m
R_EQUATOR = 6378000 # m semi-major axis of earth
R_POLAR = 6357000 # m semi-minor axis of earth
ECCENTRICITY_EARTH = np.sqrt(1 - R_POLAR**2 / R_EQUATOR**2) # eccentricity of earth
FLATTENING_EARTH = 1 - R_POLAR / R_EQUATOR
K_DBW_per_HZ_K = -228.6 # Boltzmann's constant

# Communication specs
BW = 230e6 # bandwidth in Hz
BIT_RATE = 3e8 # bit rate in bps
F_UP_FORWARD = 28.12e9 # uplink frequency in Hz
F_DOWN_FORWARD = F_UP_FORWARD - 8.3e9 # downlink frequency in Hz
F_UP_RETURN = 29.62e9 # uplink frequency in Hz
F_DOWN_RETURN = F_UP_RETURN - 11.30e9 # downlink frequency in Hz
link_reliability = 0.995

# gateway specs
gate = {
    'lat': 36, # degrees
    'lon': 19, # degrees
    'alt': 0.0, # meters
    'PSD': 57.5e-6, # dBW/Hz
    'EIRP': 57.5e-6 + 10*np.log10(BW), # dBW
    'G/T': 38.2, # dB/K
    'C/IM': 20.0 # dB
}
# user terminal specs
uterm = {
    'lat': 36, # degrees
    'lon': 19, # degrees
    'alt': 0.3048 * 30000, # meters
    'EIRP': 58.5, # dBW
    'G/T': 15.0 # dB/K
}
EIRP_sat_f = kgq.highest_EIRP_query(uterm['lat'], uterm['lon']) # dBW
GT_sat_f = kgq.highest_GT_query(gate['lat'], gate['lon']) # dB/K
EIRP_sat_r = kgq.highest_EIRP_query(gate['lat'], gate['lon']) # dBW
GT_sat_r = kgq.highest_GT_query(uterm['lat'], uterm['lon']) # dB/K
# satellite specs
sat = {
    'lat': 0, # degrees
    'lon': 31.0, # degrees
    'alt': R_GEO - R_EQUATOR, # meters
    'EIRP_f': EIRP_sat_f, # dBW
    'G/T_f': GT_sat_f, # dB/K
    'EIRP_r': EIRP_sat_r, # dBW
    'G/T_r': GT_sat_r # dB/K
}

# geographic coords of the user terminal
ulat_list = np.linspace(-90, 90, 181) # degrees
ulon_list = np.linspace(-180, 180, 361) # degrees
CN0_fmap = np.zeros((len(ulat_list), len(ulon_list)))
CN0_rmap = np.zeros((len(ulat_list), len(ulon_list)))

'''
for r, ulat in enumerate(ulat_list):
    print("========================")
    print(f"Calculating for LAT = {ulat}...")
    for c, ulon in enumerate(ulon_list):
        print(f"LON: {ulon}")
    '''
for c, ulon in enumerate(ulon_list):
    print("========================")
    print(f"Calculating for LON = {ulon}...")
    for r, ulat in enumerate(ulat_list):
        
        uterm['lat'] = ulat
        uterm['lon'] = ulon
        sat['EIRP_f'] = kgq.highest_EIRP_query(uterm['lat'], uterm['lon']) # dBW
        sat['G/T_r'] = kgq.highest_GT_query(uterm['lat'], uterm['lon']) # dB/K

        if sat['EIRP_f'] is None or sat['G/T_f'] is None or sat['EIRP_r'] is None or sat['G/T_r'] is None:
            CN0_fmap[r, c] = -np.inf
            CN0_rmap[r, c] = -np.inf
            continue
        print(f"LAT: {ulat}")

        forward_results = lb.full_link_budget(
            EIRP_Earth_start=gate['EIRP'], # dBW
            Earth_start_lat_lon_alt=(np.radians(gate['lat']), np.radians(gate['lon']), gate['alt']), # lat, lon, alt in radians and meters
            sat_lat_lon_alt=(np.radians(sat['lat']), np.radians(sat['lon']), sat['alt']), # lat, lon, alt in radians and meters
            G_per_T_sat=sat['G/T_f'], # dB/K
            EIRP_sat=sat['EIRP_f'], # dBW
            Earth_end_lat_lon_alt=(np.radians(uterm['lat']), np.radians(uterm['lon']), uterm['alt']), # lat, lon, alt in radians and meters
            G_per_T_Earth_end=uterm['G/T'], # dB/K
            f_up=F_UP_FORWARD, # frequency in Hz
            f_down=F_DOWN_FORWARD, # frequency in Hz
            BW=BW, # bandwidth in Hz
            link_reliability=link_reliability, # link reliability
            #bit_rate=BIT_RATE, # bit rate in bps
            C_IM=gate['C/IM'] # dB
        )

        return_results = lb.full_link_budget(
            EIRP_Earth_start=uterm['EIRP'], # dBW
            Earth_start_lat_lon_alt=(np.radians(uterm['lat']), np.radians(uterm['lon']), uterm['alt']), # lat, lon, alt in radians and meters
            sat_lat_lon_alt=(np.radians(sat['lat']), np.radians(sat['lon']), sat['alt']), # lat, lon, alt in radians and meters
            G_per_T_sat=sat['G/T_r'], # dB/K
            EIRP_sat=sat['EIRP_r'], # dBW
            Earth_end_lat_lon_alt=(np.radians(gate['lat']), np.radians(gate['lon']), gate['alt']), # lat, lon, alt in radians and meters
            G_per_T_Earth_end=gate['G/T'], # dB/K
            f_up=F_UP_RETURN, # frequency in Hz
            f_down=F_DOWN_RETURN, # frequency in Hz
            BW=BW, # bandwidth in Hz
            link_reliability=link_reliability, # link reliability
            #bit_rate=BIT_RATE, # bit rate in bps
            C_IM=gate['C/IM'] # dB
        )

        CN0_fmap[r, c] = forward_results['CN0_dB']
        CN0_rmap[r, c] = return_results['CN0_dB']

# save data
np.savetxt('outputs\\CN0_fmap.csv', CN0_fmap, fmt='%.4f', delimiter=',')
np.savetxt('outputs\\CN0_rmap.csv', CN0_rmap, fmt='%.4f', delimiter=',')
# create lat and lon tables
ulon_grid, ulat_grid = np.meshgrid(ulon_list, ulat_list)
np.savetxt('outputs\\LON.csv', ulon_grid, fmt='%.2f', delimiter=',')
np.savetxt('outputs\\LAT.csv', ulat_grid, fmt='%.2f', delimiter=',')

