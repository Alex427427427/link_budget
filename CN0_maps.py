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
# beam frequency and bandwidth plans
link_reliability = 0.995 # probability of link closure
link_margin = 3.0 # dB (implementation loss)
roll_off = 0.15 # ASSUMPTION
SR_FORWARD = 42e6 # Mega symbols per second
SR_RETURN = 7e6 # Mega symbols per second
BW_ALLOCATED_FORWARD = SR_FORWARD*(1+roll_off) # Hz
BW_ALLOCATED_RETURN = SR_RETURN*(1+roll_off) # Hz
BW_FORWARD = 250.0e6 # Hz
BW_RETURN = 240.0e6 # Hz
OBO_FORWARD = 3.0 # dB
OBO_RETURN = 4.0 # dB
beam_plans = {
    "fu": {
        "17": 28.32e9, # L
        "22": 28.32e9, # R

        "21": 28.45e9, # L
        "24": 28.45e9, # R

        "25": 28.67e9, # R
        "28": 28.67e9, # L

        "26": 28.94e9, # L
        "30": 28.94e9, # R
    },
    "fd": {
        "22": 19.52e9, # L
        "17": 19.52e9, # R

        "24": 20.28e9, # L
        "21": 20.28e9, # R

        "28": 19.84e9, # R
        "25": 19.84e9, # L

        "30": 21.08e9, # L
        "26": 21.08e9, # R
    },
    "ru": {
        "17": 30.62e9, # L
        "22": 30.62e9, # R

        "21": 30.17e9, # L
        "24": 30.17e9, # R

        "25": 30.62e9, # R
        "28": 30.62e9, # L

        "26": 30.17e9, # L
        "30": 30.17e9, # R
    },
    "rd": {
        "22": 19.12e9, # L
        "17": 19.12e9, # R

        "24": 17.97e9, # L
        "21": 17.97e9, # R

        "28": 19.62e9, # R
        "25": 19.62e9, # L

        "30": 18.09e9, # L
        "26": 18.09e9, # R
    }
}

# gateway specs
gate = {
    'lat': 43.22, # degrees
    'lon': 15.10, # degrees
    'alt': 0.0, # meters
    'PSD': 58.0 - 60, # dBW/Hz
    'EIRP': 58.0 - 60 + 10*np.log10(BW_ALLOCATED_FORWARD), # dBW
    'G/T': 40.0, # dB/K
    'C/IM': 30.0 # dB
}
# user terminal specs
uterm = {
    'lat': 43.22, # degrees
    'lon': 15.10, # degrees
    'alt': 0.3048 * 30000, # meters
    'EIRP': 57.0, # dBW
    'G/T': 17.0 # dB/K
}
# satellite specs
sat = {
    'lat': 0, # degrees
    'lon': 11.0, # degrees
    'alt': lb.R_GEO - lb.R_EQUATOR, # meters
    'EIRP_f': 0.0, # dBW
    'G/T_f': 0.0, # dB/K
    'EIRP_r': 0.0, # dBW
    'G/T_r': 0.0, # dB/K
}


# geographic coords of the user terminal
ulat_list = np.linspace(30, 60, 31) # degrees
ulon_list = np.linspace(0, 40, 81) # degrees
CN0_fmap = np.zeros((len(ulat_list), len(ulon_list)))
CN0_rmap = np.zeros((len(ulat_list), len(ulon_list)))

for c, ulon in enumerate(ulon_list):
    print("========================")
    print(f"Calculating for LON = {ulon}...")
    for r, ulat in enumerate(ulat_list):
        
        uterm['lat'] = ulat
        uterm['lon'] = ulon
        # satellite specs
        EIRP_sat_f, beam_id_fd = kgq.highest_EIRP_query(uterm['lat'], uterm['lon']) # dBW
        GT_sat_f, beam_id_fu = kgq.highest_GT_query(gate['lat'], gate['lon']) # dB/K
        EIRP_sat_r, beam_id_rd = kgq.highest_EIRP_query(gate['lat'], gate['lon']) # dBW
        GT_sat_r, beam_id_ru = kgq.highest_GT_query(uterm['lat'], uterm['lon']) # dB/K
        if EIRP_sat_f is None or GT_sat_f is None or EIRP_sat_r is None or GT_sat_r is None:
            # set the maps to -inf
            CN0_fmap[r, c] = -np.inf
            CN0_rmap[r, c] = -np.inf
            continue
        # satellite specs
        sat['EIRP_f'] = EIRP_sat_f - OBO_FORWARD + np.log10(BW_ALLOCATED_FORWARD/BW_FORWARD) # dBW
        sat['G/T_f'] = GT_sat_f # dB/K
        sat['EIRP_r'] = EIRP_sat_r - OBO_RETURN + np.log10(BW_ALLOCATED_RETURN/BW_RETURN)
        sat['G/T_r'] = GT_sat_r

        try: 
            # Communication specs
            F_UP_FORWARD = beam_plans["fu"][str(beam_id_fu)] # uplink frequency in Hz
            F_DOWN_FORWARD = beam_plans["fd"][str(beam_id_fd)] # downlink frequency in Hz
            F_UP_RETURN = beam_plans["ru"][str(beam_id_ru)] # uplink frequency in Hz
            F_DOWN_RETURN = beam_plans["rd"][str(beam_id_rd)]# downlink frequency in Hz
        except KeyError as e:
            # set the maps to -inf
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
            BW=BW_ALLOCATED_FORWARD, # bandwidth in Hz
            link_reliability=link_reliability, # probability of link closure
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
            BW=BW_ALLOCATED_RETURN, # bandwidth in Hz
            link_reliability=link_reliability, # probability of link closure
            #bit_rate=BIT_RATE, # bit rate in bps
            C_IM=gate['C/IM'] # dB
        )

        CN0_fmap[r, c] = forward_results['C/N0_dB']
        CN0_rmap[r, c] = return_results['C/N0_dB']

# save data
np.savetxt('outputs\\CN0_fmap.csv', CN0_fmap, fmt='%.4f', delimiter=',')
np.savetxt('outputs\\CN0_rmap.csv', CN0_rmap, fmt='%.4f', delimiter=',')
# create lat and lon tables
ulon_grid, ulat_grid = np.meshgrid(ulon_list, ulat_list)
np.savetxt('outputs\\LON.csv', ulon_grid, fmt='%.2f', delimiter=',')
np.savetxt('outputs\\LAT.csv', ulat_grid, fmt='%.2f', delimiter=',')

