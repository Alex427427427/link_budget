import numpy as np
import matplotlib.pyplot as plt
import kml_gain_query as kgq
import L_atmos as la
import L_rain as lr
import modcod_select as ms
import rain_rate as rrb

C_FREE_SPACE = 299792458.0 # m/s
R_GEO = 42164000 # m
R_EQUATOR = 6378000 # m semi-major axis of earth
R_POLAR = 6357000 # m semi-minor axis of earth
ECCENTRICITY_EARTH = np.sqrt(1 - R_POLAR**2 / R_EQUATOR**2) # eccentricity of earth
FLATTENING_EARTH = 1 - R_POLAR / R_EQUATOR
K_DBW_per_HZ_K = -228.6 # Boltzmann's constant


def dB_from_power(value):
    return 10*np.log10(value)

def dB_from_amplitude(value):
    return 20*np.log10(value)

def amplitude_from_dB(dB):
    return 10**(dB/20.0)

def power_from_dB(dB):
    return 10**(dB/10.0)

def geo_to_ECEF(lat, lon, alt):
    '''
    Converts geodetic coordinate (latitude, longitude, altitude) to Earth Centered Earth Fixed Coordinates (cartesian)
    ''' 
    prime_vertical_radius = R_EQUATOR / np.sqrt(1 - ECCENTRICITY_EARTH**2 * np.sin(lat)**2)
    x = (prime_vertical_radius + alt)*np.cos(lat)*np.cos(lon)
    y = (prime_vertical_radius + alt)*np.cos(lat)*np.sin(lon)
    z = ((1-ECCENTRICITY_EARTH**2)*prime_vertical_radius + alt)*np.sin(lat)
    return x, y, z

def elevation(lat1, lon1, alt1, lat2, lon2, alt2):
    '''
    Elevation pointing from ground station to geo satellite
    '''
    x1, y1, z1 = geo_to_ECEF(lat1, lon1, alt1)
    x2, y2, z2 = geo_to_ECEF(lat2, lon2, alt2)
    # calculate the distance between the two points
    d = dist(x1, y1, z1, x2, y2, z2)
    # calculate the elevation angle using vector geometry
    r1 = np.array([x1, y1, z1])
    r2 = np.array([x2, y2, z2])
    v = r2 - r1 
    v_norm = np.linalg.norm(v)
    v_unit = v / v_norm
    r1_unit = r1 / np.linalg.norm(r1)
    elevation_angle = np.arcsin(np.dot(v_unit, r1_unit))
    return elevation_angle


def dist(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def FSPL(d, f):
    return 20*np.log10(4*np.pi*d*f/C_FREE_SPACE)

def noise_power_per_T(BW):
    '''
    BW: bandwidth in Hz
    returns: power in dBW per K
    '''
    return K_DBW_per_HZ_K + 10*np.log10(BW)

def atmospheric_loss(lat_gnd, lon_gnd, alt_gnd, lat_sat, lon_sat, alt_sat, f):
    ''' 
    using simple model for atmospheric loss
    ITU-R P.676-13
    lat_gnd, lon_gnd, alt_gnd: ground station coordinates in radians and meter
    lat_sat, lon_sat, alt_sat: satellite coordinates in radians and meter
    f: frequency in Hz
    returns: atmospheric loss in dB
    '''
    # convert units
    f_ghz = f / 1e9 # GHz
    alt_gnd_km = alt_gnd / 1000.0 # km
    # calculate the elevation angle
    elevation_angle = elevation(lat_gnd, lon_gnd, alt_gnd, lat_sat, lon_sat, alt_sat)
    L_atm = la.total_atten_integrated(f_ghz, elevation_angle, initial_h = alt_gnd_km) # dB
    return L_atm

def rain_loss(lat_gnd, lon_gnd, alt_gnd, lat_sat, lon_sat, alt_sat, f, p_angle, rr=10.0):
    f_ghz = f/1e9
    elev_angle = elevation(lat_gnd, lon_gnd, alt_gnd, lat_sat, lon_sat, alt_sat)
    return lr.rain_total_atten(f_ghz, rr, lat_gnd, lon_gnd, elev_angle, p_angle)

def single_link_CNR(EIRP_tx, L_total, G_per_T_rx, noise_power_per_T):
    return EIRP_tx - L_total + G_per_T_rx - noise_power_per_T

def total_CNR(CNR_list):
    sum_of_recip = 0.0
    for CNR in CNR_list:
        sum_of_recip += 10**(-CNR/10.0)
    recip_of_sum = 1.0 / sum_of_recip
    return 10*np.log10(recip_of_sum)

def full_link_budget(
        EIRP_Earth_start, Earth_start_lat_lon_alt, 
        sat_lat_lon_alt, G_per_T_sat, EIRP_sat, 
        Earth_end_lat_lon_alt, G_per_T_Earth_end, 
        f_up, f_down, BW, 
        link_reliability=0.999, # link reliability
        C_IM=10000
    ):
    '''
    Calculates the receive power in dBW at the Earth_end terminal in the forward link (Earth_start -> sat -> Earth_end)
    '''
    Earth_start_x, Earth_start_y, Earth_start_z = geo_to_ECEF(Earth_start_lat_lon_alt[0], Earth_start_lat_lon_alt[1], Earth_start_lat_lon_alt[2])
    sat_x, sat_y, sat_z = geo_to_ECEF(sat_lat_lon_alt[0], sat_lat_lon_alt[1], sat_lat_lon_alt[2])
    Earth_end_x, Earth_end_y, Earth_end_z = geo_to_ECEF(Earth_end_lat_lon_alt[0], Earth_end_lat_lon_alt[1], Earth_end_lat_lon_alt[2])
    dist_up = dist(Earth_start_x, Earth_start_y, Earth_start_z, sat_x, sat_y, sat_z)
    dist_down = dist(sat_x, sat_y, sat_z, Earth_end_x, Earth_end_y, Earth_end_z)
    L_up_path = FSPL(dist_up, f_up) # dB
    L_down_path = FSPL(dist_down, f_down) # dB

    L_atm_up = atmospheric_loss(
        Earth_start_lat_lon_alt[0], Earth_start_lat_lon_alt[1], Earth_start_lat_lon_alt[2],
        sat_lat_lon_alt[0], sat_lat_lon_alt[1], sat_lat_lon_alt[2],
        f_up
    ) # dB, atmospheric loss in uplink
    L_atm_down = atmospheric_loss(
        Earth_end_lat_lon_alt[0], Earth_end_lat_lon_alt[1], Earth_end_lat_lon_alt[2],
        sat_lat_lon_alt[0], sat_lat_lon_alt[1], sat_lat_lon_alt[2],
        f_down
    ) # dB, atmospheric loss in downlink

    p_rain = 1 - link_reliability # probability of rain = 0.1% of time
    rr_up = rrb.rr_upper_bound(Earth_start_lat_lon_alt[0], Earth_start_lat_lon_alt[1], p_rain) # mm/h
    rr_down = rrb.rr_upper_bound(Earth_end_lat_lon_alt[0], Earth_end_lat_lon_alt[1], p_rain) # mm/h
    L_rain_up = rain_loss(
        Earth_start_lat_lon_alt[0], Earth_start_lat_lon_alt[1], Earth_start_lat_lon_alt[2],
        sat_lat_lon_alt[0], sat_lat_lon_alt[1], sat_lat_lon_alt[2],
        f_up, np.pi/4, rr_up
    )
    L_rain_down = rain_loss(
        Earth_end_lat_lon_alt[0], Earth_end_lat_lon_alt[1], Earth_end_lat_lon_alt[2],
        sat_lat_lon_alt[0], sat_lat_lon_alt[1], sat_lat_lon_alt[2],
        f_down, np.pi/4, rr_down
    )

    noise = noise_power_per_T(BW) # dBW/K

    L_up_total = L_up_path + L_atm_up + L_rain_up # dB
    L_down_total = L_down_path + L_atm_down + L_rain_down # dB

    uplink_CNR = single_link_CNR(EIRP_Earth_start, L_up_total, G_per_T_sat, noise) # dB
    downlink_CNR = single_link_CNR(EIRP_sat, L_down_total, G_per_T_Earth_end, noise) # dB

    # final results
    total_CNR_dB = total_CNR([uplink_CNR, downlink_CNR, C_IM]) # dB
    CNR_total_value = power_from_dB(total_CNR_dB) # linear value
    CN0_total_value = CNR_total_value * BW # Hz
    CN0_dB = dB_from_power(CN0_total_value) # dB-Hz
    #EbN0_total_value = CN0_total_value / bit_rate # dimensionless ratio
    #EbN0_dB = dB_from_power(EbN0_total_value) # dB

    return {
        'FSPL_up_dB': L_up_path,
        'FSPL_down_dB': L_down_path,
        'L_atm_up_dB': L_atm_up,
        'L_atm_down_dB': L_atm_down,
        'L_rain_up_dB': L_rain_up,
        'L_rain_down_dB': L_rain_down,
        #'bit_rate_Mbps': bit_rate*10**(-6), # bps
        'BW_MHz': BW*10**(-6),
        'CNR_dB': total_CNR_dB,
        'CN0_dB': CN0_dB,
        #'Eb/N0_dB': EbN0_dB,
    }

# main code
if __name__ == "__main__":

    # Communication specs
    BW = 230e6 # bandwidth in Hz
    #BIT_RATE = 3e8 # bit rate in bps
    F_UP_FORWARD = 28.12e9 # uplink frequency in Hz
    F_DOWN_FORWARD = F_UP_FORWARD - 8.3e9 # downlink frequency in Hz
    F_UP_RETURN = 29.62e9 # uplink frequency in Hz
    F_DOWN_RETURN = F_UP_RETURN - 11.30e9 # downlink frequency in Hz
    link_reliability = 0.999
    link_margin = 10.0 # dB
    roll_off = 0.1 
    

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
    EIRP_sat = kgq.highest_EIRP_query(uterm['lat'], uterm['lon']) # dBW
    GT_sat = kgq.highest_GT_query(gate['lat'], gate['lon']) # dB/K
    #EIRP_sat = 63.0 # dBW
    #GT_sat = 18.0 # dB/K
    # satellite specs
    sat = {
        'lat': 0, # degrees
        'lon': 31.0, # degrees
        'alt': R_GEO - R_EQUATOR, # meters
        'EIRP': EIRP_sat, # dBW
        'G/T': GT_sat # dB/K
    }
    
    if sat['EIRP'] is None or sat['G/T'] is None:
        print("No link found for the given coordinates.")
    else:
        forward_results = full_link_budget(
            EIRP_Earth_start=gate['EIRP'], # dBW
            Earth_start_lat_lon_alt=(np.radians(gate['lat']), np.radians(gate['lon']), gate['alt']), # lat, lon, alt in radians and meters
            sat_lat_lon_alt=(np.radians(sat['lat']), np.radians(sat['lon']), sat['alt']), # lat, lon, alt in radians and meters
            G_per_T_sat=sat['G/T'], # dB/K
            EIRP_sat=sat['EIRP'], # dBW
            Earth_end_lat_lon_alt=(np.radians(uterm['lat']), np.radians(uterm['lon']), uterm['alt']), # lat, lon, alt in radians and meters
            G_per_T_Earth_end=uterm['G/T'], # dB/K
            f_up=F_UP_FORWARD, # frequency in Hz
            f_down=F_DOWN_FORWARD, # frequency in Hz
            BW=BW, # bandwidth in Hz
            #bit_rate=BIT_RATE, # bit rate in bps
            C_IM=gate['C/IM'] # dB
        )

        return_results = full_link_budget(
            EIRP_Earth_start=uterm['EIRP'], # dBW
            Earth_start_lat_lon_alt=(np.radians(uterm['lat']), np.radians(uterm['lon']), uterm['alt']), # lat, lon, alt in radians and meters
            sat_lat_lon_alt=(np.radians(sat['lat']), np.radians(sat['lon']), sat['alt']), # lat, lon, alt in radians and meters
            G_per_T_sat=sat['G/T'], # dB/K
            EIRP_sat=sat['EIRP'], # dBW
            Earth_end_lat_lon_alt=(np.radians(gate['lat']), np.radians(gate['lon']), gate['alt']), # lat, lon, alt in radians and meters
            G_per_T_Earth_end=gate['G/T'], # dB/K
            f_up=F_UP_RETURN, # frequency in Hz
            f_down=F_DOWN_RETURN, # frequency in Hz
            BW=BW, # bandwidth in Hz
            #bit_rate=BIT_RATE, # bit rate in bps
            C_IM=gate['C/IM'] # dB
        )

        print("\nLink Budget Calculation Results")
        print("=====================================")

        print("Forward Link: ")
        print("=====================================")
        for k, v in forward_results.items():
            print(f"{k}: {v}")

        print("\n")
        
        print("Return Link: ")
        print("=====================================")
        for k, v in return_results.items():
            print(f"{k}: {v}")

        # given the Eb/N0, find the MODCOD
        forward_CN0 = forward_results['CN0_dB']
        return_CN0 = return_results['CN0_dB']
        forward_MOD, forward_COD, forward_SE = ms.modcod_select(forward_CN0, link_margin=link_margin)
        return_MOD, return_COD, return_SE = ms.modcod_select(return_CN0, link_margin=link_margin)

        if forward_SE is None or return_SE is None:
            print("No link found for the given coordinates.")
            exit(0)

        forward_bitrate = BW/(1+roll_off) * forward_SE
        return_bitrate = BW/(1+roll_off) * return_SE
        forward_EbN0 = forward_CN0 - 10*np.log10(forward_bitrate)
        return_EbN0 = return_CN0 - 10*np.log10(return_bitrate)

        print("\n")
        print("Forward Link MODCOD: ")
        print("=====================================")
        print(f"MOD: {forward_MOD}")
        print(f"COD: {forward_COD}")
        print(f"link margin: {link_margin} dB")
        print(f"bitrate: {forward_bitrate} bps")
        print(f"Eb/N0: {forward_EbN0} dB")
        print("\n")
        print("Return Link MODCOD: ")
        print("=====================================")
        print(f"MOD: {return_MOD}")
        print(f"COD: {return_COD}")
        print(f"link margin: {link_margin} dB")
        print(f"bitrate: {return_bitrate} bps")
        print(f"Eb/N0: {return_EbN0} dB")

        print("\n")
