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
        link_reliability=0.0, # link reliability
        C_IM=10000.0, # dB, carrier to interference ratio
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
    #print(f"dist_up: {dist_up/1e3:.2f} km, dist_down: {dist_down/1e3:.2f} km, L_up: {L_up_path:.2f} dB, L_down: {L_down_path:.2f} dB")
    
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

    if link_reliability < 0.5:
        L_rain_up = 0.0 # dB, rain loss in uplink
        L_rain_down = 0.0
    else:
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

    #print(f"uplink CNR: {uplink_CNR:.2f} dB, downlink CNR: {downlink_CNR:.2f} dB")

    # final results
    total_CNR_dB = total_CNR([uplink_CNR, downlink_CNR, C_IM]) # dB
    CNR_total_value = power_from_dB(total_CNR_dB) # linear value
    CN0_total_value = CNR_total_value * BW # Hz
    CN0_dB = dB_from_power(CN0_total_value) # dB-Hz
    #EbN0_total_value = CN0_total_value / bit_rate # dimensionless ratio
    #EbN0_dB = dB_from_power(EbN0_total_value) # dB

    #print(f"total CNR: {total_CNR_dB:.2f} dB, CN0: {CN0_dB:.2f} dB-Hz")

    return {
        'FSPL_up_dB': L_up_path,
        'FSPL_down_dB': L_down_path,
        'L_atm_up_dB': L_atm_up,
        'L_atm_down_dB': L_atm_down,
        'L_rain_up_dB': L_rain_up,
        'L_rain_down_dB': L_rain_down,
        #'bit_rate_Mbps': bit_rate*10**(-6), # bps
        #'BW_MHz': BW*10**(-6),
        'CNR_dB': total_CNR_dB,
        'C/N0_dB': CN0_dB,
        #'Eb/N0_dB': EbN0_dB,
    }

# main code
if __name__ == "__main__":

    # beam frequency and bandwidth plans
    link_reliability = 0.995 # probability of link closure
    link_margin = 1.0 # dB (implementation loss)
    roll_off = 0.1 # ASSUMPTION
    SR_FORWARD = 52e6 # Mega symbols per second
    SR_RETURN = 10e6 # Mega symbols per second
    BW_ALLOCATED_FORWARD = SR_FORWARD*(1+roll_off) # Hz
    BW_ALLOCATED_RETURN = SR_RETURN*(1+roll_off) # Hz
    BW_FORWARD = 230.0e6 # Hz
    BW_RETURN = 220.0e6 # Hz
    OBO_FORWARD = 3.5 # dB
    OBO_RETURN = 5.8 # dB
    beam_plans = {
        "fu": {
            "17": 28.12e9, # L
            "22": 28.12e9, # R

            "21": 28.38e9, # L
            "24": 28.38e9, # R

            "25": 28.67e9, # R
            "28": 28.67e9, # L

            "26": 28.93e9, # L
            "30": 28.93e9, # R
        },
        "fd": {
            "22": 19.82e9, # L
            "17": 19.82e9, # R

            "24": 20.08e9, # L
            "21": 20.08e9, # R

            "28": 19.82e9, # R
            "25": 19.82e9, # L

            "30": 20.08e9, # L
            "26": 20.08e9, # R
        },
        "ru": {
            "17": 29.62e9, # L
            "22": 29.62e9, # R

            "21": 29.87e9, # L
            "24": 29.87e9, # R

            "25": 29.62e9, # R
            "28": 29.62e9, # L

            "26": 29.87e9, # L
            "30": 29.87e9, # R
        },
        "rd": {
            "22": 18.32e9, # L
            "17": 18.32e9, # R

            "24": 18.57e9, # L
            "21": 18.57e9, # R

            "28": 18.82e9, # R
            "25": 18.82e9, # L

            "30": 19.07e9, # L
            "26": 19.07e9, # R
        }
    }

    # gateway specs
    gate = {
        'lat': 33.22, # degrees
        'lon': 35.10, # degrees
        'alt': 0.0, # meters
        'PSD': 57.5 - 60, # dBW/Hz
        'EIRP': 57.5 - 60 + 10*np.log10(BW_ALLOCATED_FORWARD), # dBW
        'G/T': 38.2, # dB/K
        'C/IM': 20.0 # dB
    }
    # user terminal specs
    uterm = {
        'lat': 33.22, # degrees
        'lon': 35.10, # degrees
        'alt': 0.3048 * 30000, # meters
        'EIRP': 58.5, # dBW
        'G/T': 15.0 # dB/K
    }
    # satellite specs
    EIRP_sat_f, beam_id_fd = kgq.highest_EIRP_query(uterm['lat'], uterm['lon']) # dBW
    GT_sat_f, beam_id_fu = kgq.highest_GT_query(gate['lat'], gate['lon']) # dB/K
    EIRP_sat_r, beam_id_rd = kgq.highest_EIRP_query(gate['lat'], gate['lon']) # dBW
    GT_sat_r, beam_id_ru = kgq.highest_GT_query(uterm['lat'], uterm['lon']) # dB/K
    if EIRP_sat_f is None or GT_sat_f is None or EIRP_sat_r is None or GT_sat_r is None:
        print("No link found for the given coordinates.")
        exit(0)
    # satellite specs
    sat = {
        'lat': 0, # degrees
        'lon': 31.0, # degrees
        'alt': R_GEO - R_EQUATOR, # meters
        'EIRP_f': EIRP_sat_f - OBO_FORWARD + np.log10(BW_ALLOCATED_FORWARD/BW_FORWARD), # dBW
        'G/T_f': GT_sat_f, # dB/K
        'EIRP_r': EIRP_sat_r - OBO_RETURN + np.log10(BW_ALLOCATED_RETURN/BW_RETURN) , # dBW
        'G/T_r': GT_sat_r, # dB/K
        'beam_id_fd': beam_id_fd, # beam id for forward downward link
        'beam_id_fu': beam_id_fu, # beam id for forward upward link
        'beam_id_rd': beam_id_rd, # beam id for return downward link
        'beam_id_ru': beam_id_ru, # beam id for return upward link
    }
    

    try: 
        # Communication specs
        F_UP_FORWARD = beam_plans["fu"][str(beam_id_fu)] # uplink frequency in Hz
        F_DOWN_FORWARD = beam_plans["fd"][str(beam_id_fd)] # downlink frequency in Hz
        F_UP_RETURN = beam_plans["ru"][str(beam_id_ru)] # uplink frequency in Hz
        F_DOWN_RETURN = beam_plans["rd"][str(beam_id_rd)]# downlink frequency in Hz
    except KeyError as e:
        print(f"No beam plan found for the beam id. Please check the beam plans.")
        exit(0)


    # Print the link budget specs
    print()
    print("=====================================")
    print("Link Budget Specs")
    print("=====================================")
    print(f"Gateway Lat/Lon/Alt: {gate['lat']:.2f}\u00b0 / {gate['lon']:.2f}\u00b0 / {gate['alt']:.2f} m")
    print(f"User Terminal Lat/Lon/Alt: {uterm['lat']:.2f}\u00b0 / {uterm['lon']:.2f}\u00b0 / {uterm['alt']:.2f} m")
    print(f"Satellite Lat/Lon/Alt: {sat['lat']:.2f}\u00b0 / {sat['lon']:.2f}\u00b0 / {sat['alt']:.2f} m")

    print(f"Link Reliability: {link_reliability*100:.2f} %")
    print(f"Minimum Link Margin: {link_margin:.2f} dB")
    print("Bit Error Rate: pseudo error free")
    #print(f"Gateway Beam ID: {beam_id_fu}")
    #print(f"User Beam ID: {beam_id_fd}")

    #print(f"Forward Link Transponder Bandwidth: {BW_FORWARD/1e6:.2f} MHz")
    #print(f"Return Link Transponder Bandwidth: {BW_RETURN/1e6:.2f} MHz")

    print(f"Forward Link Allocated Bandwidth: {BW_ALLOCATED_FORWARD/1e6:.2f} MHz")
    print(f"Return Link Allocated Bandwidth: {BW_ALLOCATED_RETURN/1e6:.2f} MHz")

    print(f"Forward UpLink Frequency: {F_UP_FORWARD/1e9:.2f} GHz")
    print(f"Forward DownLink Frequency: {F_DOWN_FORWARD/1e9:.2f} GHz")
    print(f"Return UpLink Frequency: {F_UP_RETURN/1e9:.2f} GHz")
    print(f"Return DownLink Frequency: {F_DOWN_RETURN/1e9:.2f} GHz")
    print()



        
    forward_results = full_link_budget(
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

    return_results = full_link_budget(
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

    print("\nLink Budget Calculation Results")
    print("=====================================")

    print("Forward Link: ")
    print("=====================================")
    for k, v in forward_results.items():
        print(f"{k}: {v:.2f}")

    print("\n")
    
    print("Return Link: ")
    print("=====================================")
    for k, v in return_results.items():
        print(f"{k}: {v:.2f}")

    # given the Eb/N0, find the MODCOD
    forward_CN0 = forward_results['C/N0_dB']
    return_CN0 = return_results['C/N0_dB']
    forward_MOD, forward_COD, forward_SE, forward_bpsym, forward_lm = ms.modcod_select(forward_CN0, link_margin=link_margin)
    return_MOD, return_COD, return_SE, return_bpsym, return_lm = ms.modcod_select(return_CN0, link_margin=link_margin)

    if forward_SE is None or return_SE is None:
        print("No link found for the given coordinates.")
        exit(0)

    forward_bitrate = SR_FORWARD * forward_bpsym
    return_bitrate = SR_RETURN * return_bpsym
    forward_EbN0 = forward_CN0 - 10*np.log10(forward_bitrate)
    return_EbN0 = return_CN0 - 10*np.log10(return_bitrate)

    forward_info_bitrate = BW_ALLOCATED_FORWARD * forward_SE
    return_info_bitrate = BW_ALLOCATED_RETURN * return_SE

    print("\n")
    print("Forward Link MODCOD: ")
    print("=====================================")
    print(f"MOD: {forward_MOD}")
    print(f"COD: {forward_COD}")
    print(f"actual link margin: {forward_lm:.2f} dB")
    print(f"info bitrate: {forward_info_bitrate/1e6:.2f} Mbps")
    print(f"Eb/N0: {forward_EbN0:.2f} dB")
    print("\n")
    print("Return Link MODCOD: ")
    print("=====================================")
    print(f"MOD: {return_MOD}")
    print(f"COD: {return_COD}")
    print(f"actual link margin: {return_lm:.2f} dB")
    print(f"info bitrate: {return_info_bitrate/1e6:.2f} Mbps")
    print(f"Eb/N0: {return_EbN0:.2f} dB")

    print("\n")
