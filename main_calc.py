import numpy as np

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
        f_up, f_down, BW, bit_rate,
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

    L_atm_up = 0 # dB, atmospheric loss in uplink
    L_atm_down = 0 # dB, atmospheric loss in downlink

    noise = noise_power_per_T(BW) # dBW/K

    L_up_total = L_up_path + L_atm_up # dB
    L_down_total = L_down_path + L_atm_down # dB

    uplink_CNR = single_link_CNR(EIRP_Earth_start, L_up_total, G_per_T_sat, noise) # dB
    downlink_CNR = single_link_CNR(EIRP_sat, L_down_total, G_per_T_Earth_end, noise) # dB

    # final results
    total_CNR_dB = total_CNR([uplink_CNR, downlink_CNR, C_IM]) # dB
    CNR_total_value = power_from_dB(total_CNR_dB) # linear value
    CN0_total_value = CNR_total_value * BW # Hz
    CN0_dB = dB_from_power(CN0_total_value) # dB-Hz
    EbN0_total_value = CN0_total_value / bit_rate # dimensionless ratio
    EbN0_dB = dB_from_power(EbN0_total_value) # dB

    return {
        'FSPL_up_dB': L_up_path,
        'FSPL_down_dB': L_down_path,
        'L_atm_up_dB': L_atm_up,
        'L_atm_down_dB': L_atm_down,
        'bit_rate_Mbps': bit_rate*10**(-6), # bps
        'BW_MHz': BW*10**(-6), # Hz
        'CNR_dB': total_CNR_dB,
        'Eb/N0_dB': EbN0_dB,
    }

# main code
if __name__ == "__main__":

    # MODCOD table
    EbN0_table = np.array([
        -5.36,-4.25,-3.31,-2.01,-0.78,0.09,1.02,1.67,2.17,0.729,3.19,3.41,1.849,3.139,2.949,4.579,4.189,5.919,6.209,5.009,
        5.589,5.740,6.869,7.109,6.65,7.29,8.7,9.06
    ])
    MOD_table = np.array([
        "QPSK","QPSK","QPSK","QPSK","QPSK","QPSK","QPSK","QPSK","QPSK","8PSK","QPSK","QPSK","8PSK","8PSK","16APSK","8PSK",
        "16APSK","8PSK","8PSK","16APSK","16APSK","32APSK","16APSK",
        "16APSK","32APSK","32APSK","32APSK","32APSK"
    ])
    COD_table = np.array([
        "1/4","1/3","2/5","1/2",
        "3/5","2/3","3/4","4/5","5/6","3/5","8/9","9/10","2/3","3/4","2/3","5/6","3/4","8/9",
        "9/10","4/5","5/6","3/4","8/9","9/10","4/5","5/6","8/9","9/10"
    ])
    sorted_indices = np.argsort(EbN0_table)
    EbN0_table = EbN0_table[sorted_indices]
    MOD_table = MOD_table[sorted_indices]
    COD_table = COD_table[sorted_indices]

    # Communication specs
    BW = 200e6 # bandwidth in Hz
    BIT_RATE = 1e9 # bit rate in bps
    F_UP_FORWARD = 27e9 # uplink frequency in Hz
    F_DOWN_FORWARD = F_UP_FORWARD - 8e9 # downlink frequency in Hz
    F_UP_RETURN = 25e9 # uplink frequency in Hz
    F_DOWN_RETURN = F_UP_RETURN - 10e9 # downlink frequency in Hz

    # gateway specs
    GATE = {
        'lat': 15, # degrees
        'lon': 15, # degrees
        'alt': 100.0, # meters
        'PSD': 50e-6, # dBW/Hz
        'EIRP': 50e-6 + 10*np.log10(BW), # dBW
        'G/T': 30, # dB/K
        'C/IM': 30.0 # dB
    }

    # satellite specs
    SAT = {
        'lat': 0, # degrees
        'lon': 31.0, # degrees
        'alt': R_GEO - R_EQUATOR, # meters
        'EIRP': 50, # dBW
        'G/T': 30 # dB/K
    }

    # user terminal specs
    USER = {
        'lat': 20, # degrees
        'lon': 40, # degrees
        'alt': 6000, # meters
        'EIRP': 50, # dBW
        'G/T': 30.0 # dB/K
    }

    forward_results = full_link_budget(
        EIRP_Earth_start=GATE['EIRP'], # dBW
        Earth_start_lat_lon_alt=(np.radians(GATE['lat']), np.radians(GATE['lon']), GATE['alt']), # lat, lon, alt in radians and meters
        sat_lat_lon_alt=(np.radians(SAT['lat']), np.radians(SAT['lon']), SAT['alt']), # lat, lon, alt in radians and meters
        G_per_T_sat=SAT['G/T'], # dB/K
        EIRP_sat=SAT['EIRP'], # dBW
        Earth_end_lat_lon_alt=(np.radians(USER['lat']), np.radians(USER['lon']), USER['alt']), # lat, lon, alt in radians and meters
        G_per_T_Earth_end=USER['G/T'], # dB/K
        f_up=F_UP_FORWARD, # frequency in Hz
        f_down=F_DOWN_FORWARD, # frequency in Hz
        BW=BW, # bandwidth in Hz
        bit_rate=BIT_RATE, # bit rate in bps
        C_IM=GATE['C/IM'] # dB
    )

    return_results = full_link_budget(
        EIRP_Earth_start=USER['EIRP'], # dBW
        Earth_start_lat_lon_alt=(np.radians(USER['lat']), np.radians(USER['lon']), USER['alt']), # lat, lon, alt in radians and meters
        sat_lat_lon_alt=(np.radians(SAT['lat']), np.radians(SAT['lon']), SAT['alt']), # lat, lon, alt in radians and meters
        G_per_T_sat=SAT['G/T'], # dB/K
        EIRP_sat=SAT['EIRP'], # dBW
        Earth_end_lat_lon_alt=(np.radians(GATE['lat']), np.radians(GATE['lon']), GATE['alt']), # lat, lon, alt in radians and meters
        G_per_T_Earth_end=GATE['G/T'], # dB/K
        f_up=F_UP_RETURN, # frequency in Hz
        f_down=F_DOWN_RETURN, # frequency in Hz
        BW=BW, # bandwidth in Hz
        bit_rate=BIT_RATE, # bit rate in bps
        C_IM=GATE['C/IM'] # dB
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
    forward_EbN0 = forward_results['Eb/N0_dB']
    return_EbN0 = return_results['Eb/N0_dB']
    if forward_EbN0 < EbN0_table[0]:
        forward_MOD = "None"
        forward_COD = "None"
    elif forward_EbN0 > EbN0_table[-1]:
        forward_MOD = MOD_table[-1]
        forward_COD = COD_table[-1]
    else:
        forward_MOD = MOD_table[np.where(EbN0_table >= forward_EbN0)[0][0]-1]
        forward_COD = COD_table[np.where(EbN0_table >= forward_EbN0)[0][0]-1]
    if return_EbN0 < EbN0_table[0]:
        return_MOD = "None"
        return_COD = "None"
    elif return_EbN0 > EbN0_table[-1]:
        return_MOD = MOD_table[-1]
        return_COD = COD_table[-1]
    else:
        return_MOD = MOD_table[np.where(EbN0_table >= return_EbN0)[0][0]-1]
        return_COD = COD_table[np.where(EbN0_table >= return_EbN0)[0][0]-1]

    print("\n")
    print("Forward Link MODCOD: ")
    print("=====================================")
    print(f"MOD: {forward_MOD}")
    print(f"COD: {forward_COD}")
    print("\n")
    print("Return Link MODCOD: ")
    print("=====================================")
    print(f"MOD: {return_MOD}")
    print(f"COD: {return_COD}")
    print("\n")
