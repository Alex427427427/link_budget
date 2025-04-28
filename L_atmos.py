import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
import scipy.integrate

# oxygen and water spectroscopic data
oxydf = pd.read_csv(Path(__file__).parent / "atmos_data" / "oxyspec.csv")
waterdf = pd.read_csv(Path(__file__).parent / "atmos_data" / "waterspec.csv")
f0_o2, a1_o2, a2_o2, a3_o2, a4_o2, a5_o2, a6_o2 = oxydf["f0"].astype(float).values, oxydf["a1"].astype(float).values, \
                        oxydf["a2"].astype(float).values, oxydf["a3"].astype(float).values, oxydf["a4"].astype(float).values, \
                        oxydf["a5"].astype(float).values, oxydf["a6"].astype(float).values
f0_h2o, b1_h2o, b2_h2o, b3_h2o, b4_h2o, b5_h2o, b6_h2o = waterdf["f0"].astype(float).values, \
                        waterdf["b1"].astype(float).values, waterdf["b2"].astype(float).values, waterdf["b3"].astype(float).values, \
                        waterdf["b4"].astype(float).values, waterdf["b5"].astype(float).values, waterdf["b6"].astype(float).values

# 1976 standard atmosphere ITU-R 835
grads = [-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0] # temperature gradients in K/km
alts = [0, 11, 20, 32, 47, 51, 71, 85] # altitudes in km
Ts = [288.15] # temperature in K, 15 C
Ps = [1013.25] # pressure in hPa
# populate the temperature and pressure at those altitudes
for i in range(len(grads)):
    Ts.append(Ts[i] + grads[i] * (alts[i+1] - alts[i]))
    if grads[i] == 0.0:
        Ps.append(Ps[i] * np.exp(-34.163*(alts[i+1] - alts[i]) / Ts[i]))
    else:
        Ps.append(Ps[i] * (Ts[i] / Ts[i+1])**(34.163 / grads[i]))

def standard_T(h):
    '''
    Calculate the temperature at height h in K.
    h: height in km
    '''
    if h < alts[0]:
        raise ValueError("Height must be positive")
    elif h < alts[-1]:
        for i in range(len(grads)):
            if h < alts[i+1]:
                return Ts[i] + grads[i] * (h - alts[i])
    else:
        temp = Ts[-1] + grads[-1] * (h - alts[-1])
        if temp < 0:
            return 0.0
        return temp
    
def standard_P(h):
    '''
    Calculate the pressure at height h in hPa.
    h: height in km
    '''
    if h < alts[0]:
        raise ValueError("Height must be positive")
    elif h < alts[-1]:
        for i in range(len(grads)):
            if h < alts[i+1]:
                if grads[i] == 0.0:
                    return Ps[i] * np.exp(-34.163*(h - alts[i]) / Ts[i])
                else:
                    return Ps[i] * (Ts[i] / standard_T(h))**(34.163 / grads[i])
    else:
        return 0.0
    
def standard_rho(h):
    return 7.5 * np.exp(- h / 6.0) # water vapour density in g/m^3

def shape_factor(f, f0, f_Delta, delta):
    '''
    Calculate the shape factor for a given frequency, line width, and interference correction.
    f: frequency in GHz
    f0: line center frequency in GHz
    f_Delta: line width in GHz
    delta: interference correction in GHz
    '''
    return f / f0 * (
        (f_Delta - delta*(f0 - f))
        /
        ((f0 - f)**2 + f_Delta**2) 
        + 
        (f_Delta - delta*(f0 + f))
        /
        ((f0+f)**2 + f_Delta**2)
    ) # shape factor

def line_width_o2(a3, p, th, a4, e):
    '''
    Calculate the line width for oxygen.
    a3: line width coefficient
    p: pressure in hPa
    th: temperature in K
    a4: line width exponent
    e: water vapour pressure in hPa
    '''
    f_Delta = a3 * 10**(-4) * (p * th**(0.8-a4) + 1.1 * e * th) # line width
    f_Delta = np.sqrt(f_Delta**2 + 2.25e-6) # zeeman splitting of oxygen lines
    return f_Delta

def interference_o2(a5, a6, th, p, e):
    '''
    Calculate the interference correction for oxygen.
    a5: interference correction coefficient
    a6: interference correction exponent
    th: temperature in K
    p: pressure in hPa
    e: water vapour pressure in hPa
    '''
    inter_d = (a5 + a6 * th) * 10**(-4) * (p + e)*th**0.8 # interference correction
    return inter_d


def specific_absorption(f, T=288.15, p=1013.25, rho=7.5):
    '''
    Calculate the specific absorption of oxygen and water vapour in dB/km.
    f: frequency in GHz
    T: temperature in K (default 288.15 K)
    p: pressure in hPa (default 1013.25 hPa)
    rho: water vapour density in g/m^3 (default 7.5 g/m^3)
    '''
    th = 300 / T
    e = rho * T / 216.7 # water vapour pressure in hPa
    # Oxygen absorption
    # dry air continuum arising from non-resonant Debye spectrum of oxygen below 10 GHz and a pressure induced nitrogen attenuation above 100 GHz. 
    d = 5.6e-4*(p+e)*th**0.8 # dry air continuum
    N_dry = f * p * th**2 * (
        6.14e-5 / (d*(1 + (f/d)**2)) + 1.4e-12*p*th**1.5 / (1+1.9e-5 * f**1.5)
    )
    N_o2 = N_dry
    for f0, a1, a2, a3, a4, a5, a6 in zip(f0_o2, a1_o2, a2_o2, a3_o2, a4_o2, a5_o2, a6_o2):
        S_i = a1*10**(-7) * p * th**3 * np.exp(a2*(1-th)) # strength of line i
        delta_f = line_width_o2(a3, p, th, a4, e) # line width
        inter_d = interference_o2(a5, a6, th, p, e) # interference correction
        F_i = shape_factor(f, f0, delta_f, inter_d) # shape factor
        N_o2 += S_i * F_i
        #N_o2_list.append(N_o2)
    # water vapour absorption
    N_h2o = 0
    for f0, b1, b2, b3, b4, b5, b6 in zip(f0_h2o, b1_h2o, b2_h2o, b3_h2o, b4_h2o, b5_h2o, b6_h2o):
        S_i = b1*10**(-1) * e * th**3.5 * np.exp(b2*(1-th))
        delta_f = b3 * 10**(-4) * (p * th**b4 + b5 * e * th**b6) # line width
        delta_f = 0.535*delta_f + np.sqrt(0.217*delta_f**2 + 2.1316e-12*f0**2 / th) # zeeman splitting of oxygen lines
        F_i = shape_factor(f, f0, delta_f, 0.0) # shape factor
        N_h2o += S_i * F_i
        #N_h2o_list.append(N_h2o)
    gamma = 0.182 * f * (N_o2 + N_h2o) # specific absorption 
    return gamma # in dB/km
'''
# plot shape factor over frequencies
f_list = np.linspace(50, 70, 1000) # frequency in GHz
index = 10
f0 = f0_o2[index] # line center frequency in GHz
f_Delta = line_width_o2(a3_o2[index], 1013.25, 300/288.15, a4_o2[index], 7.5*288.15/216.7) # line width in GHz
print(f"Line width: {f_Delta} GHz")
#f_Delta = 0.5
delta = interference_o2(a5_o2[index], a6_o2[index], 300/288.15, 1013.25, 7.5*288.15/216.7) # interference correction in GHz
#delta = 0.0
F_list = []
for f in f_list:
    F = shape_factor(f, f0, f_Delta, delta)
    F_list.append(F)
F_list = np.array(F_list)
plt.plot(f_list, F_list, color='k', label="Shape factor")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Shape factor")
plt.title("Shape factor for oxygen and water vapour")
plt.grid()
plt.show()
total_area = scipy.integrate.simps(F_list, f_list) # total area under the curve
print(f"Total area under the curve: {total_area}")
'''
'''
start_freq = 50 # GHz
end_freq = 70 # GHz
freq_list = np.linspace(start_freq, end_freq, 1000) # frequency in GHz
altitude_list = np.array([0,5,10,15,20])
gamma_list = np.zeros((len(altitude_list), len(freq_list))) # specific absorption in dB/km
for i, h in enumerate(altitude_list):
    for j, f in enumerate(freq_list):
        T = standard_T(h)
        P = standard_P(h)
        rho = standard_rho(h)
        gamma_list[i, j] = specific_absorption(f, T, P, rho)

# plot
plt.figure(figsize=(10, 5))
for i, h in enumerate(altitude_list):
    plt.plot(freq_list, gamma_list[i], label=f"{h} km")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Specific absorption (dB/km)")
plt.title("Specific absorption of oxygen")
# set y axis to log scale
plt.yscale("log")
plt.xlim(start_freq, end_freq)
plt.legend()
plt.grid()
plt.savefig("o2_specific_absorption_at_altitudes.png")
plt.show()
'''
def total_attenuation(f, elevation_angle, initial_h=0, final_h=85, num_h=1000):
    '''
    Calculate the total attenuation in dB/km.
    f: frequency in GHz
    elevation_angle: elevation angle in radians 
    '''
    if elevation_angle < 0:
        raise ValueError("Elevation angle must be positive")
    elif elevation_angle > 90:
        raise ValueError("Elevation angle must be less than 90")
    
    # integrate the absorption over the path length
    h_list = np.linspace(initial_h, final_h, num_h) # height in km
    gamma_list = []
    for h in h_list:
        T = standard_T(h)
        P = standard_P(h)
        rho = standard_rho(h)
        gamma = specific_absorption(f, T, P, rho) # specific absorption in dB/km
        gamma_list.append(gamma / np.sin(elevation_angle))
        #print(np.sin(elevation_angle), gamma)
    gamma_list = np.array(gamma_list)

    return gamma_list

def standard_T_list(h_list):
    '''
    Calculate the temperature at height h in K.
    h: height in km
    '''
    return np.array([standard_T(h) for h in h_list])

def standard_P_list(h_list):
    '''
    Calculate the pressure at height h in hPa.
    h: height in km
    '''
    return np.array([standard_P(h) for h in h_list])

def standard_rho_list(h_list):
    '''
    Calculate the water vapour density at height h in g/m^3.
    h: height in km
    '''
    return np.array([standard_rho(h) for h in h_list])

'''
# plot gamma
h_list = np.linspace(0, 85, 100)
plt.plot(h_list, total_attenuation(30, 0.5, initial_h=0, final_h=85, num_h=100), color='k', label="Total attenuation")
plt.xlabel("Altitude (km)")
plt.ylabel("Total specific attenuation (dB/km)")
plt.title("Total specific attenuation at 30 GHz")
plt.grid()
plt.savefig("total_attenuation.png")
plt.show()

# plot temperature, pressure, and water vapour density for 0-85 km
h_list = np.linspace(0, 85, 100)
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle("Standard atmosphere")
axes[0].plot(h_list, standard_T_list(h_list), color='k', label="Temperature")
axes[0].set_ylabel("Temperature (K)")
#axes[0].set_xlabel("Altitude (km)")
axes[0].grid()
axes[1].plot(h_list, standard_P_list(h_list), color='k', label="Pressure")
axes[1].set_ylabel("Pressure (hPa)")
#axes[1].set_xlabel("Altitude (km)")
axes[1].grid()
axes[2].plot(h_list, standard_rho_list(h_list), color='k', label="Water vapour density")
axes[2].set_ylabel("Water vapour density (g/m^3)")
axes[2].set_xlabel("Altitude (km)")
axes[2].grid()
plt.tight_layout()
plt.savefig("standard_atmosphere.png")
plt.show()
'''
# plot specific absorption at standard atmosphere for 0-1000 GHz
f_list = np.linspace(0, 1000, 1000) # frequency in GHz
alts = np.array([0, 5, 10, 15, 20])
gamma_list = np.zeros((len(alts), len(f_list))) # specific absorption in dB/km
for i, h in enumerate(alts):
    T = standard_T(h)
    P = standard_P(h)
    rho = standard_rho(h)
    for j, f in enumerate(f_list):
        gamma_list[i, j] = specific_absorption(f, T, P, rho) # specific absorption in dB/km
# plot
plt.figure(figsize=(10, 5))
for i, h in enumerate(alts):
    plt.plot(f_list, gamma_list[i], label=f"{h} km")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Specific absorption (dB/km)")
plt.title("Specific absorption of oxygen and water vapour")
# set y axis to log scale
plt.yscale("log")
plt.xlim(0, 1000)
plt.legend()
plt.grid()
plt.savefig("total_specific_absorption_at_altitudes.png")
plt.show()