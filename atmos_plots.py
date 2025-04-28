from L_atmos import *
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



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
plt.savefig("plots\\o2_specific_absorption_at_altitudes.png")
plt.show()



# plot gamma
h_list = np.linspace(0, 85, 100)
plt.plot(h_list, total_attenuation(30, 0.5, initial_h=0, final_h=85, num_h=100), color='k', label="Total attenuation")
plt.xlabel("Altitude (km)")
plt.ylabel("Total specific attenuation (dB/km)")
plt.title("Total specific attenuation at 30 GHz")
plt.grid()
plt.savefig("plots\\total_attenuation_over_distance.png")
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
plt.savefig("plots\\standard_atmosphere.png")
plt.show()

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
plt.savefig("plots\\total_specific_absorption_at_altitudes.png")
plt.show()
