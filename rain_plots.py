from L_rain import *
import matplotlib.pyplot as plt
import numpy as np

# to plot the rain attenuation coefficients for different frequencies 
# and polarisation states

f_list = np.logspace(start=0, stop=3, num=1000) # frequency in GHz
k_h_list = np.zeros(len(f_list))
k_v_list = np.zeros(len(f_list))
alpha_h_list = np.zeros(len(f_list))
alpha_v_list = np.zeros(len(f_list))
for i, f in enumerate(f_list):
    k_h_list[i] = k_calc(f, 'h')
    k_v_list[i] = k_calc(f, 'v')
    alpha_h_list[i] = alpha_calc(f, 'h')
    alpha_v_list[i] = alpha_calc(f, 'v')

# plot
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("Rain attenuation coefficients")
axes[0,0].plot(f_list, k_h_list, color='k')
axes[0,0].set_xlabel("Frequency (GHz)")
axes[0,0].set_ylabel("$k_H$")
axes[0,0].loglog()
axes[0,0].grid(True, which='both')
axes[1,0].plot(f_list, k_v_list, color='k')
axes[1,0].set_xlabel("Frequency (GHz)")
axes[1,0].set_ylabel("$k_V$")
axes[1,0].loglog()
axes[1,0].grid(True, which='both')
axes[0,1].plot(f_list, alpha_h_list, color='k')
axes[0,1].set_xlabel("Frequency (GHz)")
axes[0,1].set_ylabel("$\\alpha_H$")
axes[0,1].set_xscale("log")
axes[0,1].grid(True, which='both')
axes[1,1].plot(f_list, alpha_v_list, color='k')
axes[1,1].set_xlabel("Frequency (GHz)")
axes[1,1].set_ylabel("$\\alpha_V$")
axes[1,1].set_xscale("log")
axes[1,1].grid(True, which='both')
plt.tight_layout()
plt.savefig("plots\\rain_attenuation_coefficients.png")
plt.show()

# Plot rain total atten over rain rate
rr_list = np.linspace(0, 50, 50)
gamma_list = np.zeros(len(rr_list))
for i, rr in enumerate(rr_list):
    gamma = rain_total_atten(30.0, rr, 35.1*np.pi/180.0, 100.0*np.pi/180.0, np.pi/2, np.pi/4)
    gamma_list[i] = gamma
plt.plot(rr_list, gamma_list, color='k')
plt.xlabel("rain rate [mm/h]")
plt.ylabel("$\\gamma$ [dB/km]")
plt.title("rain specific attenuation over rain rate")
plt.savefig("plots\\rain_atten_over_rr.png")
plt.show()




## Plot the rain height map
Y = Lat_grid
X = Lon_grid
Z = h0_grid
plt.figure(figsize=(10,5))
plt.pcolormesh(X, Y, Z, cmap="YlGnBu")
plt.xlabel("Longitude [$^\\circ$]")
plt.ylabel("Latitude [$^\\circ$]")
plt.title("Rain Height [km]")
plt.colorbar()
plt.savefig("plots\\rain_height.png")
plt.show()
