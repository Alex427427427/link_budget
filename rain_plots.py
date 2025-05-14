from L_rain import *
import matplotlib.pyplot as plt
import numpy as np
import rain_rate as rr


# plot the rain rate vs probability of excession at a given latitude
lat_deg = 1.0
lon_deg = 19.0
rr_nom_array, p0_array = rr.rain_nominal_arrays(lat_deg, lon_deg)
rr_t_array = np.linspace(0.0, 100.0, 100)
px_ann_array = np.zeros(len(rr_t_array))
for i, rr_t in enumerate(rr_t_array):
    px_ann_array[i] = rr.px_annual(rr_t, p0_array, rr_nom_array)
plt.plot(rr_t_array, px_ann_array*100, color='k')
plt.xlabel("rain rate threshold [mm/h]")
plt.ylabel("probability of excession [%]")
plt.title(f"probability of exceeding rain rates at ({lat_deg:.2f}$^\\circ$,{lon_deg:.2f}$^\\circ$)")
plt.savefig(f"plots\\probability of exceeding rain rates.png")
plt.show()


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



## Plot the rain height map
Y = lat_grid_rain_height
X = lon_grid_rain_height
Z = h0_grid + 0.36
plt.figure(figsize=(10,5))
plt.pcolormesh(X, Y, Z, cmap="YlGnBu")
plt.xlabel("Longitude [$^\\circ$]")
plt.ylabel("Latitude [$^\\circ$]")
plt.title("Rain Height [km]")
plt.colorbar()
plt.savefig("plots\\rain_height.png")
plt.show()
