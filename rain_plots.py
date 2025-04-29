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
