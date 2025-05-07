import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import modcod_select as ms
import geopandas as gpd
import distinctipy

# read in the world map
world = gpd.read_file("gain_maps\\ne_110m_admin_0_countries.zip")
plot_opacity = 0.8
world_opacity = 1.0





## plotting CN0 -------------------------------------------------------------------
lat_grid = pd.read_csv("outputs\\LAT.csv", header=None).to_numpy()
lon_grid = pd.read_csv("outputs\\LON.csv", header=None).to_numpy()
# change longitude (2D numpy array) to -180 to 180
lon_grid[lon_grid > 180] -= 360

CN0_fmap = pd.read_csv("outputs\\CN0_fmap.csv", header=None).to_numpy()
CN0_rmap = pd.read_csv("outputs\\CN0_rmap.csv", header=None).to_numpy()

CN0_f_masked = ma.masked_where(CN0_fmap == -np.inf, CN0_fmap)
CN0_r_masked = ma.masked_where(CN0_rmap == -np.inf, CN0_rmap)
CN0_f_masked_filled = CN0_f_masked.filled(np.nan)
CN0_r_masked_filled = CN0_r_masked.filled(np.nan)
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='black')

## CN0 plots
fig, axes = plt.subplots(2,1, sharex=True, figsize = (10,10))
world.plot(ax=axes[0], color="lightgray", edgecolor="black", alpha=world_opacity)
pcf = axes[0].contourf(lon_grid, lat_grid, CN0_f_masked_filled, cmap = cmap, alpha=plot_opacity)
axes[0].set_xlabel("Longitude [$^\\circ$]")
axes[0].set_ylabel("Latitude [$^\\circ$]")
axes[0].set_title("Forward Link $C/N_0$ [dB Hz]")
axes[0].set_xlim(-30, 60)
axes[0].set_ylim(20, 65)
world.plot(ax=axes[1], color="lightgray", edgecolor="black", alpha=world_opacity)
pcr = axes[1].contourf(lon_grid, lat_grid, CN0_r_masked_filled, cmap = cmap, alpha=plot_opacity) 
axes[1].set_xlabel("Longitude [$^\\circ$]")
axes[1].set_ylabel("Latitude [$^\\circ$]")
axes[1].set_title("Return Link $C/N_0$ [dB Hz]")
axes[1].set_xlim(-30, 60)
axes[1].set_ylim(20, 65)
fig.colorbar(pcf, ax=axes)
plt.savefig("outputs\\CN0_fr.png")
plt.show()

## MODCOD plots --------------------------------------------------
modcod_fmap = CN0_fmap.copy()
modcod_rmap = CN0_rmap.copy()
# modcod maps
modcod_list = (ms.MOD_table + " " + ms.COD_table).tolist()
# create label maps and colour maps
label_map = {}
for i, modcod in enumerate(modcod_list):
    label_map[i] = str(modcod)

levels = np.arange(len(modcod_list)+1)
colors = distinctipy.get_colors(len(modcod_list)+1)

# calculate MODCOD map
link_margin = 1.0 # dB
for r in range(len(CN0_fmap)):
    for c in range(len(CN0_fmap[0])):
        CN0_f = CN0_fmap[r,c]
        CN0_r = CN0_rmap[r,c]
        if CN0_f == -np.inf or CN0_r == -np.inf:
            # no MODCOD, no link
            modcod_fmap[r,c] = np.nan
            modcod_rmap[r,c] = np.nan
            continue
        mod_f, cod_f, SE_f = ms.modcod_select(CN0_f, link_margin)
        mod_r, cod_r, SE_r = ms.modcod_select(CN0_r, link_margin)
        if mod_f == "None" or cod_f == "None" or mod_r == "None" or cod_r == "None":
            # no MODCOD, no link
            modcod_fmap[r,c] = np.nan
            modcod_rmap[r,c] = np.nan
            continue
        modcod_f = mod_f + " " + cod_f
        modcod_r = mod_r + " " + cod_r
        #print(f"MODCOD: {modcod_f} and {modcod_r}")
        modcod_fmap[r,c] = modcod_list.index(modcod_f) + 0.5
        modcod_rmap[r,c] = modcod_list.index(modcod_r) + 0.5

# save to csv
np.savetxt('outputs\\MODCOD_fmap.csv', modcod_fmap, fmt='%.1f', delimiter=',')
np.savetxt('outputs\\MODCOD_rmap.csv', modcod_rmap, fmt='%.1f', delimiter=',')

## modcod plots

fig, axes = plt.subplots(2,1, sharex=True, figsize = (10,10))
world.plot(ax=axes[0], color="lightgray", edgecolor="black", alpha=world_opacity)
pcf = axes[0].contourf(lon_grid, lat_grid, modcod_fmap, levels, colors=colors, alpha=plot_opacity)
axes[0].set_xlabel("Longitude [$^\\circ$]")
axes[0].set_ylabel("Latitude [$^\\circ$]")
axes[0].set_title("Forward Link MODCOD")
axes[0].set_xlim(-30, 60)
axes[0].set_ylim(20, 65)
world.plot(ax=axes[1], color="lightgray", edgecolor="black", alpha=world_opacity)
pcr = axes[1].contourf(lon_grid, lat_grid, modcod_rmap, levels, colors=colors, alpha=plot_opacity)
axes[1].set_xlabel("Longitude [$^\\circ$]")
axes[1].set_ylabel("Latitude [$^\\circ$]")
axes[1].set_title("Return Link MODCOD")
axes[1].set_xlim(-30, 60)
axes[1].set_ylim(20, 65)

# formatting contour labels
'''
fmt = {}
for l, s in zip(levels, modcod_list):
    fmt[l] = s

axes[0].clabel(pcf, inline=True, fontsize=8, fmt=fmt, colors='black', use_clabeltext=True)
axes[1].clabel(pcr, inline=True, fontsize=8, fmt=fmt, colors='black', use_clabeltext=True)
'''
# colorbar
cbar = fig.colorbar(pcf, ax=axes, ticks=range(len(modcod_list)))
cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(modcod_list):
    cbar.ax.text(1.1, j + 0.5 , lab, va='center')
#cbar.ax.get_yaxis().labelpad = 15
#cbar.ax.set_ylabel('# of contacts', rotation=270)

plt.savefig("outputs\\MODCOD_fr.png")
plt.show()
