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

lon_min = -30
lon_max = 60
lat_min = 20
lat_max = 65



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
cmap = plt.cm.rainbow.copy()
cmap.set_bad(color='black')


cbar_max = np.ceil(max(np.nanmax(CN0_f_masked_filled), np.nanmax(CN0_r_masked_filled)))
cbar_min = np.floor(min(np.nanmin(CN0_f_masked_filled), np.nanmin(CN0_r_masked_filled)))
cn0_levels = np.arange(cbar_min, cbar_max + 1, 1)

## CN0 plots
fig, axes = plt.subplots(2,1, sharex=True, figsize = (10,10))
world.plot(ax=axes[0], color="lightgray", edgecolor="black", alpha=world_opacity)
pcf = axes[0].contourf(lon_grid, lat_grid, CN0_f_masked_filled, levels=cn0_levels, cmap = cmap, alpha=plot_opacity)
axes[0].set_xlabel("Longitude [$^\\circ$]")
axes[0].set_ylabel("Latitude [$^\\circ$]")
axes[0].set_title("Forward Link $C/N_0$ [dB Hz]")
axes[0].set_xlim(lon_min, lon_max)
axes[0].set_ylim(lat_min, lat_max)
world.plot(ax=axes[1], color="lightgray", edgecolor="black", alpha=world_opacity)
pcr = axes[1].contourf(lon_grid, lat_grid, CN0_r_masked_filled, levels=cn0_levels, cmap = cmap, alpha=plot_opacity) 
axes[1].set_xlabel("Longitude [$^\\circ$]")
axes[1].set_ylabel("Latitude [$^\\circ$]")
axes[1].set_title("Return Link $C/N_0$ [dB Hz]")
axes[1].set_xlim(lon_min, lon_max)
axes[1].set_ylim(lat_min, lat_max)
fig.colorbar(pcr, ax=axes)
plt.savefig("outputs\\CN0_fr.png")
plt.show()






## MODCOD, birate, EbN0 plots --------------------------------------------------
symbol_rate_f = 52e6 # Hz
symbol_rate_r = 10e6 # Hz
roll_off = 0.1
bw_f = symbol_rate_f * (1 + roll_off) # Hz
bw_r = symbol_rate_r * (1 + roll_off) # Hz

modcod_fmap = CN0_fmap.copy()
modcod_rmap = CN0_rmap.copy()
# bitrate maps
br_fmap = CN0_fmap.copy()
br_rmap = CN0_rmap.copy()
# EbN0 maps
ebn0_fmap = CN0_fmap.copy()
ebn0_rmap = CN0_rmap.copy()
# modcod maps
modcod_list = (ms.MOD_table + " " + ms.COD_table).tolist()
# create label maps and colour maps
label_map = {}
for i, modcod in enumerate(modcod_list):
    label_map[i] = str(modcod)

levels = np.arange(len(modcod_list)+1)
#colors = distinctipy.get_colors(len(modcod_list)+1, rng=1283971837)
colors = []
for i in range(len(modcod_list)+1):
    colors.append(plt.cm.nipy_spectral(i/len(modcod_list)))

# calculate maps
link_margin = 1.0 # dB
for r in range(len(CN0_fmap)):
    for c in range(len(CN0_fmap[0])):
        CN0_f = CN0_fmap[r,c]
        CN0_r = CN0_rmap[r,c]
        if CN0_f == -np.inf or CN0_r == -np.inf:
            # no MODCOD, no link
            modcod_fmap[r,c] = np.nan
            modcod_rmap[r,c] = np.nan
            br_fmap[r,c] = np.nan
            br_rmap[r,c] = np.nan
            ebn0_fmap[r,c] = np.nan
            ebn0_rmap[r,c] = np.nan
            continue
        mod_f, cod_f, se_f, bpsym_f, lm_f = ms.modcod_select(CN0_f, link_margin)
        mod_r, cod_r, se_r, bpsym_r, lm_r = ms.modcod_select(CN0_r, link_margin)
        if mod_f == "None" or cod_f == "None" or mod_r == "None" or cod_r == "None":
            # no MODCOD, no link
            modcod_fmap[r,c] = np.nan
            modcod_rmap[r,c] = np.nan
            br_fmap[r,c] = np.nan
            br_rmap[r,c] = np.nan
            ebn0_fmap[r,c] = np.nan
            ebn0_rmap[r,c] = np.nan
            continue
        modcod_f = mod_f + " " + cod_f
        modcod_r = mod_r + " " + cod_r
        # info bitrate
        br_f = bpsym_f * symbol_rate_f * float(cod_f.split("/")[0]) / float(cod_f.split("/")[1])
        br_r = bpsym_r * symbol_rate_r * float(cod_r.split("/")[0]) / float(cod_r.split("/")[1])
        # EbN0
        ebn0_f = CN0_f - 10*np.log10(bpsym_f*symbol_rate_f)
        ebn0_r = CN0_r - 10*np.log10(bpsym_r*symbol_rate_r)
        # save to maps
        #print(f"MODCOD: {modcod_f} and {modcod_r}")
        modcod_fmap[r,c] = modcod_list.index(modcod_f) + 0.5
        modcod_rmap[r,c] = modcod_list.index(modcod_r) + 0.5
        br_fmap[r,c] = br_f
        br_rmap[r,c] = br_r
        ebn0_fmap[r,c] = ebn0_f
        ebn0_rmap[r,c] = ebn0_r

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
axes[0].set_xlim(lon_min, lon_max)
axes[0].set_ylim(lat_min, lat_max)
world.plot(ax=axes[1], color="lightgray", edgecolor="black", alpha=world_opacity)
pcr = axes[1].contourf(lon_grid, lat_grid, modcod_rmap, levels, colors=colors, alpha=plot_opacity)
axes[1].set_xlabel("Longitude [$^\\circ$]")
axes[1].set_ylabel("Latitude [$^\\circ$]")
axes[1].set_title("Return Link MODCOD")
axes[1].set_xlim(lon_min, lon_max)
axes[1].set_ylim(lat_min, lat_max)


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



# bitrate plots
br_cbar_max = np.ceil(max(np.nanmax(br_fmap/1e6), np.nanmax(br_rmap/1e6)))
br_cbar_min = np.floor(min(np.nanmin(br_fmap/1e6), np.nanmin(br_rmap/1e6)))
br_cbar_levels = np.arange(br_cbar_min, br_cbar_max + 1, 1)
br_cmap = plt.cm.rainbow.copy()
br_cmap.set_bad(color='black')

fig, axes = plt.subplots(2,1, sharex=True, figsize = (10,10))
world.plot(ax=axes[0], color="lightgray", edgecolor="black", alpha=world_opacity)
pcf = axes[0].contourf(lon_grid, lat_grid, br_fmap/1e6, levels=br_cbar_levels, cmap=br_cmap, alpha=plot_opacity)
axes[0].set_xlabel("Longitude [$^\\circ$]")
axes[0].set_ylabel("Latitude [$^\\circ$]")
axes[0].set_title("Forward Link Bitrate [Mbps]")
axes[0].set_xlim(lon_min, lon_max)
axes[0].set_ylim(lat_min, lat_max)
world.plot(ax=axes[1], color="lightgray", edgecolor="black", alpha=world_opacity)
pcr = axes[1].contourf(lon_grid, lat_grid, br_rmap/1e6, levels=br_cbar_levels, cmap=br_cmap, alpha=plot_opacity)
axes[1].set_xlabel("Longitude [$^\\circ$]")
axes[1].set_ylabel("Latitude [$^\\circ$]")
axes[1].set_title("Return Link Bitrate [Mbps]")
axes[1].set_xlim(lon_min, lon_max)
axes[1].set_ylim(lat_min, lat_max)
fig.colorbar(pcr, ax=axes)
plt.savefig("outputs\\BR_fr.png")
plt.show()


# EbN0 plots
EbN0_cbar_max = np.ceil(max(np.nanmax(ebn0_fmap), np.nanmax(ebn0_rmap)))
EbN0_cbar_min = np.floor(min(np.nanmin(ebn0_fmap), np.nanmin(ebn0_rmap)))
EbN0_cbar_levels = np.arange(EbN0_cbar_min, EbN0_cbar_max + 1, 1)
EbN0_cmap = plt.cm.rainbow.copy()
EbN0_cmap.set_bad(color='black')

fig, axes = plt.subplots(2,1, sharex=True, figsize = (10,10))
world.plot(ax=axes[0], color="lightgray", edgecolor="black", alpha=world_opacity)
pcf = axes[0].contourf(lon_grid, lat_grid, ebn0_fmap, levels=EbN0_cbar_levels, cmap=EbN0_cmap, alpha=plot_opacity)
axes[0].set_xlabel("Longitude [$^\\circ$]")
axes[0].set_ylabel("Latitude [$^\\circ$]")
axes[0].set_title("Forward Link $E_b/N_0$ [dB]")
axes[0].set_xlim(lon_min, lon_max)
axes[0].set_ylim(lat_min, lat_max)
world.plot(ax=axes[1], color="lightgray", edgecolor="black", alpha=world_opacity)
pcr = axes[1].contourf(lon_grid, lat_grid, ebn0_rmap, levels=EbN0_cbar_levels, cmap=EbN0_cmap, alpha=plot_opacity)
axes[1].set_xlabel("Longitude [$^\\circ$]")
axes[1].set_ylabel("Latitude [$^\\circ$]")
axes[1].set_title("Return Link $E_b/N_0$ [dB]")
axes[1].set_xlim(lon_min, lon_max)
axes[1].set_ylim(lat_min, lat_max)
fig.colorbar(pcr, ax=axes)
plt.savefig("outputs\\EbN0_fr.png")
plt.show()
