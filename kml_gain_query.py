from fastkml import kml
from pathlib import Path
from shapely import Polygon, Point
import matplotlib.pyplot as plt

# kml format:
# <kml>
#   <Document>
#     <name>EIRP</name>
#     <description>Hylas 2 and 4 EIRP gain maps</description>
#     <Placemark>
#       <name>22:56.0</name>                # beam id: 22, gain: 56.0
#       <Polygon>
#         <outerBoundaryIs>
#           <LinearRing>
#             <coordinates>...</coordinates> (LONGITUDE, LATITUDE, ALTITUDE)
#           </LinearRing>
#         </outerBoundaryIs>
#       </Polygon>
#     </Placemark>
#     ...
#   <Document>
# </kml>

EIRP_path = Path(__file__).parent / "gain_maps" / "EIRP.kml"
GT_path = Path(__file__).parent / "gain_maps" / "GT.kml"
# read kml file
k_EIRP = kml.KML.parse(EIRP_path)
k_GT = kml.KML.parse(GT_path)
doc_EIRP = k_EIRP.features[0]
doc_GT = k_GT.features[0]
# initialise lists to store polygons and gains
polygons_EIRP = []
gains_EIRP = []
beam_ids_EIRP = []
polygons_GT = []
gains_GT = []
beam_ids_GT = []
for placemark in doc_EIRP.features:
    gain = float(placemark.name.split(":")[1])
    beam_id = int(placemark.name.split(":")[0])
    coords = placemark.geometry.coords[0]
    coord_list = [(float(coord[0]), float(coord[1])) for coord in coords]
    polygon = Polygon(coord_list)
    polygons_EIRP.append(polygon)
    gains_EIRP.append(gain)
    beam_ids_EIRP.append(beam_id)
for placemark in doc_GT.features:
    gain = float(placemark.name.split(":")[1])
    beam_id = int(placemark.name.split(":")[0])
    coords = placemark.geometry.coords[0]
    coord_list = [(float(coord[0]), float(coord[1])) for coord in coords]
    polygon = Polygon(coord_list)
    polygons_GT.append(polygon)
    gains_GT.append(gain)
    beam_ids_GT.append(beam_id)

# sort the polygons and gains by gain, descending
sorted_indices_EIRP = sorted(range(len(gains_EIRP)), key=lambda i: gains_EIRP[i], reverse=True)
sorted_gains_EIRP = [gains_EIRP[i] for i in sorted_indices_EIRP]
sorted_polygons_EIRP = [polygons_EIRP[i] for i in sorted_indices_EIRP]
sorted_beam_ids_EIRP = [beam_ids_EIRP[i] for i in sorted_indices_EIRP]
sorted_indices_GT = sorted(range(len(gains_GT)), key=lambda i: gains_GT[i], reverse=True)
sorted_gains_GT = [gains_GT[i] for i in sorted_indices_GT]
sorted_polygons_GT = [polygons_GT[i] for i in sorted_indices_GT]
sorted_beam_ids_GT = [beam_ids_GT[i] for i in sorted_indices_GT]

def highest_EIRP_query(lat, lon):
    """
    Given a latitude and longitude, return the highest gain in the EIRP polygons.
    """
    point = Point(lon, lat)
    for i, polygon in enumerate(sorted_polygons_EIRP):
        if polygon.contains(point):
            return sorted_gains_EIRP[i], sorted_beam_ids_EIRP[i]
    return None, None

def highest_GT_query(lat, lon):
    """
    Given a latitude and longitude, return the highest gain in the GT polygons.
    """
    point = Point(lon, lat)
    for i, polygon in enumerate(sorted_polygons_GT):
        if polygon.contains(point):
            return sorted_gains_GT[i], sorted_beam_ids_GT[i]
    return None, None

# plot all EIRP polygons
if __name__ == "__main__":
    import geopandas as gpd
    world = gpd.read_file("gain_maps\\ne_110m_admin_0_countries.zip")
    plot_opacity = 0.1
    world_opacity = 1.0

    lon_min = -30
    lon_max = 60
    lat_min = 20
    lat_max = 65

    ## plots
    fig, axes = plt.subplots(2,1, sharex=True, figsize = (10,10))
    world.plot(ax=axes[0], color="lightgray", edgecolor="black", alpha=world_opacity)
    for i, polygon in enumerate(sorted_polygons_EIRP):
        x, y = polygon.exterior.xy
        axes[0].fill(x, y, alpha=plot_opacity, fc='blue', ec='black')
    axes[0].set_xlabel("Longitude [$^\\circ$]")
    axes[0].set_ylabel("Latitude [$^\\circ$]")
    axes[0].set_title("EIRP Contours")
    axes[0].set_xlim(lon_min, lon_max)
    axes[0].set_ylim(lat_min, lat_max)
    world.plot(ax=axes[1], color="lightgray", edgecolor="black", alpha=world_opacity)
    for i, polygon in enumerate(sorted_polygons_GT):
        x, y = polygon.exterior.xy
        axes[1].fill(x, y, alpha=plot_opacity, fc='blue', ec='black')
    axes[1].set_xlabel("Longitude [$^\\circ$]")
    axes[1].set_ylabel("Latitude [$^\\circ$]")
    axes[1].set_title("G/T Contours")
    axes[1].set_xlim(lon_min, lon_max)
    axes[1].set_ylim(lat_min, lat_max)
    plt.savefig("plots\\eirp_gt.png")
    plt.show()



#lat = 36
#lon = 19
#print(highest_EIRP_query(lat, lon))  # should return the highest gain in the EIRP polygons
#print(highest_GT_query(lat, lon))  # should return the highest gain in the GT polygons
