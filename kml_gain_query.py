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
#       <name>56.0</name>
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

EIRP_path = Path(__file__).parent / "gain_maps" / "Hy2_Hy4_ViasatBeams_EIRP_polygon.kml"
GT_path = Path(__file__).parent / "gain_maps" / "Hy2_Hy4_ViasatBeams_GT_polygon.kml"
# read kml file
k_EIRP = kml.KML.parse(EIRP_path)
k_GT = kml.KML.parse(GT_path)
doc_EIRP = k_EIRP.features[0]
doc_GT = k_GT.features[0]
# initialise lists to store polygons and gains
polygons_EIRP = []
gains_EIRP = []
polygons_GT = []
gains_GT = []
for placemark in doc_EIRP.features:
    gain = float(placemark.name)
    coords = placemark.geometry.coords[0]
    coord_list = [(float(coord[0]), float(coord[1])) for coord in coords]
    polygon = Polygon(coord_list)
    polygons_EIRP.append(polygon)
    gains_EIRP.append(gain)
for placemark in doc_GT.features:
    gain = float(placemark.name)
    coords = placemark.geometry.coords[0]
    coord_list = [(float(coord[0]), float(coord[1])) for coord in coords]
    polygon = Polygon(coord_list)
    polygons_GT.append(polygon)
    gains_GT.append(gain)

# sort the polygons and gains by gain, descending
sorted_indices_EIRP = sorted(range(len(gains_EIRP)), key=lambda i: gains_EIRP[i], reverse=True)
sorted_gains_EIRP = [gains_EIRP[i] for i in sorted_indices_EIRP]
sorted_polygons_EIRP = [polygons_EIRP[i] for i in sorted_indices_EIRP]
sorted_indices_GT = sorted(range(len(gains_GT)), key=lambda i: gains_GT[i], reverse=True)
sorted_gains_GT = [gains_GT[i] for i in sorted_indices_GT]
sorted_polygons_GT = [polygons_GT[i] for i in sorted_indices_GT]

def highest_EIRP_query(lat, lon):
    """
    Given a latitude and longitude, return the highest gain in the EIRP polygons.
    """
    point = Point(lon, lat)
    for i, polygon in enumerate(sorted_polygons_EIRP):
        if polygon.contains(point):
            return sorted_gains_EIRP[i]
    point = Point(lon - 360, lat)  # shift the point to the left by 360 degrees
    for i, polygon in enumerate(sorted_polygons_EIRP):
        if polygon.contains(point):
            return sorted_gains_EIRP[i]
    return None

def highest_GT_query(lat, lon):
    """
    Given a latitude and longitude, return the highest gain in the GT polygons.
    """
    point = Point(lon, lat)
    for i, polygon in enumerate(sorted_polygons_GT):
        if polygon.contains(point):
            return sorted_gains_GT[i]
    point = Point(lon - 360, lat)  # shift the point to the left by 360 degrees
    for i, polygon in enumerate(sorted_polygons_EIRP):
        if polygon.contains(point):
            return sorted_gains_EIRP[i]
    return None

# plot all EIRP polygons
if __name__ == "__main__":
    opacity = 0.3
    # plot all EIRP polygons
    fig, ax = plt.subplots()
    for polygon in sorted_polygons_EIRP:
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=opacity, fc='blue', ec='black')
    ax.set_title("EIRP polygons")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()
    # plot all GT polygons
    fig, ax = plt.subplots()
    for polygon in sorted_polygons_GT:
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=opacity, fc='red', ec='black')
    ax.set_title("GT polygons")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()


#lat = 36
#lon = 19
#print(highest_EIRP_query(lat, lon))  # should return the highest gain in the EIRP polygons
#print(highest_GT_query(lat, lon))  # should return the highest gain in the GT polygons
