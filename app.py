this works beautifully well please just also include just a routing to pass on sea too you get shey please and labels of each creek and name at every turning should come up to as well, a point and name of river or creek send the updated python code ----# app_okiori_otakipo_streamlit.py 
"""
Streamlit app: HydroRIVERS-based route survey for Okiori -> Otakipo
- Upload HydroRIVERS (zipped shapefile) or paste URL
- Build waterway graph, find route or reachable subnetworks + nearest gap
- Folium map on Esri World Imagery
- Export GeoJSON and zipped shapefiles
Notes:
 - No satellite-derived bathymetry included (as requested).
 - Requires: streamlit, pyshp (shapefile), shapely, networkx, folium, requests, branca
"""

import os
import io
import math
import zipfile
import tempfile
import json
import random
from typing import List, Tuple, Dict

import streamlit as st
import shapefile  # pyshp
from shapely.geometry import LineString, Point, mapping, shape
from shapely.ops import unary_union
import networkx as nx
import folium
from folium import FeatureGroup
from streamlit_folium import folium_static
import requests
import branca

# ----------------- Helpers -----------------
st.set_page_config(page_title="Okiori → Otakipó Route Survey (HydroRIVERS)", layout="wide")

ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Esri World Imagery"

def extract_shp_from_zip(zpath: str, outdir: str) -> str:
    with zipfile.ZipFile(zpath, "r") as z:
        shp = next((n for n in z.namelist() if n.lower().endswith(".shp")), None)
        if not shp:
            return None
        stem = os.path.splitext(shp)[0]
        members = [m for m in z.namelist() if m.startswith(stem)]
        z.extractall(outdir, members)
        return os.path.join(outdir, shp)

def download_file(url: str, out_path: str, timeout=120):
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    return out_path

def read_river_shapefile(shp_path: str):
    """Return list of features: {'geom': LineString, 'props': dict, 'bbox': (minx,miny,maxx,maxy)}"""
    r = shapefile.Reader(shp_path)
    fields = [f[0] for f in r.fields[1:]]
    feats = []
    for sr in r.iterShapeRecords():
        shp = sr.shape
        rec = sr.record.as_dict() if hasattr(sr.record, "as_dict") else dict(zip(fields, list(sr.record)))
        # shapefile points are (lon, lat)
        parts = list(shp.parts) + [len(shp.points)]
        for i in range(len(parts)-1):
            seg = shp.points[parts[i]:parts[i+1]]
            if len(seg) < 2:
                continue
            try:
                geom = LineString(seg)
                feats.append({"geom": geom, "props": rec, "bbox": geom.bounds})
            except Exception:
                continue
    return feats, fields

def haversine_meters(lat1, lon1, lat2, lon2):
    # lat/lon in degrees -> meters
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def build_graph(features: List[Dict]):
    """Return networkx.Graph with nodes as (lon,lat) and edge weights in meters."""
    G = nx.Graph()
    for idx, f in enumerate(features):
        coords = list(f["geom"].coords)
        for i in range(len(coords)-1):
            a = coords[i]; b = coords[i+1]
            lon1, lat1 = a[0], a[1]
            lon2, lat2 = b[0], b[1]
            w = haversine_meters(lat1, lon1, lat2, lon2)
            if w <= 0 or (lon1==lon2 and lat1==lat2):
                continue
            G.add_node((lon1, lat1))
            G.add_node((lon2, lat2))
            if not G.has_edge((lon1, lat1), (lon2, lat2)):
                G.add_edge((lon1, lat1), (lon2, lat2), weight=w, seg_idx=idx)
    return G

def nearest_graph_node(G: nx.Graph, lat, lon):
    best = None; bestd = float("inf")
    for (x,y) in G.nodes():
        d = haversine_meters(lat, lon, y, x)
        if d < bestd:
            bestd = d; best=(x,y)
    return best, bestd

def path_linestring_from_nodes(nodes: List[Tuple[float,float]]):
    # nodes are sequence of (lon,lat)
    return LineString([(n[0], n[1]) for n in nodes])

def graph_component_edges_to_lines(G: nx.Graph, comp_nodes: List[Tuple[float,float]]):
    seen = set()
    lines = []
    for u in comp_nodes:
        for v in G.neighbors(u):
            a,b = (u,v) if u<=v else (v,u)
            if (a,b) in seen: continue
            seen.add((a,b))
            lines.append(LineString([a,b]))
    return lines

def make_geojson_feature_collection(items):
    return {"type":"FeatureCollection","features":items}

def feature_to_geojson(geom, props=None):
    return {"type":"Feature", "properties":props or {}, "geometry": mapping(geom)}

def buffer_degrees_approx(geom: LineString, meters: float):
    # approximate buffer: convert to degrees by mean latitude
    if geom.is_empty:
        return geom
    lon_mean = sum([c[0] for c in geom.coords]) / len(geom.coords)
    lat_mean = sum([c[1] for c in geom.coords]) / len(geom.coords)
    # convert meters to degrees (rough)
    deg_per_meter_lat = 1.0 / 111320.0
    deg_per_meter_lon = 1.0 / (111320.0 * math.cos(math.radians(lat_mean)))
    # scale x (lon) coordinates before buffering to make buffer roughly isotropic
    # transform coords into scaled space, buffer, then inverse-scale
    from shapely.affinity import scale, translate
    scale_x = deg_per_meter_lon / deg_per_meter_lat if deg_per_meter_lat!=0 else 1.0
    geom_scaled = scale(geom, xfact=scale_x, yfact=1.0, origin=(0,0))
    buff_scaled = geom_scaled.buffer(meters * deg_per_meter_lat)
    buff = scale(buff_scaled, xfact=1.0/scale_x, yfact=1.0, origin=(0,0))
    return buff

def write_shapefile_polyline_zip(lines: List[LineString], out_zip_path: str, layer_name="lines", props_list=None, crs_epsg=4326):
    """
    Write a zipped shapefile (polylines) using pyshp.
    lines: list of shapely LineString
    props_list: optional list of dicts for attributes (same length)
    """
    import shapefile as _shp
    tmpdir = tempfile.mkdtemp(prefix="shpout_")
    shp_p = os.path.join(tmpdir, f"{layer_name}.shp")
    w = _shp.Writer(shp_p, shapeType=_shp.POLYLINE)
    # add fields if any props present
    if props_list and len(props_list)>0:
        keys = list(props_list[0].keys())
        for k in keys:
            w.field(k, "C", size=254)
    else:
        keys = []
    for i, ln in enumerate(lines):
        parts = [list(ln.coords)]
        w.line(parts)
        if keys:
            row = [str(props_list[i].get(k,"")) for k in keys]
            w.record(*row)
    w.close()
    # build zip
    shp_files = [os.path.join(tmpdir, f"{layer_name}{ext}") for ext in [".shp",".shx",".dbf"]]
    with zipfile.ZipFile(out_zip_path, "w") as z:
        for p in shp_files:
            if os.path.exists(p):
                z.write(p, arcname=os.path.basename(p))
    return out_zip_path

def write_shapefile_point_zip(points: List[Tuple[float,float]], out_zip_path: str, layer_name="points", props_list=None):
    import shapefile as _shp
    tmpdir = tempfile.mkdtemp(prefix="shpout_")
    shp_p = os.path.join(tmpdir, f"{layer_name}.shp")
    w = _shp.Writer(shp_p, shapeType=_shp.POINT)
    if props_list and len(props_list)>0:
        keys = list(props_list[0].keys())
        for k in keys:
            w.field(k, "C", size=254)
    else:
        keys = []
    for i, pt in enumerate(points):
        w.point(pt[0], pt[1])
        if keys:
            row = [str(props_list[i].get(k,"")) for k in keys]
            w.record(*row)
    w.close()
    shp_files = [os.path.join(tmpdir, f"{layer_name}{ext}") for ext in [".shp",".shx",".dbf"]]
    with zipfile.ZipFile(out_zip_path, "w") as z:
        for p in shp_files:
            if os.path.exists(p):
                z.write(p, arcname=os.path.basename(p))
    return out_zip_path

# ----------------- Streamlit UI -----------------
st.title("🌊 Okiori → Otakipó Route Survey (HydroRIVERS routing)")

st.markdown("""
This tool builds a HydroRIVERS-only routing network and attempts to find a continuous water route between Start and End.
If no continuous HydroRIVERS path exists, the app exports the reachable subnetworks and the nearest gap (so you can see where the inland waterway breaks).
Basemap: **Esri World Imagery**. No satellite-derived bathymetry included (per request).
""")

with st.sidebar:
    st.header("1) HydroRIVERS input")
    choice = st.radio("Provide HydroRIVERS shapefile (.zip or .shp):", ["Upload ZIP", "Paste ZIP URL", "Local path on server"])
    shp_path = None
    if choice == "Upload ZIP":
        upl = st.file_uploader("Upload HydroRIVERS .zip (zipped shapefile)", type=["zip"])
        if upl is not None:
            tmp = tempfile.mkdtemp(prefix="hydro_upload_")
            zf = os.path.join(tmp, "uploaded.zip")
            with open(zf, "wb") as f:
                f.write(upl.read())
            st.success("Saved uploaded zip.")
            shp_path = extract_shp_from_zip(zf, tmp)
            if not shp_path:
                st.error("Uploaded zip does not contain .shp files.")
    elif choice == "Paste ZIP URL":
        url = st.text_input("Direct .zip URL (HydroRIVERS .zip)")
        if st.button("Download ZIP from URL"):
            try:
                tmp = tempfile.mkdtemp(prefix="hydro_dl_")
                zf = os.path.join(tmp, "hydro.zip")
                download_file(url, zf)
                shp_path = extract_shp_from_zip(zf, tmp)
                if not shp_path:
                    st.error("Downloaded zip did not contain a .shp.")
                else:
                    st.success("Downloaded and extracted shapefile.")
            except Exception as e:
                st.error(f"Download failed: {e}")
    else:
        local = st.text_input("Local path to .zip or .shp on server (full path)")
        if local:
            if os.path.exists(local):
                if local.lower().endswith(".zip"):
                    tmp = tempfile.mkdtemp(prefix="hydro_local_")
                    shp_path = extract_shp_from_zip(local, tmp)
                    if shp_path:
                        st.success("Extracted shapefile from local zip.")
                    else:
                        st.error("Local zip did not contain .shp")
                elif local.lower().endswith(".shp"):
                    shp_path = local
                    st.success("Using provided .shp.")
                else:
                    st.error("Provide full path to .zip or .shp")
            else:
                st.info("Path not found (yet).")

    st.markdown("---")
    st.header("2) Anchors (Start / End)")
    col1, col2 = st.columns(2)
    start_lat = col1.number_input("Start lat (deg)", value=4.925000, format="%.6f")
    start_lon = col2.number_input("Start lon (deg)", value=6.285000, format="%.6f")
    col3, col4 = st.columns(2)
    end_lat = col3.number_input("End lat (deg)", value=4.815000, format="%.6f")
    end_lon = col4.number_input("End lon (deg)", value=7.035000, format="%.6f")

    st.markdown("---")
    st.header("3) Options")
    corridor_m = st.slider("Corridor buffer (meters)", 50, 2000, 500, step=50)
    allow_autobridge = st.checkbox("Allow short autobridge (propose connector across small gaps)", value=True)
    max_autobridge = st.number_input("Max autobridge length (m)", value=2500)
    sample_limit = st.number_input("Max nodes sampled per component (performance)", value=5000)

    st.markdown("---")
    compute = st.button("Compute Route / Export")

# If shp_path not set from sidebar actions (upload/download/local), try to use previously obtained
if 'shp_path' not in st.session_state:
    st.session_state['shp_path'] = None

if shp_path:
    st.session_state['shp_path'] = shp_path

if st.session_state.get('shp_path'):
    st.sidebar.success("HydroRIVERS loaded.")
else:
    st.sidebar.info("No HydroRIVERS shapefile yet; upload or provide URL or server path.")

# Main compute logic
if compute:
    if not st.session_state.get('shp_path'):
        st.error("No HydroRIVERS shapefile available. Provide it in the sidebar.")
    else:
        try:
            with st.spinner("Reading HydroRIVERS shapefile..."):
                feats, fields = read_river_shapefile(st.session_state['shp_path'])
                st.info(f"HydroRIVERS features read: {len(feats)} parts (split segments).")
            # restrict features to bbox around anchors (pad a little)
            pad_deg = 0.6
            minlon = min(start_lon, end_lon) - pad_deg
            maxlon = max(start_lon, end_lon) + pad_deg
            minlat = min(start_lat, end_lat) - pad_deg
            maxlat = max(start_lat, end_lat) + pad_deg
            aoi_feats = [f for f in feats if not (f['bbox'][0] > maxlon or f['bbox'][2] < minlon or f['bbox'][1] > maxlat or f['bbox'][3] < minlat)]
            if not aoi_feats:
                st.warning("No HydroRIVERS features inside expanded AOI — using full dataset instead.")
                aoi_feats = feats

            # Build networkx graph
            with st.spinner("Building water graph..."):
                G = build_graph(aoi_feats)
                st.success(f"Graph built: nodes={len(G.nodes())}, edges={len(G.edges())}.")

            # Snap anchors
            start_node, start_snap_m = nearest_graph_node(G, start_lat, start_lon)
            end_node, end_snap_m = nearest_graph_node(G, end_lat, end_lon)
            st.write(f"Snap distances → Start: {start_snap_m/1000:.2f} km, End: {end_snap_m/1000:.2f} km")

            # attempt path
            with st.spinner("Attempting shortest path on HydroRIVERS network..."):
                try:
                    path_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
                    st.success("Continuous HydroRIVERS path found.")
                    route_geom = path_linestring_from_nodes(path_nodes)
                    found_route = True
                except nx.NetworkXNoPath:
                    st.warning("No continuous HydroRIVERS path found between Start and End.")
                    route_geom = None
                    found_route = False
                except Exception as e:
                    st.error(f"Routing error: {e}")
                    route_geom = None
                    found_route = False

            # If no path: compute components, nearest gap, and optionally autobridge
            reach_start_lines = reach_end_lines = []
            gap_line = None
            gap_info = None
            if not found_route:
                with st.spinner("Computing reachable subnetworks and nearest gap..."):
                    comps = list(nx.connected_components(G))
                    # map node->comp index
                    comp_map = {}
                    for i, c in enumerate(comps):
                        for n in c:
                            comp_map[n] = i
                    s_comp = comp_map.get(start_node, None)
                    t_comp = comp_map.get(end_node, None)

                    if s_comp is None or t_comp is None:
                        st.error("Could not identify component for start or end node.")
                    else:
                        # build edge lists -> LineStrings
                        compS_nodes = list(comps[s_comp])
                        compT_nodes = list(comps[t_comp])
                        reach_start_lines = graph_component_edges_to_lines(G, compS_nodes)
                        reach_end_lines = graph_component_edges_to_lines(G, compT_nodes)

                        # nearest gap between components (sample limited)
                        def nearest_gap_sample(G, A_nodes, B_nodes, cap):
                            A = A_nodes if len(A_nodes) <= cap else random.sample(A_nodes, cap)
                            B = B_nodes if len(B_nodes) <= cap else random.sample(B_nodes, cap)
                            best = (None, None, 1e12)
                            for a in A:
                                for b in B:
                                    lat_a, lon_a = a[1], a[0]
                                    lat_b, lon_b = b[1], b[0]
                                    d = haversine_meters(lat_a, lon_a, lat_b, lon_b)
                                    if d < best[2]:
                                        best = (a, b, d)
                            return best
                        ia_node, ib_node, gap_m = nearest_gap_sample(G, compS_nodes, compT_nodes, cap=sample_limit)
                        gap_line = LineString([ (ia_node[0], ia_node[1]), (ib_node[0], ib_node[1]) ])
                        gap_info = {"gap_m": gap_m, "a_node": ia_node, "b_node": ib_node}
                        st.info(f"Nearest gap ≈ {gap_m/1000.0:.2f} km")

                        # Autobridge option: add edge if within threshold and retry path
                        if allow_autobridge and gap_m <= max_autobridge:
                            st.info("Autobridge allowed and gap within threshold → proposing connector and retrying route...")
                            # add edge to G temporarily
                            try:
                                G.add_edge(ia_node, ib_node, weight=gap_m, seg_idx=-999)
                                # retry
                                try:
                                    path_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
                                    route_geom = path_linestring_from_nodes(path_nodes)
                                    found_route = True
                                    st.success("Route found after adding autobridge connector.")
                                except nx.NetworkXNoPath:
                                    st.warning("Still no path after autobridge (unexpected).")
                                    found_route = False
                            except Exception as e:
                                st.error(f"Autobridge failed: {e}")

            # Prepare GeoJSONs for export and map layers
            geo_layers = {}
            # HydroRIVERS (AOI) as FeatureCollection (sample to keep map fast)
            sample_N = 2000
            sample_feats = aoi_feats if len(aoi_feats) <= sample_N else random.sample(aoi_feats, sample_N)
            hr_features = [feature_to_geojson(f['geom'], f['props']) for f in sample_feats]
            geo_layers['hydrorivers_aoi'] = make_geojson_feature_collection(hr_features)

            if found_route and route_geom is not None:
                geo_layers['route'] = make_geojson_feature_collection([feature_to_geojson(route_geom, {"name":"Route_OK→OT"})])
                # corridor polygon
                corridor_poly = buffer_degrees_approx(route_geom, corridor_m)
                geo_layers['corridor'] = make_geojson_feature_collection([feature_to_geojson(corridor_poly, {"name":"Corridor"})])
            else:
                # export reaches and gap
                if reach_start_lines:
                    geo_layers['reach_start'] = make_geojson_feature_collection([feature_to_geojson(l, {"role":"reach_start"}) for l in reach_start_lines])
                if reach_end_lines:
                    geo_layers['reach_end'] = make_geojson_feature_collection([feature_to_geojson(l, {"role":"reach_end"}) for l in reach_end_lines])
                if gap_line is not None:
                    geo_layers['gap_line'] = make_geojson_feature_collection([feature_to_geojson(gap_line, {"gap_m": gap_info['gap_m']})])
                    # break points
                    geo_layers['break_points'] = make_geojson_feature_collection([
                        feature_to_geojson(Point(gap_info['a_node'][0], gap_info['a_node'][1]), {"role":"break_A","gap_m":gap_info['gap_m']}),
                        feature_to_geojson(Point(gap_info['b_node'][0], gap_info['b_node'][1]), {"role":"break_B","gap_m":gap_info['gap_m']})
                    ])

            # anchors
            anchors_fc = make_geojson_feature_collection([
                feature_to_geojson(Point(start_lon, start_lat), {"name":"Start"}),
                feature_to_geojson(Point(end_lon, end_lat), {"name":"End"})
            ])
            geo_layers['anchors'] = anchors_fc

            # Build Folium map
            center_lat = (start_lat + end_lat) / 2.0
            center_lon = (start_lon + end_lon) / 2.0
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)
            folium.TileLayer(ESRI_TILES, name=ESRI_ATTR, attr=ESRI_ATTR, overlay=False, control=False).add_to(m)

            # Add HydroRIVERS (sample)
            fg_hr = FeatureGroup(name="HydroRIVERS (sample)", show=True)
            folium.GeoJson(geo_layers['hydrorivers_aoi'], name="HydroRIVERS",
                           style_function=lambda x: {"color":"#2c7fb8", "weight":1, "opacity":0.6}).add_to(fg_hr)
            fg_hr.add_to(m)

            # Anchors
            fg_anchors = FeatureGroup(name="Anchors", show=True)
            for f in geo_layers['anchors']['features']:
                coords = f['geometry']['coordinates']
                folium.Marker([coords[1], coords[0]], popup=f"<b>{f['properties']['name']}</b>", icon=folium.Icon(color='green' if f['properties']['name']=="Start" else 'red')).add_to(fg_anchors)
            fg_anchors.add_to(m)

            # Route / corridor
            if 'route' in geo_layers:
                folium.GeoJson(geo_layers['route'], name="Route", style_function=lambda x: {"color":"#ff7f00","weight":4}).add_to(m)
            if 'corridor' in geo_layers:
                folium.GeoJson(geo_layers['corridor'], name="Corridor", style_function=lambda x: {"color":"#ff7f00","fillColor":"#ff7f0044","weight":1, "fillOpacity":0.2}).add_to(m)

            # Reaches / gap
            if 'reach_start' in geo_layers:
                folium.GeoJson(geo_layers['reach_start'], name="Reachable from Start", style_function=lambda x: {"color":"#00b050","weight":2}).add_to(m)
            if 'reach_end' in geo_layers:
                folium.GeoJson(geo_layers['reach_end'], name="Reachable from End", style_function=lambda x: {"color":"#ff9f00","weight":2}).add_to(m)
            if 'gap_line' in geo_layers:
                folium.GeoJson(geo_layers['gap_line'], name="Nearest Gap", style_function=lambda x: {"color":"#d73027","weight":3,"dashArray":"6,6"}).add_to(m)
            if 'break_points' in geo_layers:
                folium.GeoJson(geo_layers['break_points'], name="Break Points", marker=folium.CircleMarker(radius=6)).add_to(m)

            folium.LayerControl().add_to(m)

            # Legend (simple HTML)
            legend_html = """
             <div style="
             position: fixed; 
             bottom: 50px; left: 10px; width: 220px; height: auto; 
             background-color: rgba(255,255,255,0.9);
             box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
             z-index:9999; font-size:12px;
             padding:10px; border-radius:6px;">
             <b>Legend</b><br>
             <i style="background:#2c7fb8;width:18px;height:6px;display:inline-block;margin-right:6px;"></i> HydroRIVERS (sample)<br>
             <i style="background:#ff7f00;width:18px;height:6px;display:inline-block;margin-right:6px;"></i> Route (corridor)<br>
             <i style="background:#00b050;width:18px;height:6px;display:inline-block;margin-right:6px;"></i> Reach from Start<br>
             <i style="background:#ff9f00;width:18px;height:6px;display:inline-block;margin-right:6px;"></i> Reach from End<br>
             <i style="background:#d73027;width:18px;height:6px;display:inline-block;margin-right:6px;"></i> Nearest Gap<br>
             <div style='margin-top:6px;font-size:11px;color:#555;'>Snap distances (km): Start={:.2f}, End={:.2f}</div>
             </div>
            """.format(start_snap_m/1000.0, end_snap_m/1000.0)
            m.get_root().html.add_child(branca.element.MacroElement().add_child(branca.element.Element(legend_html)))

            st.subheader("Map preview")
            folium_static(m, width=1000, height=700)

            # Exports: GeoJSONs and zipped shapefiles
            st.subheader("Export results")
            # prepare files in-memory
            mem_files = {}
            # route GeoJSON
            if 'route' in geo_layers:
                mem_files['route.geojson'] = json.dumps(geo_layers['route'])
            if 'corridor' in geo_layers:
                mem_files['corridor.geojson'] = json.dumps(geo_layers['corridor'])
            if 'reach_start' in geo_layers:
                mem_files['reach_start.geojson'] = json.dumps(geo_layers['reach_start'])
            if 'reach_end' in geo_layers:
                mem_files['reach_end.geojson'] = json.dumps(geo_layers['reach_end'])
            if 'gap_line' in geo_layers:
                mem_files['gap_line.geojson'] = json.dumps(geo_layers['gap_line'])
            if 'break_points' in geo_layers:
                mem_files['break_points.geojson'] = json.dumps(geo_layers['break_points'])
            mem_files['anchors.geojson'] = json.dumps(geo_layers['anchors'])

            # Offer GeoJSON zips for download
            for name, data in mem_files.items():
                st.download_button(f"Download {name}", data, file_name=name, mime="application/geo+json")

            # Also create zipped shapefile (route + corridor + gap + reaches + points) if present
            shp_zip_path = os.path.join(tempfile.mkdtemp(prefix="shpzip_"), "okiori_otakipo_outputs.zip")
            created_any = False
            to_zip_files = []
            # create shapefile zips individually and then combine into one big zip
            temp_shp_zips = []
            if 'route' in geo_layers:
                route_coords = shape(geo_layers['route']['features'][0]['geometry'])
                p = write_shapefile_polyline_zip([route_coords], os.path.join(os.path.dirname(shp_zip_path), "route_shp.zip"), layer_name="route")
                temp_shp_zips.append(p); created_any = True
            if 'corridor' in geo_layers:
                corridor_geom = shape(geo_layers['corridor']['features'][0]['geometry'])
                p = write_shapefile_polyline_zip([corridor_geom.boundary], os.path.join(os.path.dirname(shp_zip_path), "corridor_shp.zip"), layer_name="corridor")
                temp_shp_zips.append(p); created_any = True
            if 'reach_start' in geo_layers:
                lines = [shape(f['geometry']) for f in geo_layers['reach_start']['features']]
                p = write_shapefile_polyline_zip(lines, os.path.join(os.path.dirname(shp_zip_path), "reach_start_shp.zip"), layer_name="reach_start")
                temp_shp_zips.append(p); created_any = True
            if 'reach_end' in geo_layers:
                lines = [shape(f['geometry']) for f in geo_layers['reach_end']['features']]
                p = write_shapefile_polyline_zip(lines, os.path.join(os.path.dirname(shp_zip_path), "reach_end_shp.zip"), layer_name="reach_end")
                temp_shp_zips.append(p); created_any = True
            if 'gap_line' in geo_layers:
                lines = [shape(f['geometry']) for f in geo_layers['gap_line']['features']]
                p = write_shapefile_polyline_zip(lines, os.path.join(os.path.dirname(shp_zip_path), "gap_shp.zip"), layer_name="gap")
                temp_shp_zips.append(p); created_any = True
            if 'break_points' in geo_layers:
                pts = []
                for f in geo_layers['break_points']['features']:
                    g = shape(f['geometry'])
                    pts.append((g.x, g.y))
                p = write_shapefile_point_zip(pts, os.path.join(os.path.dirname(shp_zip_path), "break_points_shp.zip"), layer_name="break_points")
                temp_shp_zips.append(p); created_any = True
            # anchors
            anchors_pts = []
            for f in geo_layers['anchors']['features']:
                g = shape(f['geometry'])
                anchors_pts.append((g.x, g.y))
            p = write_shapefile_point_zip(anchors_pts, os.path.join(os.path.dirname(shp_zip_path), "anchors_shp.zip"), layer_name="anchors")
            temp_shp_zips.append(p); created_any = True

            # Combine the individual zips into a single zip (flat)
            if created_any:
                with zipfile.ZipFile(shp_zip_path, "w") as outz:
                    for zf in temp_shp_zips:
                        with zipfile.ZipFile(zf, "r") as zin:
                            for member in zin.namelist():
                                outz.writestr(os.path.basename(zf).replace(".zip","") + "/" + member, zin.read(member))
                # read into bytes for download
                with open(shp_zip_path, "rb") as f:
                    data = f.read()
                st.download_button("Download all shapefile outputs (zipped)", data=data, file_name="okiori_otakipo_shapefiles_bundle.zip", mime="application/zip")
            else:
                st.info("No shapefile outputs to create.")

        except Exception as e:
            st.exception(e)
