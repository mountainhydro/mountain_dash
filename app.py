import json
from pathlib import Path

import ee
import folium
import streamlit as st
from streamlit_folium import st_folium

SAMPLE_PATH = Path("data/mountain_sample_1000.geojson")

st.set_page_config(page_title="Kapos Mountain Classes", layout="wide")
st.title("Kapos Mountain Classes (Global)")
st.caption(
    "Elevation-based mountain classification after Kapos et al. (2000) "
    "derived from USGS SRTM 30 m. Satellite basemap © Google."
)

# ── GEE authentication ────────────────────────────────────────────────────────
# Local: uses credentials from `earthengine authenticate`
# Streamlit Cloud: store your service-account JSON in st.secrets["GEE_SERVICE_ACCOUNT"]
PROJECT = "promising-era-496715-j5"

@st.cache_resource
def init_ee():
    import json
    if "GEE_SERVICE_ACCOUNT" in st.secrets:
        info = st.secrets["GEE_SERVICE_ACCOUNT"]
        credentials = ee.ServiceAccountCredentials(
            email=info["client_email"],
            key_data=json.dumps(dict(info)),
        )
        try:
            ee.Initialize(credentials, project=PROJECT)
        except Exception as exc:
            st.error(f"Earth Engine initialization failed: {exc}")
            st.stop()
    else:
        try:
            ee.Initialize(project=PROJECT)
        except Exception as exc:
            st.error(f"Earth Engine credentials not found or initialization failed: {exc}")
            st.stop()

init_ee()

# ── Terrain & Kapos classification ────────────────────────────────────────────
@st.cache_resource
def build_layers():
    srtm   = ee.Image("USGS/SRTMGL1_003").select("elevation")
    srtm90 = ee.Image("CGIAR/SRTM90_V4").select("elevation")
    slope  = ee.Terrain.slope(srtm)
    local_relief = (
        srtm90.focal_max(radius=7000, kernelType="circle", units="meters")
        .subtract(srtm90.focal_min(radius=7000, kernelType="circle", units="meters"))
    )
    specs = [
        (srtm.updateMask(srtm.gt(4500)),
         {"min": 4500, "max": 8850, "palette": ["#9e9ac8", "#54278f"]},
         "K1  > 4 500 m", 0.90),
        (srtm.updateMask(srtm.gte(3500).And(srtm.lte(4500))),
         {"min": 3500, "max": 4500, "palette": ["#6baed6", "#2171b5"]},
         "K2  3 500–4 500 m", 0.90),
        (srtm.updateMask(srtm.gte(2500).And(srtm.lt(3500))),
         {"min": 2500, "max": 3500, "palette": ["#74c476", "#08519c"]},
         "K3  2 500–3 500 m", 0.90),
        (srtm.updateMask(srtm.gte(1500).And(srtm.lt(2500)).And(slope.gte(2))),
         {"min": 1500, "max": 2500, "palette": ["#a6d96a", "#1a9641"]},
         "K4  1 500–2 500 m, slope ≥ 2°", 0.85),
        (srtm.updateMask(srtm.gte(1000).And(srtm.lt(1500)).And(slope.gte(5).Or(local_relief.gt(300)))),
         {"min": 1000, "max": 1500, "palette": ["#fee08b", "#d9ef8b"]},
         "K5  1 000–1 500 m, slope ≥ 5° or relief > 300 m", 0.85),
        (srtm.updateMask(srtm.gte(300).And(srtm.lt(1000)).And(local_relief.gt(300))),
         {"min": 300, "max": 1000, "palette": ["#fdae61", "#f46d43"]},
         "K6  300–1 000 m, relief > 300 m (7 km radius)", 0.85),
    ]
    return [
        (name, ee.Image(img).getMapId(vis)["tile_fetcher"].url_format, opacity)
        for img, vis, name, opacity in specs
    ]

layers = build_layers()

# ── Folium map ────────────────────────────────────────────────────────────────
m = folium.Map(location=[30, 15], zoom_start=3)
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google", name="Satellite", overlay=False,
).add_to(m)
for name, tile_url, opacity in layers:
    folium.raster_layers.TileLayer(
        tiles=tile_url,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        opacity=opacity,
    ).add_to(m)

legend_html = """
<div style="position:fixed;bottom:30px;right:30px;z-index:1000;background:white;
            padding:12px 16px;border-radius:8px;border:1px solid #ccc;
            font-family:Arial,sans-serif;font-size:13px;line-height:2;color:black;">
  <b>Kapos Mountain Classes</b><br>
  <span style="background:#54278f;display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:middle;"></span>K1 &gt; 4 500 m<br>
  <span style="background:#2171b5;display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:middle;"></span>K2 3 500–4 500 m<br>
  <span style="background:#08519c;display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:middle;"></span>K3 2 500–3 500 m<br>
  <span style="background:#1a9641;display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:middle;"></span>K4 1 500–2 500 m, slope &#8805; 2°<br>
  <span style="background:#fee08b;display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:middle;"></span>K5 1 000–1 500 m, slope &#8805; 5° or relief &gt; 300 m<br>
  <span style="background:#f46d43;display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:middle;"></span>K6 300–1 000 m, relief &gt; 300 m (7 km radius)<br>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ── Sample points ─────────────────────────────────────────────────────────────
if SAMPLE_PATH.exists():
    with open(SAMPLE_PATH) as f:
        sample_geojson = json.load(f)

    fg = folium.FeatureGroup(name="random sample+kmeansEmb (1000p)", show=True)
    for feat in sample_geojson["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        p = feat["properties"]
        color = "#e31a1c"
        elev   = p.get("elevation")
        slope  = p.get("slope")
        kapos  = p.get("kapos_class")
        tip = (
            f"<b>{p.get('stratum', '—')}</b><br>"
            + (f"Elevation: {elev:.0f} m<br>" if elev is not None else "")
            + (f"Kapos: K{int(kapos)}<br>"    if kapos is not None else "")
            + (f"Slope: {slope:.1f}°"         if slope is not None else "")
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=0.5,
            tooltip=tip,
        ).add_to(fg)
    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, use_container_width=True, height=700)
