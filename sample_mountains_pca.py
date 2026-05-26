#!/usr/bin/env python3
"""
Mountain sampling via PCA-quantized GEE stratification + k-means++ cosine dissimilarity.

Stages (run in order):

  1. Pilot export — small random sample used to fit PCA on the embeddings
       python sample_mountains_pca.py --pilot

  2. Stratified export — apply PCA in GEE, quantize into a grid, stratifiedSample
       python sample_mountains_pca.py --sample

  3. Cluster — k-means++ with cosine dissimilarity → 1 000 final points
       python sample_mountains_pca.py --cluster

Files produced:
  data/pilot_kapos.csv          ~5 000 random points used to fit PCA
  data/candidates_pca_raw.csv   ~200 000 points, one stratum per PC-grid cell
  data/mountain_sample_pca.geojson  final 1 000 selected points
"""

import argparse
import json
import time
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT        = "promising-era-496715-j5"
SRTM_ID        = "USGS/SRTMGL1_003"
ALPHA_EARTH_IC = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
YEAR           = 2023
SAMPLE_SCALE   = 500
WORLD          = ee.Geometry.BBox(-180, -90, 180, 90)

PILOT_N        = 5_000    # points for PCA fitting
N_COMPONENTS   = 5        # PCs to use for stratification
N_BINS         = 3        # bins per PC  →  3^5 = 243 possible strata
PTS_PER_STRAT  = 2_000    # GEE samples per stratum  →  ~200–300k total

PILOT_CSV      = Path("data/pilot_kapos.csv")
CANDIDATES_CSV = Path("data/candidates_pca_raw.csv")
OUTPUT_GEOJSON = Path("data/mountain_sample_pca.geojson")
NE_CACHE       = Path("data/ne_countries.gpkg")
NE_URL         = ("https://naciscdn.org/naturalearth/110m/cultural/"
                  "ne_110m_admin_0_countries.zip")

CONTINENT_TO_STRATUM = {
    "Asia":          "Asia_Central",
    "South America": "South_America",
    "North America": "North_America",
    "Europe":        "Europe",
    "Africa":        "Africa",
    "Oceania":       "Other",
}
STRATA_QUOTAS = {
    "Asia_Central":  380,
    "South_America": 200,
    "North_America": 120,
    "Europe":        100,
    "Africa":         80,
    "Other":         120,
}
assert sum(STRATA_QUOTAS.values()) == 1000

EMB_COLS       = [f"emb_{i}" for i in range(64)]
ANCILLARY_COLS = ["elevation", "slope", "aspect", "kapos_class"]
FEATURE_COLS   = EMB_COLS + ANCILLARY_COLS


# ── GEE helpers ───────────────────────────────────────────────────────────────
def init_gee():
    try:
        ee.Initialize(project=PROJECT)
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize(project=PROJECT)
    print("GEE initialised.")


def build_composite():
    """Full Kapos (K1–K6) composite: AlphaEarth + terrain + kapos_class."""
    srtm   = ee.Image(SRTM_ID).select("elevation")
    slope  = ee.Terrain.slope(srtm).rename("slope")
    aspect = ee.Terrain.aspect(srtm).rename("aspect")

    kernel      = ee.Kernel.circle(7000, "meters")
    local_range = (srtm.reduceNeighborhood(ee.Reducer.max(), kernel, optimization="boxcar")
                       .subtract(srtm.reduceNeighborhood(ee.Reducer.min(), kernel, optimization="boxcar"))
                       .rename("local_range"))

    k1 = srtm.gte(4500)
    k2 = srtm.gte(3500).And(srtm.lt(4500))
    k3 = srtm.gte(2500).And(srtm.lt(3500))
    k4 = srtm.gte(1500).And(srtm.lt(2500)).And(slope.gte(2).Or(local_range.gte(300)))
    k5 = srtm.gte(1000).And(srtm.lt(1500)).And(slope.gte(5).Or(local_range.gte(300)))
    k6 = srtm.gte( 300).And(srtm.lt(1000)).And(local_range.gte(300))

    kapos_class   = (ee.Image(0)
                     .where(k6,6).where(k5,5).where(k4,4)
                     .where(k3,3).where(k2,2).where(k1,1)
                     .rename("kapos_class"))
    mountain_mask = k1.Or(k2).Or(k3).Or(k4).Or(k5).Or(k6)

    n_bands = len(
        ee.ImageCollection(ALPHA_EARTH_IC)
          .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
          .first().bandNames().getInfo()
    )
    alpha = (ee.ImageCollection(ALPHA_EARTH_IC)
               .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
               .mosaic()
               .rename([f"emb_{i}" for i in range(n_bands)]))

    return (alpha
            .addBands(srtm.rename("elevation"))
            .addBands(slope)
            .addBands(aspect)
            .addBands(kapos_class)
            .updateMask(mountain_mask)), n_bands


def export_to_drive(fc, description, prefix, selectors):
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder="MountAInWater",
        fileNamePrefix=prefix,
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    return task


def wait_for_task(task, poll=30):
    print(f"Task {task.id} submitted. Polling every {poll}s …")
    while True:
        state = task.status()["state"]
        print(f"  [{time.strftime('%H:%M:%S')}] {state}")
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(poll)
    if state != "COMPLETED":
        raise RuntimeError(f"Export failed: {task.status().get('error_message')}")
    print("Export complete.")


# ── Stage 1: pilot export ─────────────────────────────────────────────────────
def run_pilot(wait=False):
    """Export PILOT_N random Kapos points to fit PCA loadings."""
    init_gee()
    composite, n_bands = build_composite()

    emb_cols  = [f"emb_{i}" for i in range(n_bands)]
    selectors = ["longitude", "latitude", "elevation", "slope", "aspect",
                 "kapos_class"] + emb_cols

    fc   = composite.sample(region=WORLD, scale=SAMPLE_SCALE,
                             projection="EPSG:4326", numPixels=PILOT_N,
                             seed=0, geometries=True, dropNulls=True,
                             tileScale=4)
    fc   = fc.map(lambda f: f.set({
        "longitude": f.geometry().coordinates().get(0),
        "latitude":  f.geometry().coordinates().get(1),
    }))
    task = export_to_drive(fc, "kapos_pilot_5k", "pilot_kapos", selectors)
    print(f"Pilot task ID: {task.id}")
    if wait:
        wait_for_task(task)


# ── Stage 2: PCA + stratified export ─────────────────────────────────────────
def build_pc_image(alpha, components):
    """
    Apply PCA loading matrix to AlphaEarth bands.
    Each PC is a weighted sum of the 64 embedding bands.
    """
    pc_images = []
    for i, row in enumerate(components):
        weights  = ee.Image.constant(row.tolist())          # 64-band constant image
        pc_score = alpha.multiply(weights).reduce(ee.Reducer.sum()).rename(f"pc_{i}")
        pc_images.append(pc_score)
    return ee.Image(pc_images)


def build_strat_image(pc_image, thresholds_per_pc):
    """
    Quantize PC scores into bins and combine into a single integer stratum ID.
    thresholds_per_pc: list of (N_BINS-1) threshold values per PC
    """
    bin_images = []
    for i, thresholds in enumerate(thresholds_per_pc):
        band = pc_image.select(f"pc_{i}")
        bimg = ee.Image(0)
        for j, t in enumerate(thresholds):
            bimg = bimg.where(band.gt(float(t)), j + 1)
        bin_images.append(bimg.rename(f"bin_{i}"))

    # Encode (bin_0, …, bin_{N-1}) as a single integer: mixed-radix
    strat = ee.Image(0)
    for i in range(N_COMPONENTS):
        strat = strat.add(bin_images[i].multiply(N_BINS ** (N_COMPONENTS - 1 - i)))
    return strat.rename("strat_id")


def run_sample(wait=False):
    """Fit PCA on pilot, apply in GEE, stratifiedSample → candidates CSV."""
    if not PILOT_CSV.exists():
        raise FileNotFoundError(f"{PILOT_CSV} not found — run --pilot first.")

    # Fit PCA on pilot embeddings
    pilot = pd.read_csv(PILOT_CSV).dropna(subset=EMB_COLS)
    pca   = PCA(n_components=N_COMPONENTS)
    pilot_pcs = pca.fit_transform(pilot[EMB_COLS].values)
    print(f"PCA variance explained: "
          f"{pca.explained_variance_ratio_.cumsum()[-1]:.1%} (top {N_COMPONENTS} PCs)")

    thresholds = [
        np.percentile(pilot_pcs[:, i], np.linspace(0, 100, N_BINS + 1)[1:-1]).tolist()
        for i in range(N_COMPONENTS)
    ]

    # Save loadings alongside the pilot CSV for reproducibility
    loadings_path = PILOT_CSV.with_suffix(".pca.json")
    loadings_path.write_text(json.dumps({
        "components": pca.components_.tolist(),
        "thresholds": thresholds,
    }))
    print(f"PCA loadings saved → {loadings_path}")

    # Apply in GEE
    init_gee()
    composite, n_bands = build_composite()
    alpha      = composite.select([f"emb_{i}" for i in range(n_bands)])
    pc_image   = build_pc_image(alpha, pca.components_)
    strat_img  = build_strat_image(pc_image, thresholds)

    emb_cols  = [f"emb_{i}" for i in range(n_bands)]
    selectors = ["longitude", "latitude", "elevation", "slope", "aspect",
                 "kapos_class"] + emb_cols

    fc = composite.addBands(strat_img).stratifiedSample(
        numPoints=PTS_PER_STRAT,
        classBand="strat_id",
        region=WORLD,
        scale=SAMPLE_SCALE,
        projection="EPSG:4326",
        seed=42,
        geometries=True,
        dropNulls=True,
        tileScale=4,
    )
    fc = fc.map(lambda f: f.set({
        "longitude": f.geometry().coordinates().get(0),
        "latitude":  f.geometry().coordinates().get(1),
    }))
    task = export_to_drive(fc, "kapos_candidates_pca", "candidates_pca_raw", selectors)
    print(f"Candidates task ID: {task.id}")
    if wait:
        wait_for_task(task)


# ── Stage 3: k-means++ cosine clustering ─────────────────────────────────────
def kmeanspp_cosine(emb_n, terr_s, k, w_terrain=0.2, seed=42):
    """
    k-means++ selection using cosine dissimilarity over L2-normalised embeddings,
    with a small terrain Euclidean term as tiebreaker.

    Returns list of k selected row indices.
    """
    rng = np.random.default_rng(seed)
    n   = len(emb_n)
    # Start from the point nearest to the mean embedding direction
    start     = int(np.argmin(np.linalg.norm(emb_n - emb_n.mean(axis=0), axis=1)))
    selected  = [start]
    min_dists = np.full(n, np.inf)

    for _ in range(k - 1):
        last  = selected[-1]
        cos_d = 1.0 - emb_n @ emb_n[last]                            # (n,)
        ter_d = np.linalg.norm(terr_s - terr_s[last], axis=1)        # (n,)
        d     = cos_d + w_terrain * ter_d
        np.minimum(min_dists, d, out=min_dists)

        probs           = min_dists ** 2
        probs[selected] = 0.0
        probs          /= probs.sum()
        selected.append(int(rng.choice(n, p=probs)))

    return selected


def assign_strata(df):
    if NE_CACHE.exists():
        countries = gpd.read_file(NE_CACHE, engine="pyogrio")
    else:
        print("Downloading Natural Earth countries …")
        countries = gpd.read_file(NE_URL, engine="pyogrio")[["CONTINENT", "geometry"]]
        NE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        countries.to_file(NE_CACHE, driver="GPKG", engine="pyogrio")

    pts    = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude),
                               crs="EPSG:4326")
    joined = gpd.sjoin(pts, countries[["CONTINENT","geometry"]], how="left",
                       predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["CONTINENT"].map(CONTINENT_TO_STRATUM).fillna("Other")


def run_cluster():
    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(f"{CANDIDATES_CSV} not found — run --sample first.")

    df = pd.read_csv(CANDIDATES_CSV).dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"{len(df):,} candidates loaded · {len(EMB_COLS)} embedding dims")

    df["stratum"] = assign_strata(df)
    print(f"\nStratum counts:\n{df['stratum'].value_counts().to_string()}\n")

    # L2-normalise embeddings and standardise terrain once, reused per stratum
    emb   = df[EMB_COLS].values.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_n = emb / np.maximum(norms, 1e-8)
    terr  = StandardScaler().fit_transform(df[ANCILLARY_COLS].values.astype(np.float32))

    parts = []
    for stratum, quota in STRATA_QUOTAS.items():
        mask = df["stratum"].values == stratum
        idx  = np.where(mask)[0]
        print(f"  {stratum}: {len(idx):>8,} candidates → {quota} points", end="  ")
        if len(idx) < 2:
            print("skipped (too few)")
            continue
        sel_local = kmeanspp_cosine(emb_n[idx], terr[idx], k=min(quota, len(idx)),
                                    seed=42)
        sel_rows  = df.iloc[idx[sel_local]].copy()
        sel_rows["stratum"] = stratum
        parts.append(sel_rows)
        print(f"→ {len(sel_rows)} selected")

    selected = pd.concat(parts, ignore_index=True)
    selected["point_id"] = np.arange(len(selected))
    print(f"\nTotal selected: {len(selected)}")

    gdf = gpd.GeoDataFrame(
        selected,
        geometry=gpd.points_from_xy(selected.longitude, selected.latitude),
        crs="EPSG:4326",
    )
    OUTPUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON", engine="pyogrio")
    print(f"Saved → {OUTPUT_GEOJSON}")
    print(f"Elevation : {selected.elevation.min():.0f} – {selected.elevation.max():.0f} m")
    print(f"Lat range : {selected.latitude.min():.1f} – {selected.latitude.max():.1f}°")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot",   action="store_true", help="Stage 1: pilot export")
    parser.add_argument("--sample",  action="store_true", help="Stage 2: PCA + stratified export")
    parser.add_argument("--cluster", action="store_true", help="Stage 3: k-means++ cosine selection")
    parser.add_argument("--wait",    action="store_true", help="Block until GEE export completes")
    args = parser.parse_args()

    if args.pilot:
        run_pilot(wait=args.wait)
    elif args.sample:
        run_sample(wait=args.wait)
    elif args.cluster:
        run_cluster()
    else:
        parser.print_help()
