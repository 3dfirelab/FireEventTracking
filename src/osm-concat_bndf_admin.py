import geopandas as gpd
import pandas as pd
import glob
import os
import numpy as np
import pdb 
import matplotlib.pyplot as plt
from shapely import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from unidecode import unidecode

# ------------------------------------------------------------------
# Paths and constants
# ------------------------------------------------------------------
dir_bndf = "/data/shared/OSM_Boundaries/"
geojson_files = glob.glob(f"{dir_bndf}/*.geojson")

EQUAL_AREA_CRS = "ESRI:54034"   # World Cylindrical Equal Area
LEVELS_TO_TEST = [ 8, 9]
COVERAGE_THRESHOLD = 2

gdfs = []


def gc_to_poly(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = []
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                polys.append(g)
            elif g.geom_type == "MultiPolygon":
                polys.extend(g.geoms)
        if len(polys) == 1:
            return polys[0]
        if len(polys) > 1:
            return MultiPolygon(polys)
    return None


# ------------------------------------------------------------------
# Reference polygon selection
# ------------------------------------------------------------------
def get_reference_polygon(gdf):
    """
    Prefer admin_level=2 as country reference.
    Fallback: largest polygon among levels 3–5.
    Returns (geometry, admin_level)
    """
    gdf = gdf.dropna(subset=["admin_level"]).copy()

    admin_num = pd.to_numeric(gdf["admin_level"], errors="coerce")
    gdf = gdf.loc[admin_num.notna()].copy()
    gdf["admin_level"] = admin_num.loc[admin_num.notna()].astype(int)

    # Preferred: country boundary
    g2 = gdf[gdf["admin_level"] == 2]
    if not g2.empty:
        return g2.union_all(), 2

    # Fallback: largest polygon among regional levels
    cand = gdf[gdf["admin_level"].isin([3, 4, 5])]
    if cand.empty:
        return gdf.union_all(), None

    cand_eq = cand.to_crs(EQUAL_AREA_CRS)
    idx = cand_eq.geometry.area.idxmax()
    return cand.loc[idx, "geometry"], int(cand.loc[idx, "admin_level"])


# ------------------------------------------------------------------
# Coverage computation
# ------------------------------------------------------------------
def coverage_fraction(gdf, ref_geom, level):
    gl = gdf[gdf["admin_level"] == level]
    if gl.empty:
        return 0.0

    union_geom = gl.union_all()

    ref = gpd.GeoSeries([ref_geom], crs=gdf.crs).to_crs(EQUAL_AREA_CRS).iloc[0]
    uni = gpd.GeoSeries([union_geom], crs=gdf.crs).to_crs(EQUAL_AREA_CRS).iloc[0]

    inter = uni.intersection(ref)
    return inter.area / ref.area if ref.area > 0 else 0.0


# ------------------------------------------------------------------
# Incremental gap filling with lower admin levels
# ------------------------------------------------------------------
def incremental_fill_coverage(gdf, ref_geom, start_level, gdf_sel, min_level=3):
    """
    Incrementally fill uncovered area using lower admin levels.
    Returns (updated_gdf_sel, final_coverage, levels_used)
    """
    ref_eq = gpd.GeoSeries([ref_geom], crs=gdf.crs).to_crs(EQUAL_AREA_CRS).iloc[0]

    used_levels = [start_level] if not gdf_sel.empty else []

    if gdf_sel.empty:
        current_union_eq = None
        frac = 0.0
    else:
        current_union_eq = (
            gpd.GeoSeries([gdf_sel.union_all()], crs=gdf.crs)
            .to_crs(EQUAL_AREA_CRS)
            .iloc[0]
        )
        frac = current_union_eq.area / ref_eq.area if ref_eq.area > 0 else 0.0

    for lvl in range(10, min_level, -1):

        if lvl == start_level:
            continue
        if frac >= COVERAGE_THRESHOLD:
            break

        gl = gdf[gdf["admin_level"] == lvl]
        if gl.empty:
            continue

        missing_eq = ref_eq if current_union_eq is None else ref_eq.difference(current_union_eq)
        if missing_eq.is_empty:
            frac = 1.0
            break

        gl_eq = gl.to_crs(EQUAL_AREA_CRS)
        clipped_eq = gl_eq.geometry.intersection(missing_eq)
        idx_add = clipped_eq[~clipped_eq.is_empty].index

        if len(idx_add) == 0:
            continue

        if current_union_eq is None:
            current_union_eq = clipped_eq.loc[idx_add].union_all()
        else:
            current_union_eq = current_union_eq.union(clipped_eq.loc[idx_add].union_all())

        missing_orig = (
            gpd.GeoSeries([missing_eq], crs=EQUAL_AREA_CRS)
            .to_crs(gdf.crs)
            .iloc[0]
        )

        try:
            missing_orig = make_valid(missing_orig)
            gl_geom = gl.loc[idx_add].geometry.apply(make_valid)
        except Exception:
            missing_orig = missing_orig.buffer(0)
            gl_geom = gl.loc[idx_add].geometry.buffer(0)

        clipped_orig = gl_geom.intersection(missing_orig)

        new_rows = gl.loc[idx_add].copy()
        new_rows.geometry = clipped_orig
        new_rows["added_from_level"] = lvl

        gdf_sel = pd.concat([gdf_sel, new_rows], ignore_index=True)

        used_levels.append(lvl)
        frac = current_union_eq.area / ref_eq.area if ref_eq.area > 0 else 0.0
       
        gdf_sel.loc[gdf_sel["name"].isna(), "name"] = "unknown"

        try: 
            assert not gdf_sel["name"].isna().any()
        except:
            pdb.set_trace()

    return gdf_sel, frac, used_levels



# ------------------------------------------------------------------
# Main processing loop
# ------------------------------------------------------------------
for f in sorted(geojson_files):
    #if 'austria'  not in f: continue
    print(os.path.basename(f), end=" ")

    gdf_allLevel = gpd.read_file(f)

    if "admin_level" not in gdf_allLevel.columns:
        print("no admin_level")
        continue

    # Clean admin_level
    admin_num = pd.to_numeric(gdf_allLevel["admin_level"], errors="coerce")
    gdf_allLevel = gdf_allLevel.loc[admin_num.notna()].copy()
    gdf_allLevel["admin_level"] = admin_num.loc[admin_num.notna()].astype(int)

    # Reference country geometry
    ref_geom, ref_admin_level = get_reference_polygon(gdf_allLevel)

    # Keep only candidate admin levels
    gdf = gdf_allLevel.copy()
    gdf = gdf[gdf["admin_level"].isin(LEVELS_TO_TEST)].copy()
    if gdf.empty:
        print("no candidate levels")
        continue

    # Compute coverage per level
    frac_arr = np.array([
        coverage_fraction(gdf, ref_geom, lvl)
        for lvl in LEVELS_TO_TEST
    ])

    best_level = LEVELS_TO_TEST[frac_arr.argmax()]
    best_frac = frac_arr.max()

    if best_frac >= COVERAGE_THRESHOLD:
        gdf_sel = gdf[gdf["admin_level"] == best_level].copy()
        used_levels = [best_level]
        final_frac = best_frac
    else:
        gdf_sel = gdf[gdf["admin_level"] == best_level].copy()
        gdf_sel, final_frac, used_levels = incremental_fill_coverage(
            gdf_allLevel, ref_geom, best_level, gdf_sel, min_level=3
        )

    print(best_level, ref_admin_level, final_frac, used_levels)
    
    # Track source
    gdf_sel["source_pbf"] = os.path.basename(f).replace(".geojson", "")

    gdf_sel["name_en_clean"] = gdf_sel["name:en"].fillna(gdf_sel["name"]).fillna('unknown').apply(lambda x: unidecode(x) if x else x).replace(' ','')
    
    try: 
        assert not gdf_sel["name_en_clean"].isna().any()
    except: 
        pdb.set_trace()

    #deal with potential duplicate name
    mask = gdf_sel["name_en_clean"].duplicated(keep=False)
    gdf_sel.loc[mask, "name_en_clean"] = (
        gdf_sel.loc[mask]
        .groupby("name_en_clean")
        .cumcount()
        .astype(str)
        .radd("_")
        .radd(gdf_sel.loc[mask, "name_en_clean"])
    )

    # Keep useful columns
    keep_cols = ["name_en_clean", "admin_level", "boundary", "source_pbf", "geometry"]
    keep_cols = [c for c in keep_cols if c in gdf_sel.columns]
    gdf_sel = gdf_sel[keep_cols]

    gdf_sel["geometry"] = gdf_sel.geometry.apply(gc_to_poly)
    
    gdfs.append(gdf_sel)


# ------------------------------------------------------------------
# Concatenate all countries and write output
# ------------------------------------------------------------------
gdf_all = gpd.GeoDataFrame(
    pd.concat(gdfs, ignore_index=True),
    crs="EPSG:4326"
)

print("Total features:", len(gdf_all))

out_file = f"{dir_bndf}/osm_MED_admin_muni_boundaries.gpkg"
gdf_all.to_file(out_file, driver="GPKG")
