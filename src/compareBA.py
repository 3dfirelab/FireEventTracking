import geopandas as gpd 
import matplotlib.pyplot as plt 
from datetime import datetime, timezone, timedelta
import pandas as pd 
import pdb 
import numpy as np
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pyproj import Geod
import matplotlib.patches as mpatches

#homebrwed
import fireEventTracking as fet


def _ensure_metric_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf
    if gdf.crs.is_geographic:
        return gdf.to_crs(3857)
    return gdf


def match_spatiotemporal_events(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
    time_col_a: str,
    time_col_b: str,
    max_days: float = 10.0,
    max_km: float = 50.0,
    min_iou: float = 0.01,
    w_iou: float = 0.4,
    w_dist: float = 0.3,
    w_time: float = 0.3,
) -> pd.DataFrame:
    a = gdf_a.copy()
    b = gdf_b.copy()

    a[time_col_a] = pd.to_datetime(a[time_col_a], errors="coerce")
    b[time_col_b] = pd.to_datetime(b[time_col_b], errors="coerce")
    a = a[a[time_col_a].notna()].copy()
    b = b[b[time_col_b].notna()].copy()

    a = a.reset_index().rename(columns={"index": "_a_id"})
    b = b.reset_index().rename(columns={"index": "_b_id"})

    a_metric = _ensure_metric_crs(a)
    b_metric = _ensure_metric_crs(b)

    max_dist_m = max_km * 1000.0

    a_buf = a_metric[["_a_id", time_col_a, "geometry"]].copy()
    a_buf["geometry"] = a_buf.geometry.buffer(max_dist_m)

    candidates = gpd.sjoin(
        a_buf,
        b_metric[["_b_id", time_col_b, "geometry"]],
        how="left",
        predicate="intersects",
    )

    candidates = candidates.dropna(subset=["_b_id"]).copy()
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "_a_id",
                "_b_id",
                "score",
                "iou",
                "dist_m",
                "dt_days",
            ]
        )

    candidates["_b_id"] = candidates["_b_id"].astype(int)

    candidates = candidates.merge(
        a_metric[["_a_id", "geometry"]].rename(columns={"geometry": "geom_a"}),
        on="_a_id",
        how="left",
    )
    candidates = candidates.merge(
        b_metric[["_b_id", "geometry"]].rename(columns={"geometry": "geom_b"}),
        on="_b_id",
        how="left",
    )

    inter_area = candidates.geom_a.intersection(candidates.geom_b).area
    union_area = candidates.geom_a.union(candidates.geom_b).area
    candidates["iou"] = inter_area.where(union_area > 0, 0) / union_area.where(union_area > 0, 1)

    candidates["dist_m"] = candidates.geom_a.centroid.distance(candidates.geom_b.centroid)
    #candidates["dist_m"] = candidates.geom_a.distance(candidates.geom_b)
    candidates = candidates[candidates["dist_m"] <= max_dist_m].copy()

    candidates["dt_days"] = (
        (candidates[time_col_a] - candidates[time_col_b]).abs().dt.total_seconds()
        / 86400.0
    )

    candidates = candidates[candidates["dt_days"] <= max_days].copy()
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "_a_id",
                "_b_id",
                "score",
                "iou",
                "dist_m",
                "dt_days",
            ]
        )

    candidates = candidates[candidates["iou"] >= min_iou].copy()
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "_a_id",
                "_b_id",
                "score",
                "iou",
                "dist_m",
                "dt_days",
            ]
        )

    penalty_km = 5.0
    penalty_m = penalty_km * 1000.0

    dist_score = 1.0 - (candidates["dist_m"].clip(upper=penalty_m) / penalty_m)
    dist_score = dist_score.where(candidates["dist_m"] > penalty_m, 0.)

    time_score = 1.0 - (candidates["dt_days"].clip(upper=max_days) / max_days)
    candidates["score"] = (w_iou * candidates["iou"]) + (w_dist * dist_score) + (w_time * time_score)

    #dist_score = 1.0 - (candidates["dist_m"].clip(upper=max_dist_m) / max_dist_m)
    #time_score = 1.0 - (candidates["dt_days"].clip(upper=max_days) / max_days)
    #candidates["score"] = (w_iou * candidates["iou"]) + (w_dist * dist_score) + (w_time * time_score)

    candidates = candidates.sort_values(
        by=["score", "iou", "dist_m"],
        ascending=[False, False, True],
    )

    best = candidates.groupby("_a_id").head(1).reset_index(drop=True)
    return best[["_a_id", "_b_id", "score", "iou", "dist_m", "dt_days"]]


def add_scalebar(ax, length_km=200, location=(-9.5, 35.5), linewidth=3):
    """
    Add a horizontal scale bar to a Cartopy axis.
    
    length_km : length of scale in km
    location  : (lon, lat) of left end of scale bar
    """
    
    geod = Geod(ellps="WGS84")
    lon0, lat0 = location

    # Compute endpoint at given distance eastward (azimuth=90°)
    lon1, lat1, _ = geod.fwd(lon0, lat0, 90, length_km * 1000)

    # Draw line
    ax.plot([lon0, lon1], [lat0, lat1],
            transform=ccrs.PlateCarree(),
            color='k', linewidth=linewidth)

    # Add label centered
    ax.text((lon0 + lon1)/2, lat0,
            f"{length_km} km",
            transform=ccrs.PlateCarree(),
            horizontalalignment='center',
            verticalalignment='bottom')
###############################
if __name__ == '__main__':
###############################
    
    bndf = gpd.read_file('/data/shared/Boundaries/NaturalEarth_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
    bndf = bndf[bndf['SOV_A3'].isin(['PRT', 'ESP', 'FRA'])]
    #bndf = bndf[bndf['SOV_A3'].isin(['GRC'])]
    

    #load FCI data
    inputName = 'MED3'
    sensorName = 'FCI'
    params = fet.init(inputName,sensorName)
    start = datetime.strptime(params['event']['start_time'], '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)
    end   =  datetime.strptime(params['event']['end_time_hard'], '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)


    gdf_fci = gpd.read_file("{:s}/Stats/{:s}-gdf_{:s}_{:s}.geojson".format(
                            params['event']['dir_data'],params['general']['domainName'], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")) )
    bndf = bndf.to_crs(gdf_fci.crs)
    #gdf_fci = gpd.clip(gdf_fci, bndf) 

    gdf_fci['area_m2'] = gdf_fci.geometry.area
    gdf_fci_large = gdf_fci[gdf_fci["area_m2"] >= 500.e4 ] #100.e4] # > 10ha
    #gdf_fci_large = gdf_fci.copy()

    #load EFFIS data
    gdf_effis = gpd.read_file('/data/shared/BurntArea/EFFIS/modis.ba.poly.shp').to_crs(gdf_fci_large.crs)
    # Ensure datetime dtype (safe check)
    gdf_effis["FIREDATE"] = pd.to_datetime(
                                        gdf_effis["FIREDATE"],
                                        format="ISO8601",
                                        errors="coerce"
                                    ).dt.tz_localize("UTC")
    # Filter year == 2025
    gdf_effis = gdf_effis[gdf_effis["FIREDATE"].dt.year == 2025]
    #gdf_effis = gpd.clip(gdf_effis, bndf) 
   
    print('run match_spatiotemporal_events')
    matches = match_spatiotemporal_events(
        gdf_fci_large,
        gdf_effis,
        time_col_a="time_start",
        time_col_b="FIREDATE",
        max_days=2,
        max_km=10,
        min_iou=0.01,
    )

    gdf_fci_large = gdf_fci_large.reset_index().rename(columns={"index": "_a_id"})
    gdf_fci_large = (
        gdf_fci_large.merge(
            matches.rename(
                columns={
                    "_b_id": "match_effis_id",
                    "score": "match_score",
                    "iou": "match_iou",
                    "dist_m": "match_dist_m",
                    "dt_days": "match_dt_days",
                }
            ),
            on="_a_id",
            how="left",
        )
        .set_index("_a_id")
    )

    print("Matched FCI events:", gdf_fci_large["match_effis_id"].notna().sum())
    print("Unmatched FCI events:", gdf_fci_large["match_effis_id"].isna().sum())
    
    matched_effis_ids = gdf_fci_large.loc[gdf_fci_large["match_effis_id"].notna(), "match_effis_id"].astype(int)
    effis_unmatched = gdf_effis.loc[~gdf_effis.index.isin(matched_effis_ids)]
    print("Unmatched EFFIS events:", effis_unmatched.shape[0])
    
    # Area correlation for matched events
    fci_m  = gdf_fci_large[ gdf_fci_large["match_effis_id"].notna()].copy()
    fci_nm = gdf_fci_large[~gdf_fci_large["match_effis_id"].notna()].copy()
    
    # Ensure metric CRS for area
    fci_m = _ensure_metric_crs(fci_m)
    fci_nm = _ensure_metric_crs(fci_nm)
    effis_m = _ensure_metric_crs(gdf_effis)

    ids = fci_m["match_effis_id"].astype(int).values
    effis_m = effis_m.loc[ids].reset_index(drop=True)
    fci_m = fci_m.reset_index(drop=True)
    fci_nm = fci_nm.reset_index(drop=True)

    fci_m["area_ha"] = fci_m.geometry.area / 1.e4
    fci_nm["area_ha"] = fci_nm.geometry.area / 1.e4
    effis_m["area_ha"] = effis_m.geometry.area / 1.e4



    x = fci_m["area_ha"].values
    y = effis_m["area_ha"].values

    # Linear fit with covariance
    (m, p), cov = np.polyfit(x, y, 1, cov=True)
    m_err = np.sqrt(cov[0, 0])
    p_err = np.sqrt(cov[1, 1])

    print(f"Fit: y = m x + p")
    print(f"m = {m:.4f} ± {m_err:.4f}")
    print(f"p = {p:.4f} ± {p_err:.4f}")
    corr = fci_m["area_ha"].corr(effis_m["area_ha"])
    print("Area correlation (FCI vs EFFIS):", corr)

    # Choose total figure height
    H = 8  # inches

    # Width = height of second subplot = 75% of total height
    W = 0.75 * H

    fig = plt.figure(figsize=(W, H))
    gs = GridSpec(2, 1, height_ratios=[1, 2], figure=fig)

    # ─────────────────────────────
    # Top plot (25%)
    # ─────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    ax1.scatter(x, y, s=8, alpha=0.6, label="matches")

    xx = np.linspace(x.min(), x.max(), 200)
    yy = m * xx + p
    ax1.plot(xx, yy, color="red", label=f"y = [{m:.4f} ± {m_err:.4f}] x + [{p:.4f} ± {p_err:.4f}]")

    ax1.set_xlabel("FCI area (ha)")
    ax1.set_ylabel("EFFIS area (ha)")
    ax1.set_title(
        f"Area fit EFFIS vs FCI correlation: {corr:.3f}\n"
        f"Entire Mediterranean Basin & FCI Area > 500 ha"
    )

    ax1.legend()


    # ─────────────────────────────
    # Bottom subplot with Cartopy
    # ─────────────────────────────
    ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())

    # Set extent (lon_min, lon_max, lat_min, lat_max)
    ax2.set_extent([-10, 4, 39, 44], crs=ccrs.PlateCarree())

    # Add coastlines and borders
    ax2.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
    ax2.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5)

    # Optional: land background
    ax2.add_feature(cfeature.LAND, facecolor='lightgrey', alpha=0.2)

    # Plot GeoDataFrames (already in 4326 or reproject on the fly)
    fci_m.to_crs(4326).plot(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        color='red'
    )

    fci_nm.to_crs(4326).plot(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        color='pink'
    )

    effis_m.to_crs(4326).plot(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        color='k',
        alpha=0.5
    )

    ax2.set_aspect('auto')  # Cartopy manages projection aspect

    # ─────────────────────────────
    # Manual legend
    # ─────────────────────────────
    legend_handles = [
        mpatches.Patch(color='red', label='FCI matched'),
        mpatches.Patch(color='pink', label='FCI non-matched'),
        mpatches.Patch(color='black', label='EFFIS', alpha=0.5),
    ]

    ax2.legend(
        handles=legend_handles,
        loc='center right',
        frameon=True
    )
    
    # Add gridlines
    gl = ax2.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )

    # Show labels only on left and bottom
    gl.top_labels = False
    gl.right_labels = False

    # Format labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Optional: control tick spacing
    gl.xlocator = mticker.FixedLocator(range(-10, 5, 2))
    gl.ylocator = mticker.FixedLocator(range(35, 45, 2))

    add_scalebar(ax2, length_km=200, location=(1, 39.2))

    plt.tight_layout()
    
    fig.savefig('correlation_area_fci_effis.png',dpi=300)
    plt.close(fig)

