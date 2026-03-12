import argparse
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Polygon
import fireEventTracking as fet
import matplotlib.colors as mcolors
import cartopy.crs as ccrs


def h3_cell(lat: float, lon: float, res: int) -> str | None:
    if hasattr(h3, "latlng_to_cell"):
        try:
            return h3.latlng_to_cell(lat, lon, res)  # type: ignore[attr-defined]
        except Exception:
            return None
    try:
        return h3.geo_to_h3(lat, lon, res)  # type: ignore[attr-defined]
    except Exception:
        return None


def h3_boundary(cell: str):
    if hasattr(h3, "cell_to_boundary"):
        return h3.cell_to_boundary(cell)  # type: ignore[attr-defined]
    return h3.h3_to_geo_boundary(cell)  # type: ignore[attr-defined]


def build_hex_polygon(cell: str) -> Polygon:
    boundary = h3_boundary(cell)
    ring = [[lng, lat] for lat, lng in boundary]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return Polygon(ring)


def load_stats_gdf(input_name: str, sensor_name: str) -> gpd.GeoDataFrame:
    params = fet.init(input_name, sensor_name)
    start = datetime.strptime(params["event"]["start_time"], "%Y-%m-%d_%H%M").replace(
        tzinfo=timezone.utc
    )
    end = datetime.strptime(params["event"]["end_time_hard"], "%Y-%m-%d_%H%M").replace(
        tzinfo=timezone.utc
    )
    stats_path = Path(
        f"{params['event']['dir_data']}/Stats/"
        f"{params['general']['domainName']}-gdf_{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}.geojson"
    )
    return gpd.read_file(stats_path)

def filter_short_point_events(gdf: gpd.GeoDataFrame, max_minutes: int = 10) -> gpd.GeoDataFrame:
    if "time_start" not in gdf.columns or "time_end" not in gdf.columns:
        return gdf
    gdf = gdf.copy()
    gdf["time_start"] = pd.to_datetime(gdf["time_start"], errors="coerce")
    gdf["time_end"] = pd.to_datetime(gdf["time_end"], errors="coerce")
    is_point = gdf.geometry.geom_type == "Point"
    dt_min = (gdf["time_end"] - gdf["time_start"]).dt.total_seconds() / 60.0
    mask_drop = is_point & dt_min.notna() & (dt_min <= max_minutes)
    return gdf.loc[~mask_drop].copy()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build H3 map of mean FRE per event (resolution 4)."
    )
    parser.add_argument("--inputName", required=True, help="configuration input name")
    parser.add_argument("--sensorName", required=True, help="sensor name")
    parser.add_argument("--h3-res", type=int, default=4, help="H3 resolution (default 4)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GeoJSON (default: Stats/h3_mean_fre_res4.geojson)",
    )
    args = parser.parse_args()

    gdf = load_stats_gdf(args.inputName, args.sensorName)
    if "fre" not in gdf.columns:
        raise SystemExit("Missing column 'fre' in stats GeoJSON.")

    gdf = filter_short_point_events(gdf, max_minutes=10)
    gdf = gdf.to_crs(4326)
    gdf["centroid"] = gdf.geometry.centroid
    gdf["lon"] = gdf["centroid"].x
    gdf["lat"] = gdf["centroid"].y

    gdf["h3"] = [
        h3_cell(lat, lon, args.h3_res) for lat, lon in zip(gdf["lat"], gdf["lon"])
    ]
    gdf = gdf[gdf["h3"].notna()].copy()

    agg = gdf.groupby("h3", as_index=False)["fre"].mean().rename(
        columns={"fre": "fre_mean"}
    )
    agg["res"] = args.h3_res

    gdf["time_start"] = pd.to_datetime(gdf.get("time_start"), errors="coerce")
    gdf["time_end"] = pd.to_datetime(gdf.get("time_end"), errors="coerce")
    gdf["duration_h"] = (
        (gdf["time_end"] - gdf["time_start"]).dt.total_seconds() / 3600.0
    )
    gdf = gdf[gdf["duration_h"].notna()].copy()
    dur = gdf.groupby("h3", as_index=False)["duration_h"].mean().rename(
        columns={"duration_h": "duration_mean_h"}
    )
    agg = agg.merge(dur, on="h3", how="left")

    features = []
    for _, row in agg.iterrows():
        geom = build_hex_polygon(row["h3"])
        features.append(
            {
                "h3": row["h3"],
                "fre_mean": row["fre_mean"],
                "res": row["res"],
                "geometry": geom,
            }
        )

    out_gdf = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")

    params = fet.init(args.inputName, args.sensorName)
    default_out = Path(params["event"]["dir_data"]) / "Stats" / f"h3_mean_fre_res{args.h3_res}.geojson"
    out_path = args.out or default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(out_path, driver="GeoJSON")

    print(f"Wrote {len(out_gdf)} H3 cells to {out_path}")

    # Integrated FRE time series (domain-wide)
    gdf = gdf[gdf["time_start"].notna()].copy()
    gdf["hour"] = gdf["time_start"].dt.floor("H")
    fre_hourly = gdf.groupby("hour", as_index=False)["fre"].sum().sort_values("hour")
    fre_daily["fre_cum"] = fre_daily["fre"].cumsum()

    # Plot H3 map with lat/lon + colorbar and time series below
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_map.coastlines(linewidth=0.4)
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.2, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    positive = out_gdf["fre_mean"][out_gdf["fre_mean"] > 0]
    vmin = positive.min() if not positive.empty else 1e-6
    vmax = out_gdf["fre_mean"].max() if len(out_gdf) else vmin
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, 1e-6))

    out_gdf.plot(
        column="fre_mean",
        ax=ax_map,
        transform=ccrs.PlateCarree(),
        norm=norm,
        cmap="viridis",
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_map, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label("Mean FRE per fire event (MJ)")

    ax_ts = fig.add_subplot(gs[1, 0])
    ax_ts.plot(fre_hourly["hour"], fre_hourly["fre"], color="black")
    ax_ts.set_ylabel("Hourly FRE (MJ)")
    ax_ts.set_xlabel("Time (hour)")
    ax_ts.set_title("Hourly FRE time series")

    fig.tight_layout()
    plt.show()

    # H3 map of mean fire duration
    fig2 = plt.figure(figsize=(10, 6))
    ax_dur = fig2.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax_dur.coastlines(linewidth=0.4)
    gl2 = ax_dur.gridlines(draw_labels=True, linewidth=0.2, linestyle="--", alpha=0.5)
    gl2.top_labels = False
    gl2.right_labels = False

    positive_dur = out_gdf["duration_mean_h"][out_gdf["duration_mean_h"] > 0]
    vmin_d = positive_dur.min() if not positive_dur.empty else 1e-3
    vmax_d = out_gdf["duration_mean_h"].max() if len(out_gdf) else vmin_d
    norm_d = mcolors.LogNorm(vmin=max(vmin_d, 1e-3), vmax=max(vmax_d, 1e-3))

    out_gdf.plot(
        column="duration_mean_h",
        ax=ax_dur,
        transform=ccrs.PlateCarree(),
        norm=norm_d,
        cmap="magma",
    )
    sm_d = plt.cm.ScalarMappable(cmap="magma", norm=norm_d)
    sm_d.set_array([])
    cbar_d = fig2.colorbar(sm_d, ax=ax_dur, orientation="vertical", fraction=0.03, pad=0.02)
    cbar_d.set_label("Mean fire duration (h)")

    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
