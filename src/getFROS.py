import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString, Point
from shapely.ops import unary_union
from shapely.validation import make_valid
import pdb 
import matplotlib.pyplot as plt
# -------------------------------------------------------
# Function definitions
# -------------------------------------------------------

def get_coords(geom):
    """Return Nx2 array of coordinates for Polygon-like geometries."""
    coords = []
    for poly in iter_polygons(geom):
        coords.extend(poly.exterior.coords)
    return np.array(coords) if coords else np.empty((0, 2))

def iter_polygons(geom):
    """Yield individual Polygon parts from Polygon-like geometries."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            if not part.is_empty:
                yield part
    elif isinstance(geom, GeometryCollection):
        for part in geom.geoms:
            yield from iter_polygons(part)

def _is_same_geometry(a, b, tol=1e-6):
    if a is b:
        return True
    if a is None or b is None:
        return False
    try:
        if a.equals(b):
            return True
    except Exception:
        pass
    try:
        return a.equals_exact(b, tol)
    except Exception:
        return False

def _intersection_excludes_endpoints(segment, obstacle_edge, tol=0.5):
    """
    Return True if an obstacle edge intersects the segment away from its endpoints.
    tol controls the exclusion radius (in CRS units) around each endpoint.
    """
    if obstacle_edge.is_empty or not obstacle_edge.intersects(segment):
        return False

    intersection = obstacle_edge.intersection(segment)
    if intersection.is_empty:
        return False

    endpoints_buffer = unary_union(
        [Point(segment.coords[0]).buffer(tol), Point(segment.coords[-1]).buffer(tol)]
    )
    cleaned = intersection.difference(endpoints_buffer)

    if cleaned.is_empty:
        return False

    if cleaned.geom_type in {"Point", "MultiPoint"}:
        return True
    if cleaned.geom_type in {"LineString", "LinearRing", "MultiLineString"}:
        return cleaned.length > tol
    if cleaned.geom_type in {"Polygon", "MultiPolygon"}:
        return True
    if cleaned.geom_type == "GeometryCollection":
        return not cleaned.is_empty

    return True

def compute_max_distance(outer, inner, obstacles=None):
    """
    Compute the largest distance between vertices of outer and inner polygons,
    ignoring candidate segments that intersect the boundary of any obstacle.
    """
    outer_pts = get_coords(outer)
    inner_pts = get_coords(inner)
    if len(outer_pts) == 0 or len(inner_pts) == 0:
        return np.nan, None, None

    obstacle_edges = []
    for geom in obstacles or []:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, (Polygon, MultiPolygon)):
            boundary = geom.boundary
            if boundary.is_empty:
                continue
            obstacle_edges.append(boundary)
        elif isinstance(geom, GeometryCollection):
            for part in geom.geoms:
                if isinstance(part, (Polygon, MultiPolygon)):
                    boundary = part.boundary
                    if not boundary.is_empty:
                        obstacle_edges.append(boundary)
                elif not part.is_empty:
                    obstacle_edges.append(part)
        else:
            obstacle_edges.append(geom)

    best_dist = -np.inf
    best_outer = None
    best_inner = None

    for outer_pt in outer_pts:
        #fig, ax = plt.subplots()
        for edge in obstacle_edges:
            geoms = edge.geoms if hasattr(edge, "geoms") else [edge]
            for geom in geoms:
                if hasattr(geom, "xy"):
                    x, y = geom.xy
                    #ax.plot(x, y, color='k', linewidth=1)
        for inner_pt in inner_pts:
            segment = LineString([(outer_pt[0], outer_pt[1]), (inner_pt[0], inner_pt[1])])
            if not outer.contains(segment):
                continue
            if inner.contains(segment):
                continue
            if any(_intersection_excludes_endpoints(segment, obs_edge) for obs_edge in obstacle_edges):
                continue

            x, y = segment.xy
            #ax.plot(x, y, color='red', linewidth=2)
            #ax.scatter(x, y, color='black')  # show endpoints
            

            dist = np.linalg.norm(outer_pt - inner_pt)
            
            if dist > best_dist:
                best_dist = dist
                best_outer = outer_pt
                best_inner = inner_pt
        #ax.set_aspect('equal')
        #plt.xlabel("X [m]")
        #plt.ylabel("Y [m]")
        #plt.show()

    if best_outer is None or best_inner is None:
        return np.nan, None, None

    return best_dist, best_outer, best_inner

# -------------------------------------------------------
# Main processing function
# -------------------------------------------------------

def compute_polygon_velocity(gdf):
    """
    For each older polygon that contains a younger one:
    - Find nested relationships
    - Compute max vertex distance (non-intersecting)
    - Compute propagation velocity
    """
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    #gdf = gdf.sort_values("timestamp").reset_index(drop=True)
    gdf["source_id"] = gdf.index
    gdf = gdf.explode(ignore_index=True)
    gdf["component_id"] = gdf.groupby("source_id").cumcount()

    # Ensure projected CRS for distance calculations
    if gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError("GeoDataFrame must have a projected CRS (units in meters).")

    results = []

    for i, row_i in gdf.iterrows():
        outer = row_i.geometry
        if not isinstance(outer, Polygon) or outer.is_empty:
            continue

        for j, row_j in gdf.iloc[i + 1:].iterrows():
            print(j)
            inner = row_j.geometry
            if not isinstance(inner, Polygon) : #or inner.is_empty:
                continue

            if False:
                fig, ax = plt.subplots()
                ax.plot(*inner.exterior.xy, color='orange', linewidth=2)
                ax.plot(*outer.exterior.xy, color='red', linewidth=2)
                ax.set_aspect('equal')
                plt.show()
                pdb.set_trace()
            # Test containment
            if not outer.is_valid or not inner.is_valid:
                continue
            if not outer.contains(inner):
                continue


            # Compute maximum vertex distance
            obstacle_geoms = gdf.geometry.tolist()
            max_dist, outer_pt, inner_pt = compute_max_distance(outer, inner, obstacle_geoms)
            print(max_dist)
            if max_dist is None or np.isnan(max_dist):
                continue

            # Time difference in hours
            t1, t2 = row_i.timestamp,  row_j.timestamp
            dt_hours = (t1 - t2).total_seconds() 
            if dt_hours <= 0:
                continue

            velocity = max_dist / dt_hours  # meters/hour
            dx = inner_pt[0] - outer_pt[0]
            dy = inner_pt[1] - outer_pt[1]
            vector_line = LineString([(outer_pt[0], outer_pt[1]), (inner_pt[0], inner_pt[1])])

            results.append({
                "outer_id": int(row_i.source_id),
                "inner_id": int(row_j.source_id),
                "outer_component": int(row_i.component_id),
                "inner_component": int(row_j.component_id),
                "outer_time": t1,
                "inner_time": t2,
                "max_dist_m": max_dist,
                "velocity_m_per_s": velocity,
                "vector_dx": dx,
                "vector_dy": dy,
                "outer_pt_x": outer_pt[0],
                "outer_pt_y": outer_pt[1],
                "inner_pt_x": inner_pt[0],
                "inner_pt_y": inner_pt[1],
                "geometry": vector_line,
            })

            break

    gdf_results = gpd.GeoDataFrame(results, geometry='geometry', crs=gdf.crs)

    return gdf_results


if __name__ == '__main__':
# -------------------------------------------------------
# Example usage
# -------------------------------------------------------

# Assuming your GeoDataFrame is named gdf_akli3
# and CRS is projected (e.g., EPSG:32630)

    gdf_akli3 = gpd.read_file('gdf_akli3.gpkg')
    gdf_akli3 = gdf_akli3.sort_values("timestamp", ascending=False).reset_index(drop=True)

# gdf_akli3 = gdf_akli3.to_crs("EPSG:32630")  # only if necessary
    df_velocity = compute_polygon_velocity(gdf_akli3)

    df_velocity.to_file('./gdf_akli3_fros.gpkg')
    print(df_velocity)
    ax = plt.subplot(111)
    gdf_akli3.plot(ax=ax,facecolor='none')
    df_velocity.plot(ax=ax)
    plt.show()
