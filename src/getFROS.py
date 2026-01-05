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
def mean_vector_line(lines):
    """Compute a mean LineString positioned in space from a list of LineStrings."""
    starts = []
    ends = []

    for line in lines:
        if line.is_empty or len(line.coords) < 2:
            continue
        x0, y0 = line.coords[0]
        x1, y1 = line.coords[-1]
        starts.append((x0, y0))
        ends.append((x1, y1))

    if not starts or not ends:
        return None

    # mean start and end coordinates
    start_mean = np.mean(starts, axis=0)
    end_mean = np.mean(ends, axis=0)

    # construct the mean line in actual space
    mean_line = LineString([tuple(start_mean), tuple(end_mean)])
    return mean_line


def densify_geometry(geom, n_points_per_segment=5):
    """Densify a LineString or Polygon by adding evenly spaced vertices along its segments."""
    if geom.is_empty:
        return geom

    if isinstance(geom, Polygon):
        line = geom.exterior
        is_polygon = True
    elif isinstance(geom, LineString):
        line = geom
        is_polygon = False
    else:
        return geom  # ignore other geometry types

    coords = list(line.coords)
    new_coords = [coords[0]]

    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        #ax.scatter(p1[0],p1[1])
        segment = LineString([p1, p2])
        new_coords.append(p1)
        # interpolate intermediate points (excluding the start)
        for j in range(1, n_points_per_segment):
            frac = j / n_points_per_segment
            new_point = segment.interpolate(frac, normalized=True)
            new_coords.append(new_point.coords[0])
            #ax.scatter(new_coords[-1][0],new_coords[-1][1])

    if is_polygon:
        # Ensure closure
        if new_coords[0] != new_coords[-1]:
            new_coords.append(new_coords[0])
        return Polygon(new_coords)
    else:
        return LineString(new_coords)


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

def _intersection_excludes_endpoints(segment, obstacle_edge, tol=0.001):
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

def compute_max_distance(outer, inner, dt, obstacles=None):
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

    best_dist  = []
    best_outer = []
    best_inner = []
    
    #fig, ax = plt.subplots()
    #gpd.GeoDataFrame(geometry=[outer]).plot(facecolor='none', edgecolor='orange',linewidth=3,ax=ax)
    #gpd.GeoDataFrame(geometry=[inner]).plot(facecolor='none', edgecolor='blue',linewidth=3,ax=ax)

    for outer_pt in outer_pts:
        for edge in obstacle_edges:
            geoms = edge.geoms if hasattr(edge, "geoms") else [edge]
            for geom in geoms:
                if hasattr(geom, "xy"):
                    x, y = geom.xy
                    #ax.plot(x, y, color='k', linewidth=4, linestyle=':')
        for inner_pt in inner_pts:
            segment = LineString([(outer_pt[0], outer_pt[1]), (inner_pt[0], inner_pt[1])])
            if not outer.contains(segment):
                continue
            if inner.contains(segment):
                continue
            if any(_intersection_excludes_endpoints(segment, obs_edge ) for obs_edge in obstacle_edges):
                continue

            x, y = segment.xy
            #ax.plot(x, y, color='red', linewidth=1)
            #ax.scatter(x, y, color='green')  # show endpoints
            

            dist = np.linalg.norm(outer_pt - inner_pt)

            if dist/dt > 10: continue
            
            best_dist.append( dist )
            best_outer.append(outer_pt )
            best_inner.append(inner_pt )
    
    #ax.set_aspect('equal')
    #plt.xlabel("X [m]")
    #plt.ylabel("Y [m]")
    #plt.show()
    #pdb.set_trace()
    #if best_outer is None or best_inner is None:
    #    return np.nan, None, None

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

    #global ax
    #ax =plt.subplot(111)
    gdfc = gdf.copy()
    gdf['geometry'] = gdf['geometry'].apply(densify_geometry)
   
    #gdfc.plot(facecolor='none', edgecolor='k',ax=ax)
    ##gdf.plot(facecolor='none', edgecolor='r', ax=ax)
    #plt.show()
    #pdb.set_trace()

    # Ensure projected CRS for distance calculations
    if gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError("GeoDataFrame must have a projected CRS (units in meters).")

    results = []

    for i, row_i in gdf.iterrows():
        outer = row_i.geometry
        if not isinstance(outer, Polygon) or outer.is_empty:
            continue

        for j, row_j in gdf.iloc[i + 1:].iterrows():
            #print(j)
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
            
            # Time difference in hours
            t1, t2 = row_i.timestamp,  row_j.timestamp
            dt_hours = (t1 - t2).total_seconds() 
            if dt_hours <= 0:
                continue

            # Compute maximum vertex distance
            obstacle_geoms = gdf.geometry.tolist()
            max_dist, outer_pt, inner_pt = compute_max_distance(outer, inner, dt_hours, obstacle_geoms )
           
           
            velocity = []
            vector_line = []
            for max_dist_, outer_pt_, inner_pt_ in zip(max_dist, outer_pt, inner_pt):
                
                if max_dist_ is None or np.isnan(max_dist_):
                    continue

                velocity.append( max_dist_ / dt_hours )  # meters/hour
                dx = inner_pt_[0] - outer_pt_[0]
                dy = inner_pt_[1] - outer_pt_[1]
                vector_line.append( LineString([(outer_pt_[0], outer_pt_[1]), (inner_pt_[0], inner_pt_[1])]) )

            if len(velocity) == 0 : continue

            velocity_out    = np.array(velocity).mean()
            velocity_std_out    = np.array(velocity).std()
            vector_line_out = mean_vector_line(vector_line)
            if velocity_out > 10: 
                pdb.set_trace()

            results.append({
                "outer_time": t1,
                "inner_time": t2,
                "velocity_m_per_s": velocity_out,
                "velocity_m_per_s_std": velocity_std_out,
                "outer_pt_x": vector_line_out.coords[-1][0],
                "outer_pt_y": vector_line_out.coords[-1][1],
                "inner_pt_x": vector_line_out.coords[0][0],
                "inner_pt_y": vector_line_out.coords[0][1],
                "geometry": vector_line_out,
            })
            
    #print( results[-1]['velocity_m_per_s'])

    if len( results ) > 0:
        gdf_results = gpd.GeoDataFrame(results, geometry='geometry', crs=gdf.crs)
    else: 
        gdf_results = gpd.GeoDataFrame(results)

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
