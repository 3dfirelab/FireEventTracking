import subprocess
import glob
import os

pbf_dir = "/data/shared/OSM"
out_dir = "/data/shared/OSM_Boundaries"
tmp_dir = "/tmp/osm_admin_tmp"

os.makedirs(out_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

pbf_files = glob.glob(os.path.join(pbf_dir, "*.osm.pbf"))

for pbf in sorted(pbf_files):
    base = os.path.basename(pbf).replace(".osm.pbf", "")
    tmp_pbf = os.path.join(tmp_dir, f"{base}-admin.osm.pbf")
    out_geojson = os.path.join(out_dir, f"{base}.geojson")

    # 1. Filter administrative boundaries
    subprocess.run(
        [
            "osmium", "tags-filter",
            pbf,
            "r/boundary=administrative",
            "w/boundary=administrative",
            "-o", tmp_pbf,
            "--overwrite",
        ],
        check=True,
    )

    # 2. Export polygons (ALL TAGS ARE KEPT BY DEFAULT)
    subprocess.run(
        [
            "osmium", "export",
            tmp_pbf,
            "-o", out_geojson,
            "--geometry-types", "polygon",
            "--overwrite",
        ],
        check=True,
    )

    print(f"Processed {base}")

