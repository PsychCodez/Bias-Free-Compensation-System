import rasterio
import pyproj
import csv
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.mask import mask
import numpy as np
import os
import glob
from datetime import datetime, timedelta

# --- 1. Define your cluster polygon correctly ---
cluster_coords = [
    (76.48700798425358, 9.708178529820032),
    (76.48700798425358, 9.701501975236496),
    (76.49486105179278, 9.701501975236496),
    (76.49486105179278, 9.708178529820032),
    (76.48700798425358, 9.708178529820032)  # close the polygon
]

polygon = Polygon(cluster_coords)
gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")


def extract_ndvi_with_labels(b04_file, b08_file, ground_truth_file, output_csv):
    with rasterio.open(b04_file) as src_b04, \
         rasterio.open(b08_file) as src_b08, \
         rasterio.open(ground_truth_file) as src_label:

        print(f"📦 Image CRS: {src_b04.crs}")
        print(f"📦 Ground Truth CRS: {src_label.crs}")

        # Reproject polygon to image CRS
        gdf_img = gdf.to_crs(src_b04.crs)

        try:
            b04_crop, transform = mask(src_b04, gdf_img.geometry, crop=True)
            b08_crop, _ = mask(src_b08, gdf_img.geometry, crop=True)
        except ValueError:
            print("❌ Error: Cluster does not intersect with the satellite image.")
            return

        gdf_label = gdf.to_crs(src_label.crs)

        try:
            label_crop, _ = mask(src_label, gdf_label.geometry, crop=True)
        except ValueError:
            print("❌ Error: Cluster does not intersect with the ground truth raster.")
            return

        b04 = b04_crop[0].astype(np.float32) / 10000.0
        b08 = b08_crop[0].astype(np.float32) / 10000.0
        labels = label_crop[0]

        transformer = pyproj.Transformer.from_crs(src_b04.crs, "EPSG:4326", always_xy=True)

        with open(output_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Latitude", "Longitude", "NDVI", "Label"])

            for row in range(b04.shape[0]):
                for col in range(b04.shape[1]):
                    b4 = b04[row, col]
                    b8 = b08[row, col]
                    label = labels[row, col]

                    if (b4 + b8) == 0:
                        continue

                    ndvi = (b8 - b4) / (b8 + b4)
                    x, y = transform * (col, row)
                    lon, lat = transformer.transform(x, y)
                    writer.writerow([lat, lon, ndvi, int(label)])

        print(f"✅ Data saved to: {output_csv}")


# --- 2. Iterate over multiple months (Nov 2022 to Apr 2022) ---

ground_truth_file = "C:/Users/Hello/Documents/Capstone/Capstone_Data/Ground_Truth_Data/classified-10m-India-2022-Summer_rice-WGS84.tif"

# Month/year list in the order you want:
# November 2022, December 2022, January 2022, February 2022, March 2022, April 2022
months_years = [
    (11, 2022),
    (12, 2022),
    (1, 2022),
    (2, 2022),
    (3, 2022),
    (4, 2022),
]

for month, year in months_years:
    # folder for this month
    month_name = datetime(year, month, 1).strftime("%B")  # e.g. November
    search_root = f"C:/Users/Hello/Documents/Capstone/Capstone_Data/Satellite_Data/{month_name}"

    # get start and end date for that month
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year, month, 31)
    elif month == 11:
        end_date = datetime(year, month, 30)
    else:
        # automatic month-end
        next_month = start_date.replace(day=28) + timedelta(days=4)
        end_date = next_month - timedelta(days=next_month.day)

    current_date = start_date
    while current_date <= end_date:
        dstr = current_date.strftime("%Y%m%d")   # 20220103 for file search
        daystr = current_date.strftime("%d%m%Y") # for CSV name

        b04_candidates = glob.glob(f"{search_root}/**/*{dstr}*B04_10m.jp2", recursive=True)
        b08_candidates = glob.glob(f"{search_root}/**/*{dstr}*B08_10m.jp2", recursive=True)

        if b04_candidates and b08_candidates:
            b04_file = b04_candidates[0]
            b08_file = b08_candidates[0]

            output_csv = f"C:/Users/Hello/Documents/Capstone/3_ndvi_labels_cluster_{daystr}.csv"

            print(f"\nProcessing {current_date.date()} …")
            extract_ndvi_with_labels(b04_file, b08_file, ground_truth_file, output_csv)
        else:
            print(f"No data for {current_date.date()}")

        current_date += timedelta(days=1)
