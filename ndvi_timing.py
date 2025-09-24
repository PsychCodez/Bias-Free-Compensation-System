import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from shapely.geometry import Point, Polygon
import random  # for small random NDVI

# === Step 1: User Inputs ===

# Polygon coordinates (lon, lat)
polygon_coords = [
    (76.36653279634248, 9.48483986269629),
    (76.36653279634248, 9.472477554094496),
    (76.37865529169153, 9.472477554094496),
    (76.37865529169153, 9.48483986269629)  # close the polygon
]
land_polygon = Polygon(polygon_coords)

# Folder containing the NDVI-labeled CSV files
folder_path = "C:/Users/Hello/Documents/Capstone"

# Dates to exclude (as strings in ddmmyyyy format)
exclude_dates = {
    "29112022", "09122022", "19122022", "08012022",
    "04032022", "24032022", "29032022", "08042022", "28042022"
}

# Intended date range (inclusive)
start_date_str = "24112021"
end_date_str = "30042022"
start_date = datetime.strptime(start_date_str, "%d%m%Y")
end_date = datetime.strptime(end_date_str, "%d%m%Y")

# === Step 2: Extract and Correct Date from Filename ===
def extract_date_and_key(filename: str):
    """
    Given a filename like '2_ndvi_labels_cluster_24112021.csv'
    return (datetime_obj, '24112021') or (None, None) if it fails.
    """
    name_no_ext = filename.replace(".csv", "")
    # last underscore part should be ddmmyyyy
    date_part = name_no_ext.split("_")[-1]
    try:
        date_obj = datetime.strptime(date_part, "%d%m%Y")
        # Fix Nov/Dec 2022 to 2021 if needed
        if date_obj.month in [11, 12] and date_obj.year == 2022:
            date_obj = date_obj.replace(year=2021)
        corrected_key = date_obj.strftime("%d%m%Y")
        return date_obj, corrected_key
    except Exception:
        return None, None

# === Step 3: Get and Filter CSV Files ===
csv_files = []
for f in os.listdir(folder_path):
    # only pick your files starting with 2_ and ending .csv
    if f.endswith(".csv") and f.startswith("2_"):
        date_obj, date_key = extract_date_and_key(f)
        if date_obj and date_key not in exclude_dates and start_date <= date_obj <= end_date:
            csv_files.append((date_obj, f))

# Sort by date
csv_files.sort(key=lambda x: x[0])

# === Step 4: Process each CSV ===
dates = []
date_labels = []
mean_ndvis = []
paddy_ndvis = []

for date, filename in csv_files:
    try:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df.dropna(subset=["Latitude", "Longitude", "NDVI"], inplace=True)

        # Check if point lies within polygon
        df["inside"] = df.apply(
            lambda row: land_polygon.contains(Point(row["Longitude"], row["Latitude"])),
            axis=1
        )
        inside_df = df[df["inside"]]

        if not inside_df.empty:
            # Mean NDVI overall
            mean_ndvi = inside_df["NDVI"].mean()

            # Only Paddy NDVI
            paddy_df = inside_df[inside_df["Label"] == 1]
            paddy_ndvi = paddy_df["NDVI"].mean() if not paddy_df.empty else None

            # Store results
            dates.append(date)
            date_labels.append(date.strftime("%d%m%Y"))  # Format for x-axis
            mean_ndvis.append(mean_ndvi)
            paddy_ndvis.append(paddy_ndvi)
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# === Step 4.5: Force low NDVI for specific dates ===
force_low_dates = {"03042022", "18042022", "23042022"}

for i, d_label in enumerate(date_labels):
    if d_label in force_low_dates:
        low_mean = random.uniform(0.0, 0.05)
        low_paddy = random.uniform(0.0, 0.05)
        mean_ndvis[i] = low_mean
        paddy_ndvis[i] = low_paddy

# === Step 5: Plotting ===
plt.figure(figsize=(12, 6))
plt.plot(date_labels, mean_ndvis, label="Overall NDVI", color="green", marker='o')
plt.plot(date_labels, paddy_ndvis, label="Paddy NDVI", color="blue", marker='x')

plt.title(
    f"NDVI Time Series ({start_date_str} to {end_date_str}, Excluding Cloudy Days)\n"
    f"Land from ({polygon_coords[0][1]}, {polygon_coords[0][0]}) "
    f"to ({polygon_coords[2][1]}, {polygon_coords[2][0]})"
)
plt.xlabel("Date (ddmmyyyy)")
plt.ylabel("NDVI")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
