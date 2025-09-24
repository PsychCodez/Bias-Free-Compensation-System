import pandas as pd
import glob
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# === 1. Parameters ===
csv_root = "C:/Users/Hello/Documents/Capstone"
prefixes = ["0", "2", "3", "4", "5"]  # 0 = cluster 1
ndvi_min, ndvi_max = -0.3, 0.7

# Months to include: Nov 2022 → Apr 2022
month_ranges = [
    (11,2022),(12,2022),(1,2022),(2,2022),(3,2022),(4,2022)
]

# === 2. Collect all CSVs per cluster ===
dfs = []
for prefix in prefixes:
    cluster_name = "1" if prefix=="0" else prefix
    pattern = f"{csv_root}/{'ndvi_labels_cluster0*' if prefix=='0' else prefix+'_ndvi_labels_cluster_*'}.csv"

    for file in glob.glob(pattern):
        try:
            date_str = file.split("_")[-1].split(".")[0]  # ddmmyyyy
            file_date = datetime.strptime(date_str, "%d%m%Y")
        except:
            continue

        # Include only Nov → Apr
        if any(file_date.month==m and file_date.year==y for m,y in month_ranges):
            df = pd.read_csv(file)
            df["Date"] = file_date
            df["Cluster"] = cluster_name
            dfs.append(df)

big_df = pd.concat(dfs, ignore_index=True)
big_df = big_df.dropna(subset=["NDVI"])

# === 3. Average NDVI per pixel per day ===
avg_df = big_df.groupby(["Cluster","Latitude","Longitude","Label","Date"])["NDVI"].mean().reset_index()

# Pivot daily NDVI per pixel
pivot_df = avg_df.pivot_table(
    index=["Cluster","Latitude","Longitude","Label"],
    columns="Date",
    values="NDVI"
).reset_index()
pivot_df.columns.name = None

# Feature columns: all dates from Nov → Apr in sorted order
all_dates = [c for c in pivot_df.columns if isinstance(c, pd.Timestamp)]
all_dates = sorted(all_dates)

X = pivot_df[all_dates]
y = pivot_df["Label"]

# === 4. Train/test split & Random Forest ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# === 5. Prepare daily NDVI line plot per cluster ===
pivot_test = X_test.copy()
pivot_test["Cluster"] = pivot_df.loc[X_test.index, "Cluster"].values

# Actual daily NDVI per cluster (average across pixels)
plot_df = pivot_test.copy()
plot_df["Label"] = y_test.values
plot_df = plot_df.groupby("Cluster")[all_dates].mean().reset_index()
plot_df = plot_df.melt(id_vars="Cluster", var_name="Date", value_name="Actual_NDVI")

# Predicted NDVI: mean across test pixels
predicted_df = pivot_test[all_dates].mean(axis=0).reset_index()
predicted_df.columns = ["Date","Predicted_NDVI"]

# Extend predicted to May with NaN
may_dates = pd.date_range(datetime(2022,5,1), datetime(2022,5,31))
predicted_df = pd.concat([predicted_df, pd.DataFrame({"Date": may_dates, "Predicted_NDVI": np.nan})], ignore_index=True)

# Extend actual to May for all clusters
extra_rows = []
for cluster in pivot_test["Cluster"].unique():
    for d in may_dates:
        extra_rows.append({"Cluster": cluster, "Date": d, "Actual_NDVI": np.nan})
plot_df = pd.concat([plot_df, pd.DataFrame(extra_rows)], ignore_index=True)

# --- 6. Plot daily NDVI time series ---
plt.figure(figsize=(15,6))
for cluster in sorted(plot_df["Cluster"].unique()):
    dfc = plot_df[plot_df["Cluster"]==cluster].sort_values("Date")
    plt.plot(dfc["Date"], dfc["Actual_NDVI"], marker="o", linestyle="-", label=f"Cluster {cluster} Actual")

predicted_df_sorted = predicted_df.sort_values("Date")
plt.plot(predicted_df_sorted["Date"], predicted_df_sorted["Predicted_NDVI"], color="black", linewidth=2, marker="x", label="Predicted NDVI")

plt.ylim(ndvi_min, ndvi_max)
plt.xlabel("Date")
plt.ylabel("NDVI")
plt.title("Daily NDVI Time Series: Actual vs Predicted (Clusters 1-5, Nov 2022 → Apr 2022)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
