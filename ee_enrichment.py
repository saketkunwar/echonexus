# ee_enrichment.py
import ee
import pandas as pd
from config import PROJECT_ID
ee.Initialize(project=PROJECT_ID)


def enrich_with_dynamic_world(points_df: pd.DataFrame) -> pd.DataFrame:
    print("     Executing Earth Engine job for dominant land cover (Dynamic World)...")

    features = []
    for _, row in points_df.iterrows():
        try:
            point = ee.Geometry.Point([row["longitude"], row["latitude"]])
            event_date = ee.Date(row["event_timestamp"].strftime("%Y-%m-%d"))

            # Get the latest DW image before event_date
            dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                  .filterBounds(point)
                  .filterDate("2023-01-01", event_date)
                  .sort("system:time_start", False)  # latest first
                  .first())

            if dw:
                label = dw.select("label")
                result = label.reduceRegion(
                    reducer=ee.Reducer.mode(), geometry=point, scale=10
                ).getInfo()

                features.append({
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "dominant_land_cover": result.get("label") if result else None
                })
        except Exception as e:
            print(f"     Skipped point {row['latitude']},{row['longitude']} due to {e}")
            continue

    return pd.DataFrame(features)
