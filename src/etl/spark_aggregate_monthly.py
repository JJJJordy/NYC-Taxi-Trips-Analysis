#!/usr/bin/env python3
from pyspark.sql import SparkSession, functions as F
from pathlib import Path

YEARS = [2019]

# Build an explicit list of files to read (no wildcards at first)
files = [str(p) for y in YEARS for p in Path (f"data/raw/{y}").glob("yellow_tripdata_*.parquet")]
if not files:
    raise FileNotFoundError("No parquet files found under data/raw/<year>.")

print(f"Reading {len(files)} files, first 3:\n", "\n".join(files[:3]))

spark = SparkSession.builder.appName("NYC Taxi ETL").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(*files)
print("Schema:")
df.printSchema()

# Pick the correct pickup timestamp column (depends on taxi type)
pickup_col = next((c for c in ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"] if c in df.columns), None)
if not pickup_col:
    raise RuntimeError("Could not find a pickup datetime column in the data.")

df = (
    df
    .withColumn("pickup_ts", F.to_timestamp(F.col(pickup_col)))
    .withColumn("year", F.year("pickup_ts"))
    .withColumn("month", F.month("pickup_ts"))
)

# Cast numerics we need (skip if the column is missing)
for c in ["trip_distance", "fare_amount", "tip_amount", "total_amount", "passenger_count"]:
    if c in df.columns:
        df = df.withColumn(c, F.col(c).cast("double"))

# Derive a clean revenue and distance column
if "total_amount" in df.columns:
    df = df.withColumn("revenue", F.col("total_amount"))
if "trip_distance" in df.columns:
    df = df.withColumn("dist", F.col("trip_distance"))

# Keep only the years we expect (pre/during/post COVID window)
df = df.filter((F.col("year") >= 2018) & (F.col("year") <= 2021))
# Basic quality filters
df = df.filter((F.col("dist") > 0) & (F.col("revenue").isNotNull()))

monthly = (
    df.groupBy("year", "month")
      .agg(
          F.count("*").alias("trips"),
          F.sum("passenger_count").alias("total_passengers"),
          F.sum("dist").alias("total_distance_mi"),
          F.sum("fare_amount").alias("total_fare"),
          F.sum("tip_amount").alias("total_tip"),
          F.sum("revenue").alias("total_revenue"),
          F.avg("dist").alias("avg_trip_distance"),
          F.avg("fare_amount").alias("avg_fare"),
          F.avg("tip_amount").alias("avg_tip"),
      )
      .orderBy("year", "month")
)

monthly.show(24, truncate=False)
output_dir = "data/processed/metrics_monthly.parquet"
monthly.write.mode("overwrite").parquet(output_dir)
print("Wrote:", output_dir)