from pyspark.sql import SparkSession, functions as F
from pathlib import Path

years = [2019]

files = []
for y in years:
    files += [str(p) for p in Path(f"data/raw/{y}").glob("yellow_tripdata_1*.parquet")]

print("Files being read:", files[:5])

spark = SparkSession.builder.appName("NYC Taxi ETL").getOrCreate()

# Read multiple months (glob pattern)
df = spark.read.parquet("data/raw/2019/*.parquet")

# Add features
df = (
    df.withColumn("pickup_ts", F.to_timestamp("tpep_pickup_datetime"))
    .withColumn("year", F.year("pickup_ts"))
    .withColumn("month", F.month("pickup_ts"))
    .withColumn("revenue", F.col("total_amount"))
)

# Monthly aggregates
monthly = (
    df.groupBy("year", "month")
    .agg(
        F.count("*").alias("trips"),
        F.sum("passenger_count").alias("total_distance_mi"),
        F.sum("trip_distance").alias("total_fare"),
        F.sum("fare_amount").alias("total_tip"),
        F.sum("revenue").alias("total_revenue")
    )
    .orderBy("year","month")
)

monthly.show(10)
monthly.write.mode("overwrite").parquet("data/processed/metrics_monthly.parquet")