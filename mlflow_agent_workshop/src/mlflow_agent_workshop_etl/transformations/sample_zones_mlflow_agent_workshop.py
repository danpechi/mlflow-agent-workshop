from pyspark import pipelines as dp
from pyspark.sql.functions import col, sum


# This file defines a sample transformation.
# Edit the sample below or add new transformations
# using "+ Add" in the file browser.


@dp.table
def sample_zones_mlflow_agent_workshop():
    # Read from the "sample_trips" table, then sum all the fares
    return (
        spark.read.table(f"sample_trips_mlflow_agent_workshop")
        .groupBy(col("pickup_zip"))
        .agg(sum("fare_amount").alias("total_fare"))
    )
