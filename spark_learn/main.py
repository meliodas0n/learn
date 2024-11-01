import time
from pyspark.sql import SparkSession

if __name__ == "__main__":
  spark = SparkSession.builder.master("local[1]").appName("Test").getOrCreate()
  print(f"Spark Session : {spark}")

  dataList = [("java", 200000), ("python", 2000000), ("scala", 30000)]
  rdd = spark.sparkContext.parallelize(dataList)
  print(f"rdd : {rdd}")

  print("Quitting")
  time.sleep(500000)
  spark.stop()