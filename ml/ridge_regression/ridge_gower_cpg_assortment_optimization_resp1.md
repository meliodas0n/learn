# Ridge Regression and Gower Distance for CPG Assortment Optimization (Complete End-to-End Document)

This document explains, in full detail, how Ridge Regression and Gower Distance can be used together for assortment optimization in the Consumer Packaged Goods (CPG) industry. It is written as a single continuous technical and business narrative, intended to be readable by data scientists, data engineers, and category managers. All explanations, reasoning, mathematical intuition, and implementation logic are contained within this document, along with complete Python and PySpark code written from scratch.

The goal of assortment optimization is to decide which SKUs should be present in a store or channel so that category-level objectives such as sales, profit, or shelf productivity are maximized under real-world constraints. Unlike simple forecasting problems, assortment optimization must explicitly account for substitution effects: when a product is removed, some of its demand shifts to similar products, while some demand disappears entirely. Modeling this accurately requires both a stable demand model and a principled notion of product similarity.

Ridge Regression is used in this framework to estimate stable baseline demand for each SKU in the presence of highly correlated predictors such as price, promotion, pack size, brand, and distribution. Gower Distance is used to measure similarity between products when attributes are a mixture of numeric and categorical variables. Together, these two techniques form a robust, interpretable, and scalable approach to assortment optimization that aligns well with how CPG categories actually behave.

We begin with demand modeling. In CPG data, ordinary least squares regression is often unstable because predictors are strongly correlated. Promotions tend to coincide with price changes, premium brands cluster at higher price points, and pack size is often tied to both price and brand positioning. Ridge Regression addresses this by adding an L2 penalty to the regression coefficients, shrinking them toward zero in a smooth way and reducing variance. The ridge objective function minimizes the squared error between actual and predicted sales, plus a penalty proportional to the squared magnitude of the coefficients. The result is a model that produces realistic, smooth demand estimates rather than extreme and fragile coefficients.

Formally, given a feature matrix X and a target vector y, ridge regression solves the optimization problem: minimize (y − Xβ)ᵀ(y − Xβ) + λβᵀβ, where β is the coefficient vector and λ is the regularization parameter. The closed-form solution is β̂ = (XᵀX + λI)⁻¹Xᵀy. In practice, the intercept term is not penalized. The key output of ridge regression in an assortment context is not the coefficients themselves, but the baseline demand predictions for each SKU under the current assortment.

Once baseline demand is estimated, the next challenge is understanding how demand reallocates when the assortment changes. This requires measuring similarity between products. In CPG categories, products are described by mixed-type attributes: numeric attributes such as price, pack size, and distribution, and categorical attributes such as brand, promotion type, and sub-segment. Traditional distance metrics such as Euclidean distance cannot handle categorical variables and are sensitive to scale differences. Gower Distance was designed specifically to address this problem.

Gower Distance computes the distance between two products as the average of per-attribute distances. For numeric attributes, the distance is the absolute difference normalized by the range of that attribute across the dataset. For categorical attributes, the distance is zero if the values match and one if they differ. The resulting distance lies between zero and one. Small distances indicate strong similarity and high substitution potential, while large distances indicate weak similarity.

In a CPG assortment context, Gower Distance aligns naturally with business intuition. Two SKUs with the same brand, similar price, similar pack size, and similar promotional behavior will have a small Gower distance and will strongly substitute for one another. Two SKUs that differ across multiple dimensions will have a larger distance and weaker substitution.

The combination of ridge regression and Gower distance enables a structured simulation process. First, ridge regression produces baseline demand estimates for all SKUs. Second, Gower distance identifies the nearest neighbors of each SKU. Third, when a SKU is removed, its baseline demand is redistributed to similar SKUs in proportion to their similarity, typically using inverse-distance weighting. This allows estimation of net category impact rather than assuming that all demand is lost or that it shifts arbitrarily.

The following Python code demonstrates a complete implementation from scratch. It generates synthetic CPG-style data, computes Gower distance manually, estimates a ridge regression model using linear algebra, and simulates the removal of a SKU with demand redistribution.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

np.random.seed(42)

# -----------------------------
# Synthetic CPG Product Data
# -----------------------------
n_products = 20

data = pd.DataFrame({
    "sku_id": [f"SKU_{i}" for i in range(n_products)],
    "price": np.random.uniform(20, 100, n_products),
    "pack_size": np.random.choice([250, 500, 1000], n_products),
    "brand": np.random.choice(["BrandA", "BrandB", "BrandC"], n_products),
    "promotion": np.random.choice(["None", "Discount", "BOGO"], n_products),
    "distribution": np.random.uniform(0.4, 0.95, n_products)
})

promo_effect = {"None": 0, "Discount": 1, "BOGO": 1.5}

# Demand generation (unknown in real life)
data["sales"] = (
    220
    - 0.7 * data["price"]
    + 0.6 * data["distribution"] * 100
    + 0.4 * data["promotion"].map(promo_effect) * 20
    + np.random.normal(0, 8, n_products)
)

# -----------------------------
# Gower Distance Implementation
# -----------------------------
def gower_distance(df):
    n = df.shape[0]
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        per_feature = []
        for col in df.columns:
            if df[col].dtype == "object":
                per_feature.append(0 if df.iloc[i][col] == df.iloc[j][col] else 1)
            else:
                rng = df[col].max() - df[col].min()
                per_feature.append(abs(df.iloc[i][col] - df.iloc[j][col]) / rng)
        dist[i, j] = dist[j, i] = np.mean(per_feature)
    return dist

feature_cols = ["price", "pack_size", "brand", "promotion", "distribution"]
gower_dist = gower_distance(data[feature_cols])

# -----------------------------
# Ridge Regression from Scratch
# -----------------------------
X = pd.get_dummies(data[feature_cols], drop_first=True).astype(float)
y = data["sales"].values

X_mat = np.c_[np.ones(X.shape[0]), X.values]
lambda_ = 10

I = np.eye(X_mat.shape[1])
I[0, 0] = 0

beta_ridge = np.linalg.inv(X_mat.T @ X_mat + lambda_ * I) @ X_mat.T @ y
data["baseline_sales"] = X_mat @ beta_ridge

# -----------------------------
# Assortment Simulation
# -----------------------------
def simulate_removal(df, dist_matrix, remove_index, top_k=3):
    removed_demand = df.loc[remove_index, "baseline_sales"]
    distances = dist_matrix[remove_index]
    neighbors = np.argsort(distances)[1:top_k+1]
    weights = 1 / distances[neighbors]
    weights = weights / weights.sum()

    df_sim = df.copy()
    for w, n in zip(weights, neighbors):
        df_sim.loc[n, "baseline_sales"] += w * removed_demand

    df_sim.loc[remove_index, "baseline_sales"] = 0
    return df_sim

simulated = simulate_removal(data, gower_dist, remove_index=0)

# -----------------------------
# Visualization
# -----------------------------
plt.figure()
plt.scatter(data["sales"], data["baseline_sales"])
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales (Ridge)")
plt.title("Ridge Regression Fit")
plt.show()

plt.figure()
plt.imshow(gower_dist)
plt.colorbar()
plt.title("Gower Distance Matrix")
plt.show()
```


This Python implementation illustrates the full analytical flow: demand estimation, similarity computation, and assortment simulation. In practice, CPG datasets are much larger and require distributed processing. PySpark is commonly used for this purpose. While PySpark does not provide native Gower distance functionality, the logic can be implemented using Spark SQL expressions. Ridge regression can be implemented using Spark MLlib by configuring linear regression with elasticNetParam set to zero.

The following PySpark code demonstrates how the same logic can be expressed at scale.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, abs, when, sum as spark_sum
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

sku_df = spark.createDataFrame(data)

# Encode categorical variables
brand_indexer = StringIndexer(inputCol="brand", outputCol="brand_idx")
promo_indexer = StringIndexer(inputCol="promotion", outputCol="promo_idx")

brand_encoder = OneHotEncoder(inputCol="brand_idx", outputCol="brand_vec")
promo_encoder = OneHotEncoder(inputCol="promo_idx", outputCol="promo_vec")

assembler = VectorAssembler(
    inputCols=["price", "pack_size", "distribution", "brand_vec", "promo_vec"],
    outputCol="features"
)

lr = LinearRegression(
    featuresCol="features",
    labelCol="sales",
    elasticNetParam=0.0,
    regParam=0.1
)

# Gower Distance Components
numeric_cols = ["price", "pack_size", "distribution"]
categorical_cols = ["brand", "promotion"]

a = sku_df.alias("a")
b = sku_df.alias("b")

exprs = []
for c in numeric_cols:
    exprs.append(abs(col(f"a.{c}") - col(f"b.{c}")))

for c in categorical_cols:
    exprs.append(when(col(f"a.{c}") == col(f"b.{c}"), 0).otherwise(1))

gower_df = a.crossJoin(b).withColumn(
    "gower_distance",
    sum(exprs) / len(exprs)
)

window = Window.partitionBy("a.sku_id")

redistribution = gower_df \
    .withColumn("weight", 1 / col("gower_distance")) \
    .withColumn("norm_weight", col("weight") / spark_sum("weight").over(window))
```

In production systems, the pairwise distance computation is usually restricted to relevant subsets such as within the same category or segment, and often limited to the top-N nearest neighbors to control computational cost. The overall approach remains the same: stable demand estimation using ridge regression, similarity measurement using Gower distance, and scenario simulation through weighted demand redistribution.

This combined framework works well in real CPG environments because it balances statistical rigor with interpretability, scales to large datasets, and aligns closely with how category managers reason about substitution and assortment decisions.