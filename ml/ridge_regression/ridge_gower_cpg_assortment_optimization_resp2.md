# Ridge Regression and Gower Distance for Assortment Optimization in the CPG Industry

This document is a complete, self-contained explanation of how Ridge Regression and Gower Distance can be combined to solve assortment optimization problems in the Consumer Packaged Goods (CPG) industry. It is written as a single continuous narrative intended for data engineers, data scientists, and analytics practitioners who need both conceptual clarity and implementation-level detail. All explanations, assumptions, and code are included in this file. Nothing outside this document is required to understand the approach.

---

## Assortment Optimization Problem Definition

Assortment optimization in CPG refers to the decision-making process that determines which Stock Keeping Units (SKUs) should be listed, retained, removed, or added to a retail assortment. The objective is typically to maximize category-level outcomes such as sales, profit, or shelf productivity while respecting constraints like limited shelf space, supplier agreements, and operational feasibility.

A critical complexity in assortment optimization is that products do not behave independently. When a SKU is removed, its demand does not disappear entirely. Instead, part of that demand is redistributed to similar products (substitution or cannibalization), while the remaining demand may be lost from the category altogether. Therefore, any realistic assortment model must answer two questions simultaneously:

1. What is the expected baseline demand of each SKU?
2. How does demand shift across SKUs when the assortment changes?

Ridge Regression addresses the first question by providing stable demand estimates under multicollinearity. Gower Distance addresses the second question by quantifying similarity between products with mixed attribute types.

---

## Why Ordinary Regression Fails in CPG Data

CPG datasets are structurally prone to multicollinearity. Price, promotion, pack size, brand positioning, and distribution tend to move together. For example, premium brands are often priced higher, promoted differently, and distributed more selectively. When ordinary least squares regression is applied in such settings, it produces unstable coefficients with high variance, sometimes even reversing expected signs. This instability makes scenario simulation unreliable, which is unacceptable for assortment decisions that directly affect revenue.

---

## Ridge Regression: Conceptual Explanation

Ridge Regression is a regularized linear regression technique that adds an L2 penalty to the loss function. The optimization objective is:

(y − Xβ)ᵀ(y − Xβ) + λβᵀβ

Here, X represents the feature matrix (price, promotion, pack size, etc.), y represents observed sales, β represents model coefficients, and λ is a regularization parameter controlling the strength of shrinkage.

The closed-form solution is:

β̂ = (XᵀX + λI)⁻¹Xᵀy

The key effect of ridge regression is coefficient shrinkage. When predictors are correlated, ridge regression distributes influence smoothly across them rather than allowing any single coefficient to dominate. This produces stable and realistic demand estimates, which are essential for downstream simulations.

In CPG applications, ridge regression is not primarily used for causal inference. Instead, it is used to generate robust baseline demand estimates that behave sensibly when product attributes change.

---

## Gower Distance: Conceptual Explanation

After estimating baseline demand, the next challenge is modeling substitution. This requires a measure of similarity between products that can handle mixed data types. Product attributes in CPG include numeric variables (price, pack size, distribution) and categorical variables (brand, promotion type, segment).

Gower Distance is designed specifically for this situation. It computes distance between two products as the average of per-attribute distances. Numeric attributes are normalized by their observed range, while categorical attributes are compared using simple equality. The resulting distance lies between 0 and 1.

A smaller Gower distance implies higher similarity and stronger substitution potential. A larger distance implies weaker substitution. This aligns closely with how category managers intuitively think about products.

---

## How Ridge Regression and Gower Distance Work Together

The combined approach works as follows:

1. Use ridge regression to estimate baseline demand for each SKU under the current assortment.
2. Use Gower distance to compute pairwise similarity between SKUs.
3. When simulating the removal of a SKU, redistribute its baseline demand to similar SKUs in proportion to their similarity.
4. Measure the net impact on category sales or profit.
5. Repeat the simulation for multiple SKUs to identify optimal assortment changes.

This framework is interpretable, scalable, and suitable for production environments.

---

## Full Python Implementation From Scratch

The following Python code demonstrates the complete workflow using synthetic CPG-style data. It includes data generation, ridge regression implemented from first principles, Gower distance computation, and demand redistribution logic.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

np.random.seed(42)

# -----------------------------
# Synthetic CPG Product Dataset
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
        scores = []
        for col in df.columns:
            if df[col].dtype == "object":
                scores.append(0 if df.iloc[i][col] == df.iloc[j][col] else 1)
            else:
                rng = df[col].max() - df[col].min()
                scores.append(abs(df.iloc[i][col] - df.iloc[j][col]) / rng)
        dist[i, j] = dist[j, i] = np.mean(scores)
    return dist

feature_cols = ["price", "pack_size", "brand", "promotion", "distribution"]
gower_dist = gower_distance(data[feature_cols])

# -----------------------------
# Ridge Regression From Scratch
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
    neighbors = np.argsort(distances)[1:top_k + 1]

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

PySpark Implementation for Production Scale

In large-scale CPG environments, data is typically processed using Spark. Ridge regression can be implemented using Spark MLlib with elasticNetParam set to zero. Gower distance must be implemented manually using Spark SQL expressions.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, abs, when, sum as spark_sum
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

sku_df = spark.createDataFrame(data)

# Index categorical variables
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

# -----------------------------
# Gower Distance in Spark
# -----------------------------
numeric_cols = ["price", "pack_size", "distribution"]
categorical_cols = ["brand", "promotion"]

df_a = sku_df.alias("a")
df_b = sku_df.alias("b")

exprs = []
for c in numeric_cols:
    exprs.append(abs(col(f"a.{c}") - col(f"b.{c}")))

for c in categorical_cols:
    exprs.append(when(col(f"a.{c}") == col(f"b.{c}"), 0).otherwise(1))

gower_df = df_a.crossJoin(df_b) \
    .withColumn("gower_distance", sum(exprs) / len(exprs))

# -----------------------------
# Redistribution Logic
# -----------------------------
window = Window.partitionBy("a.sku_id")

redistribution = gower_df \
    .withColumn("weight", 1 / col("gower_distance")) \
    .withColumn("normalized_weight",
                col("weight") / spark_sum("weight").over(window))
```

Closing Remarks

Ridge Regression provides stable and realistic baseline demand estimates in the presence of correlated CPG features. Gower Distance provides a principled and interpretable measure of product similarity across mixed data types. Together, they form a powerful framework for assortment optimization that supports scenario simulation, scales to large datasets, and aligns closely with how business stakeholders reason about products and substitution effects.