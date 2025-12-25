# Ridge Regression and Gower Distance

This document gives a **from-first-principles explanation** of **Ridge Regression** and **Gower Distance**, including:
- Intuition and math
- When and why to use them
- Implementations **written from scratch (no ML libraries)**
- A **real‑world example** for each

---

## Part 1: Ridge Regression

### 1. What Problem Does Ridge Regression Solve?

In linear regression, we try to find coefficients \( \beta \) such that:

\[
\hat{y} = X\beta
\]

The coefficients are usually estimated by **Ordinary Least Squares (OLS)**:

\[
\beta = (X^TX)^{-1}X^Ty
\]

#### Problems with OLS

1. **Multicollinearity** – features are highly correlated
2. **Overfitting** – model fits noise
3. **Large coefficients** – unstable predictions

---

### 2. Ridge Regression Intuition

Ridge Regression adds a **penalty** for large coefficients.

> “Fit the data well, but keep coefficients small.”

This is called **L2 regularization**.

---

### 3. Mathematical Formulation

Ridge Regression minimizes:

\[
\mathcal{L}(\beta) = \|y - X\beta\|^2 + \lambda \|\beta\|^2
\]

Where:
- \( \|y - X\beta\|^2 \) → residual error
- \( \|\beta\|^2 = \sum \beta_j^2 \) → penalty
- \( \lambda \ge 0 \) → regularization strength

#### Closed‑Form Solution

\[
\beta = (X^TX + \lambda I)^{-1}X^Ty
\]

---

### 4. Geometric Interpretation

- OLS → unrestricted minimum
- Ridge → minimum inside a **circle constraint**
- Shrinks coefficients **continuously** (never exactly zero)

---

### 5. Ridge Regression from Scratch (Python)

```python
import math

# Matrix utilities

def transpose(A):
    return list(map(list, zip(*A)))


def matmul(A, B):
    result = [[0]*len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def inverse_2x2(M):
    det = M[0][0]*M[1][1] - M[0][1]*M[1][0]
    return [
        [ M[1][1]/det, -M[0][1]/det],
        [-M[1][0]/det,  M[0][0]/det]
    ]

# Ridge regression
def ridge_regression(X, y, lam):
    XT = transpose(X)
    XTX = matmul(XT, X)

    # Add lambda*I
    for i in range(len(XTX)):
        XTX[i][i] += lam

    XTy = matmul(XT, [[v] for v in y])
    beta = matmul(inverse_2x2(XTX), XTy)
    return beta
```

---

### 6. Real‑World Example: House Price Prediction

**Problem**: Predict house prices using:
- Size (sqft)
- Number of rooms

These features are correlated → OLS unstable.

```python
X = [
    [1, 800, 2],
    [1, 1000, 3],
    [1, 1200, 4]
]

y = [150000, 200000, 250000]

beta = ridge_regression(X, y, lam=10)
print(beta)
```

Ridge stabilizes coefficients and improves generalization.

---

## Part 2: Gower Distance

### 7. What Is Gower Distance?

Gower distance is used to measure **similarity between mixed‑type data**:
- Numeric
- Categorical
- Binary
- Ordinal

Traditional distances (Euclidean) **fail** here.

---

### 8. When Do You Need Gower Distance?

Examples:
- Customer segmentation
- Healthcare patient similarity
- Product recommendation

---

### 9. Gower Distance Formula

For two samples \( i, j \) with \( p \) features:

\[
d(i,j) = \frac{1}{p} \sum_{k=1}^p s_{ijk}
\]

Where:

- **Numeric**:
\[
|x_{ik} - x_{jk}| / (\max_k - \min_k)
\]

- **Categorical**:
\[
0 \text{ if equal, } 1 \text{ otherwise}
\]

---

### 10. Gower Distance from Scratch (Python)

```python
def gower_distance(x, y, feature_types, ranges):
    total = 0
    p = len(x)

    for i in range(p):
        if feature_types[i] == 'numeric':
            total += abs(x[i] - y[i]) / ranges[i]
        else:  # categorical
            total += 0 if x[i] == y[i] else 1

    return total / p
```

---

### 11. Real‑World Example: Customer Similarity

Customer features:
1. Age (numeric)
2. Income (numeric)
3. City (categorical)

```python
cust1 = [25, 50000, 'Delhi']
cust2 = [40, 52000, 'Mumbai']

feature_types = ['numeric', 'numeric', 'categorical']
ranges = [60, 100000, None]

d = gower_distance(cust1, cust2, feature_types, ranges)
print(d)
```

This distance can be used for:
- Clustering
- Nearest neighbors
- Recommendation systems

---

## 12. Ridge Regression vs Gower Distance

| Aspect | Ridge Regression | Gower Distance |
|------|-----------------|----------------|
| Type | Supervised | Distance metric |
| Purpose | Reduce overfitting | Measure similarity |
| Handles mixed data | ❌ | ✅ |
| Regularization | L2 | Not applicable |

---

## 13. Key Takeaways

- **Ridge Regression** combats multicollinearity and overfitting
- **Gower Distance** enables similarity for mixed‑type datasets
- Both are extremely useful in **real data engineering and ML systems**

---

If you want:
- Visualization
- Extension to Lasso / Elastic Net
- Using Gower distance in clustering

Tell me 👍

