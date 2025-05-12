# Whitening Methods for Brain Imaging Feature Engineering

This document provides technical details about various whitening methods implemented in our ABIDE brain imaging clustering pipeline.

## Overview of Whitening

Whitening is a preprocessing technique that transforms a dataset so that:
1. Features are uncorrelated with each other
2. All features have unit variance

Mathematically, if we have a data matrix $X$ with covariance matrix $\Sigma$, after applying a whitening transformation $W$, the transformed data $Z = XW$ should have a covariance matrix that approaches the identity matrix:

$$\text{Cov}(Z) = I$$

Different whitening methods achieve this using different mathematical approaches, which can impact the final clustering performance.

## PCA Whitening

PCA whitening uses eigendecomposition of the covariance matrix to decorrelate features.

### Mathematics

1. Center the data: $X_c = X - \mu_X$
2. Compute the covariance matrix: $\Sigma = \frac{1}{n-1}X_c^TX_c$
3. Perform eigendecomposition: $\Sigma = U \Lambda U^T$
4. Apply the whitening transformation: $X_{white} = X_c U \Lambda^{-1/2}$

### Characteristics
- Rotates the data to align with principal components
- Orders features by explained variance
- Can be useful for dimensionality reduction

## ZCA Whitening (Zero-phase Component Analysis)

ZCA whitening is similar to PCA whitening but includes an additional rotation to maintain similarity to the original features.

### Mathematics

1. Center the data: $X_c = X - \mu_X$
2. Compute the covariance matrix: $\Sigma = \frac{1}{n-1}X_c^TX_c$
3. Perform SVD: $\Sigma = U S V^T$
4. Apply the ZCA transformation: $X_{zca} = X_c U S^{-1/2} U^T$

### Characteristics
- Preserves the original orientation of the data
- Maintains interpretability of features
- Better for visualization and when original feature meaning is important

## Cholesky Whitening

Cholesky whitening uses the Cholesky decomposition of the covariance matrix for whitening.

### Mathematics

1. Center the data: $X_c = X - \mu_X$
2. Compute the covariance matrix: $\Sigma = \frac{1}{n-1}X_c^TX_c$
3. Cholesky decomposition: $\Sigma = LL^T$
4. Apply Cholesky whitening: $X_{chol} = X_c L^{-1}$

### Characteristics
- Computationally efficient
- Produces a lower triangular transformation matrix
- Works well when features have clear hierarchical relationships

## Mahalanobis Whitening

Mahalanobis whitening uses the Mahalanobis distance to normalize and decorrelate features.

### Mathematics

1. Center the data: $X_c = X - \mu_X$
2. Compute the covariance matrix: $\Sigma = \frac{1}{n-1}X_c^TX_c$
3. Calculate the inverse square root: $P = \Sigma^{-1/2}$
4. Apply Mahalanobis transformation: $X_{mahal} = X_c P$

### Characteristics
- Directly normalizes based on statistical distance
- Robustness to heterogeneity in feature scales
- Particularly useful when features have very different distributions

## Adaptive PCA Whitening

Adaptive whitening modifies the whitening strength based on eigenvalue magnitudes.

### Mathematics

1. Center the data: $X_c = X - \mu_X$
2. Compute eigendecomposition: $\Sigma = U \Lambda U^T$
3. Apply adaptive scaling: $\Lambda_{adaptive} = \Lambda^{1-\alpha}$
4. Apply transformation: $X_{adaptive} = X_c U \Lambda^{-1/2} \text{diag}(\Lambda_{adaptive})$

Where $\alpha$ is the whitening strength parameter (0-1).

### Characteristics
- Allows control over the degree of whitening
- Can mitigate noise amplification
- Useful for balancing decorrelation and numerical stability

## Graph-based Feature Selection with Whitening

This approach combines whitening with graph-based feature selection.

### Process

1. Apply standard PCA whitening
2. Build a feature correlation graph
3. Use graph centrality measures to select most informative features
4. Apply final dimensionality reduction

### Characteristics
- Identifies feature importance from graph structure
- Can capture nonlinear relationships between features
- Reduces dimensionality while preserving key feature relationships

## Hybrid ZCA-Adaptive Whitening

This approach combines the advantages of both ZCA whitening and adaptive PCA whitening.

### Mathematics

1. Center the data: $X_c = X - \mu_X$
2. Compute covariance matrix: $\Sigma = \frac{1}{n-1}X_c^TX_c$
3. Perform SVD: $\Sigma = U S V^T$
4. Calculate ZCA component: $T_{ZCA} = U S^{-1/2} U^T$
5. Calculate adaptive component: $T_{adaptive} = U S^{-1/2} \text{diag}(S^{1-\alpha})$
6. Apply weighted combination: $X_{hybrid} = w \cdot X_c T_{ZCA} + (1-w) \cdot X_c T_{adaptive}$

Where $w$ is the ZCA weight parameter (0-1) and $\alpha$ is the adaptive whitening strength parameter (0-1).

### Characteristics
- Preserves interpretability of features like ZCA (controlled by ZCA weight)
- Provides control over whitening strength like adaptive PCA (controlled by alpha)
- Balances between noise reduction and feature interpretability
- Particularly effective for brain imaging data where both interpretability and noise reduction are important

## Comparison and Recommendations

### When to Use Each Method

- **PCA Whitening**: General purpose, good for initial exploration
- **ZCA Whitening**: When interpretability of features is important
- **Cholesky Whitening**: When computational efficiency is critical
- **Mahalanobis Whitening**: For data with heterogeneous feature scales
- **Adaptive Whitening**: When standard whitening amplifies noise
- **Graph-based Whitening**: For high-dimensional data with complex feature relationships
- **Hybrid Whitening**: When both interpretability and noise control are needed

### For Brain Imaging Data

For ABIDE and similar brain imaging datasets:

1. ZCA whitening often works well because it preserves interpretability of brain regions
2. Adaptive whitening with α = 0.7-0.9 can provide good balance between signal enhancement and noise reduction
3. Graph-based selection helps identify relevant brain connectivity patterns
4. Hybrid whitening with ZCA weight = 0.5-0.7 and α = 0.7-0.8 can provide the best of both worlds for neuroimaging data

### Parameter Selection Guidelines

- For noisy data: decrease adaptive alpha (0.5-0.7) and increase ZCA weight (0.6-0.8)
- For clean data with complex patterns: increase adaptive alpha (0.8-1.0) and decrease ZCA weight (0.3-0.5)
- For exploratory analysis: use balanced settings (ZCA weight = 0.5, alpha = 0.7)

## References

- Kessy, A., Lewin, A., & Strimmer, K. (2018). Optimal whitening and decorrelation. The American Statistician, 72(4), 309-314.
- Bell, A. J., & Sejnowski, T. J. (1997). The "independent components" of natural scenes are edge filters. Vision research, 37(23), 3327-3338.
- Yu, S., & Principe, J. C. (2019). Understanding autoencoders with information theoretic concepts. Neural Networks, 117, 104-123. 