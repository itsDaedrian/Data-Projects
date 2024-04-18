import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

# Load the dataset
digits = load_digits()
X = digits.data
y = digits.target

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Remove constant features
selector = VarianceThreshold()
X_sel = selector.fit_transform(X_scaled)

# Check for missing or infinite values
print(np.isnan(X_sel).sum(), np.isinf(X_sel).sum())

# Remove any samples with missing or infinite values
X_sel = X_sel[np.logical_not(np.isnan(X_sel).any(axis=1))]
X_sel = X_sel[np.logical_not(np.isinf(X_sel).any(axis=1))]

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sel)

# Plot the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('PCA on Digits dataset')
plt.colorbar()
plt.show()