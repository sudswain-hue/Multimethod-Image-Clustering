import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Load Digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Q1
pca = PCA(0.90)
pca.fit(X_train)
n_components_90 = pca.n_components_
print(f"Number of components to preserve 90% variance: {n_components_90}")

#Q2
pca = PCA(n_components=n_components_90)
X_pca = pca.fit_transform(X_train)
X_reconstructed = pca.inverse_transform(X_pca)

fig, axes = plt.subplots(2, 10, figsize=(20, 4))
for i in range(10):
    # Original image
    axes[0, i].imshow(X_train[i].reshape(8, 8), cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')
    # Reconstructed image
    axes[1, i].imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title('Reconstructed')
plt.tight_layout()
plt.show()

#Q3A
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_train)
variance_preserved = np.sum(pca_2d.explained_variance_ratio_)
print(f"Variance preserved with 2 components: {variance_preserved:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS

# Reduce dataset size for faster computation
n_samples = 1000
X_sample = X_train[:n_samples]
y_sample = y_train[:n_samples]

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_sample)

# LLE
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_lle = lle.fit_transform(X_sample)

# MDS
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X_sample)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='tab10')
axes[0].set_title('t-SNE')

axes[1].scatter(X_lle[:, 0], X_lle[:, 1], c=y_sample, cmap='tab10')
axes[1].set_title('LLE')

axes[2].scatter(X_mds[:, 0], X_mds[:, 1], c=y_sample, cmap='tab10')
axes[2].set_title('MDS')

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

#Q5A
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_pca_2d)
y_kmeans_pred = kmeans.predict(pca_2d.transform(X_test))

# Accuracy computation (using labels from clustering)
accuracy_kmeans = accuracy_score(y_test, y_kmeans_pred)
print(f"K-Means Accuracy: {accuracy_kmeans:.2f}")