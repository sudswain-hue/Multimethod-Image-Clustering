
import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#Q1
# Load and preprocess images
image_data = []
folder_path = r'D:\IU-CLASS\First-Sem\AML\Homework-03-Q1\360 Rocks'
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((64, 64))  # Resize for memory management
        img_array = np.array(img).flatten()
        image_data.append(img_array)

# Create data matrix
X = np.array(image_data)
# Apply PCA
pca = PCA()
pca.fit(X)
# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
# Find number of components for 90% variance
n_components = np.argmax(cumulative_variance_ratio >= 0.9) + 1
n_components

#Q2
# Select 10 random images
random_indices = np.random.choice(X.shape[0], 10, replace=False)
selected_images = X[random_indices]
# Apply PCA specifically to the selected images
pca = PCA(n_components=n_components)
# Fit PCA on the entire dataset first
pca.fit(X)
# Transform only the selected images
transformed_selected = pca.transform(selected_images)
reconstructed_selected = pca.inverse_transform(transformed_selected)
# Plot original and reconstructed images
fig, axes = plt.subplots(10, 2, figsize=(10, 25))
plt.subplots_adjust(top=0.95)  # Adjust top margin for main title
fig.suptitle('Original vs Reconstructed Images (90% Variance Preserved)', fontsize=16, y=0.98)

for i in range(10):
    # Original image
    orig_img = selected_images[i].reshape(64, 64)
    axes[i, 0].imshow(orig_img, cmap='gray', vmin=0, vmax=255)
    axes[i, 0].axis('off')
    
    # Reconstructed image
    recon_img = reconstructed_selected[i].reshape(64, 64)
    axes[i, 1].imshow(recon_img, cmap='gray', vmin=0, vmax=255)
    axes[i, 1].axis('off')

# Add column titles only once at the top
axes[0, 0].set_title('Original', pad=15, fontsize=14)
axes[0, 1].set_title('Reconstructed', pad=15, fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout while preserving space for main title
plt.show()


#Q3
# A. PCA with 2 components
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)

# Calculate variance explained by first two components
variance_explained = pca_2d.explained_variance_ratio_
total_variance = sum(variance_explained)
print(f"Variance explained by first two principal components: {total_variance:.4f}")

from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS

# Get categories from filenames
categories = []
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        categories.append(filename[0])  # First letter of filename

# Helper function to plot examples
def plot_examples(X_transformed, images, ax, num_examples=10):
    for i in range(num_examples):
        idx = np.random.randint(0, len(X_transformed))
        x, y = X_transformed[idx]
        img = images[idx].reshape(64, 64)
        imagebox = OffsetImage(img, zoom=0.3, cmap='gray')
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

# 1. PCA Plot
plt.figure(figsize=(10, 10))
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[ord(c) for c in categories], cmap='viridis')
plt.title('PCA')
plot_examples(X_pca, X, plt.gca())
plt.show()


# 2. t-SNE Plot
plt.figure(figsize=(10, 10))
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=[ord(c) for c in categories], cmap='viridis')
plt.title('t-SNE')
plot_examples(X_tsne, X, plt.gca())
plt.show()


# 3. LLE Plot
plt.figure(figsize=(10, 10))
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_lle = lle.fit_transform(X)
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=[ord(c) for c in categories], cmap='viridis')
plt.title('LLE')
plot_examples(X_lle, X, plt.gca())
plt.show()

# 4. MDS Plot
plt.figure(figsize=(10, 10))
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)
plt.scatter(X_mds[:, 0], X_mds[:, 1], c=[ord(c) for c in categories], cmap='viridis')
plt.title('MDS')
plot_examples(X_mds, X, plt.gca())
plt.show()


#Q4
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, MDS
from scipy.spatial import procrustes
import pandas as pd

# Load human features data
human_features = np.loadtxt(r'D:\IU-CLASS\First-Sem\AML\Homework-03-Q1\mds_360.txt')

# Reduce dimensionality to 8 components
# 1. PCA with 8 components
pca_8d = PCA(n_components=8)
X_pca_8d = pca_8d.fit_transform(X)

# 2. LLE with 8 components
lle_8d = LocallyLinearEmbedding(n_components=8, random_state=42)
X_lle_8d = lle_8d.fit_transform(X)

# 3. MDS with 8 components
mds_8d = MDS(n_components=8, random_state=42)
X_mds_8d = mds_8d.fit_transform(X)

# Perform Procrustes analysis
# Compare each method with human features
mtx1_pca, mtx2_pca, disparity_pca = procrustes(human_features, X_pca_8d)
mtx1_lle, mtx2_lle, disparity_lle = procrustes(human_features, X_lle_8d)
mtx1_mds, mtx2_mds, disparity_mds = procrustes(human_features, X_mds_8d)

# Print disparities
print("\nDisparity Values:")
print(f"PCA: {disparity_pca:.4f}")
print(f"LLE: {disparity_lle:.4f}")
print(f"MDS: {disparity_mds:.4f}")

# Calculate correlation coefficients for each dimension
correlations = []
methods = [
    ('PCA', mtx1_pca, mtx2_pca),
    ('LLE', mtx1_lle, mtx2_lle),
    ('MDS', mtx1_mds, mtx2_mds)
]

for name, mtx1, mtx2 in methods:
    corrs = [np.corrcoef(mtx1[:, i], mtx2[:, i])[0,1] for i in range(8)]
    correlations.append([name] + corrs)

# Create correlation table
df_correlations = pd.DataFrame(
    correlations,
    columns=['Method', 'Dim1', 'Dim2', 'Dim3', 'Dim4', 'Dim5', 'Dim6', 'Dim7', 'Dim8']
)

# Display correlation table
print("\nCorrelation Coefficients:")
print(df_correlations.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x) if isinstance(x, float) else x))

#Q5
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Use PCA with 93 components (90% variance preserved)
pca = PCA(n_components=93)
X_reduced = pca.fit_transform(X)

# Part 1: Determine optimal number of clusters using silhouette score
silhouette_scores = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_reduced)
    score = silhouette_score(X_reduced, kmeans.labels_)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()

# Part 2: Set k=3 and evaluate clustering
kmeans_3 = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans_3.fit_predict(X_reduced)

# Get true labels from filenames (I, M, S categories)
true_labels = []
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        true_labels.append(ord(filename[0]))  # Convert first letter to number

# Convert true labels to numeric (0, 1, 2)
unique_labels = np.unique(true_labels)
label_map = {label: i for i, label in enumerate(unique_labels)}
true_labels_numeric = [label_map[label] for label in true_labels]

# Calculate clustering accuracy
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Map cluster labels to true labels
cluster_to_true = np.zeros_like(cluster_labels)
for cluster in range(3):
    mask = (cluster_labels == cluster)
    cluster_to_true[mask] = mode(np.array(true_labels_numeric)[mask])[0]

accuracy = accuracy_score(true_labels_numeric, cluster_to_true)
print(f"Clustering Accuracy: {accuracy:.4f}")

#Q6
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np

# Use PCA with 93 components (90% variance preserved from previous analysis)
pca = PCA(n_components=93)
X_reduced = pca.fit_transform(X)

# Part A: Determine optimal number of clusters using BIC
n_components_range = range(1, 10)
bic = []
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_reduced)
    bic.append(gmm.bic(X_reduced))

# Plot BIC scores
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic, 'bo-')
plt.xlabel('Number of components')
plt.ylabel('BIC score')
plt.title('Model Selection using BIC')
plt.show()

# Part B: Fit GMM with 3 clusters and evaluate accuracy
gmm_3 = GaussianMixture(n_components=3, random_state=42)
cluster_labels = gmm_3.fit_predict(X_reduced)

# Get true labels from filenames
true_labels = []
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        true_labels.append(ord(filename[0]))

# Convert true labels to numeric
unique_labels = np.unique(true_labels)
label_map = {label: i for i, label in enumerate(unique_labels)}
true_labels_numeric = [label_map[label] for label in true_labels]

# Map cluster labels to true labels
cluster_to_true = np.zeros_like(cluster_labels)
for cluster in range(3):
    mask = (cluster_labels == cluster)
    cluster_to_true[mask] = mode(np.array(true_labels_numeric)[mask])[0]

accuracy = accuracy_score(true_labels_numeric, cluster_to_true)
print(f"Clustering Accuracy: {accuracy:.4f}")

# Part C: Generate and visualize new rocks
# Generate 20 new samples
new_samples = gmm_3.sample(20)[0]

# Transform back to original space
generated_rocks = pca.inverse_transform(new_samples)

# Plot generated rocks
plt.figure(figsize=(15, 8))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(generated_rocks[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
plt.suptitle('Generated Rock Images')
plt.tight_layout()
plt.show()


#Q7
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import numpy as np
from scipy.spatial import procrustes
import pandas as pd

# Custom Dataset class
class RockDataset(Dataset):
    def __init__(self, image_data, labels):
        self.images = torch.FloatTensor(image_data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Neural Network Architecture
class RockNet(nn.Module):
    def __init__(self):
        super(RockNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(4096, 512),  # 64x64 = 4096 input features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),  # 8 neurons before softmax
            nn.ReLU()
        )
        self.classifier = nn.Linear(8, 3)  # 3 classes (I, M, S)
        
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features), features
# Convert labels to numeric
label_map = {'I': 0, 'M': 1, 'S': 2}
train_labels = [label_map[filename[0]] for filename in os.listdir(folder_path) 
                if filename.endswith(('.jpg', '.png', '.jpeg'))]

# Create datasets
train_dataset = RockDataset(X, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = RockNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

start_time = time.time()

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs, features = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Calculate metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
bias_params = sum(p.numel() for p in model.parameters() if len(p.shape) == 1)
print(f"Total parameters: {total_params}")
print(f"Bias parameters: {bias_params}")

# Create validation dataset
val_data = []
val_labels = []
val_folder_path = r'D:\IU-CLASS\First-Sem\AML\Homework-03-Q1\120 Rocks'

# Load validation images
for filename in os.listdir(val_folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(val_folder_path, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((64, 64))  # Resize for memory management
        img_array = np.array(img).flatten()
        val_data.append(img_array)
        val_labels.append(ord(filename[0]) - ord('I'))  # Convert I, M, S to 0, 1, 2

# Convert to numpy arrays
X_val = np.array(val_data)
val_labels = np.array(val_labels)

# Create validation dataset and loader
val_dataset = RockDataset(X_val, val_labels)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Normalize and prepare features for Procrustes analysis
def prepare_for_procrustes(features):
    # Center the data
    features = features - np.mean(features, axis=0)
    # Add small random noise to ensure uniqueness
    features = features + np.random.normal(0, 1e-6, features.shape)
    # Scale to unit norm
    return features / np.linalg.norm(features, 'fro')

# Procrustes analysis
def get_features(model, loader):
    features_list = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            _, features = model(inputs)
            features_list.append(features.numpy())
    return np.concatenate(features_list)
# Load human data
train_human = np.loadtxt(r'D:\IU-CLASS\First-Sem\AML\Homework-03-Q1\mds_360.txt')
val_human = np.loadtxt(r'D:\IU-CLASS\First-Sem\AML\Homework-03-Q1\mds_120.txt')

# Get model features
train_features = get_features(model, train_loader)
val_features = get_features(model, val_loader)

# Prepare all data
train_features_prep = prepare_for_procrustes(train_features)
val_features_prep = prepare_for_procrustes(val_features)
train_human_prep = prepare_for_procrustes(train_human)
val_human_prep = prepare_for_procrustes(val_human)
# Perform Procrustes analysis
train_mtx1, train_mtx2, train_disparity = procrustes(train_human_prep, train_features_prep)
val_mtx1, val_mtx2, val_disparity = procrustes(val_human_prep, val_features_prep)

# Calculate correlations
train_correlations = [np.corrcoef(train_mtx1[:, i], train_mtx2[:, i])[0,1] for i in range(8)]
val_correlations = [np.corrcoef(val_mtx1[:, i], val_mtx2[:, i])[0,1] for i in range(8)]

# Create correlation table
correlation_df = pd.DataFrame({
    'Dimension': range(1, 9),
    'Training Correlation': train_correlations,
    'Validation Correlation': val_correlations
})

print("\nDisparity Values:")
print(f"Training: {train_disparity:.4f}")
print(f"Validation: {val_disparity:.4f}")
print("\nCorrelation Table:")
print(correlation_df)
