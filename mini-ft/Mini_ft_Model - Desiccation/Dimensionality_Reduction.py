# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 14:10:02 2025

@author: Barney
"""

# Imports
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Plot settings
width = 8
plt.rcParams["figure.figsize"] = [width, 2/3*width]
plt.rcParams["figure.autolayout"] = True

# t-SNE function
def apply_tsne(mat_files, labels):
    all_data = []
    all_labels = []

    for mat_file, label in zip(mat_files, labels):
        mat = scipy.io.loadmat(mat_file)
        key = [k for k in mat.keys() if not k.startswith('__')][0]  # safer key selection
        data = pd.DataFrame(mat[key])
        all_data.append(data)
        all_labels.extend([label] * data.shape[0])

    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined data shape: {combined_data.shape}")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    n_samples = scaled_data.shape[0]
    perplexity_value = min(15, n_samples - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    tsne_result = tsne.fit_transform(scaled_data)

    return np.column_stack((tsne_result, np.array(all_labels)))

# Define your MAT file names and corresponding class labels
mat_files = [
    'features_response_front_2hr.mat',
    'features_response_front_3hr.mat',
    'features_response_front_4hr.mat',
    'features_response_front_5hr.mat',
    'features_response_front_6hr.mat',
    'features_response_front_7hr.mat'
]

# Class labels from 2 to 7
labels = np.array([29, 39, 51, 63, 74, 80])

# Run t-SNE
tsne_data = apply_tsne(mat_files, labels)

# Plotting
plt.figure()
sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=tsne_data[:, 2], palette='deep', s=40)
plt.title('t-SNE of Response Front Data (Classes 2–7)')
plt.legend(title="Class", loc='lower right')
plt.show()

# Save t-SNE results
save_filename = 'tsne_encoded_response_front.mat'
scipy.io.savemat(save_filename, {'tsne_encoded_features': tsne_data})
print(f"Saved: {save_filename}")
