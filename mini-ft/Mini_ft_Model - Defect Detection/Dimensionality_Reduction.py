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

# --- Configurable Section ---
# Choose class mode: '2', '4', or '9'
class_mode = '4'

# Define label setup based on class_mode
if class_mode == '2':
    non_normal_labels = [1] * 8
    label_suffix = '2cls'
elif class_mode == '4':
    non_normal_labels = [1, 1, 2, 2, 2, 3, 3, 3]
    label_suffix = '4cls'
elif class_mode == '9':
    non_normal_labels = list(range(1, 9))
    label_suffix = '9cls'
else:
    raise ValueError("Invalid class_mode. Use '2', '4', or '9'.")

# Shapes for non-normal files
non_normal_shapes = [
    'spherehd', 'cylinderhd', 'circle', 'sphereld',
    'cylinderld', 'slit', 'ssquare', 'bsquare'
]

# Process for each H value (2, 3, 5)
for H in [2, 3, 5]:
    normal_files = [f'features_normal{i}_S1_H{H}.mat' for i in range(11)]
    non_normal_files = [f'features_{shape}_S1_H{H}.mat' for shape in non_normal_shapes]

    all_files = normal_files + non_normal_files
    all_labels = [0] * len(normal_files) + non_normal_labels

    tsne_data = apply_tsne(all_files, all_labels)

    # Plotting
    plt.figure()
    sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=tsne_data[:, 2], palette='deep', s=40)
    plt.title(f't-SNE Result (H{H}, {label_suffix})')
    plt.legend(title="Class", loc='lower right')
    plt.show()

    # Save to file
    save_filename = f'tsne_encoded_features_{label_suffix}_{H}.mat'
    scipy.io.savemat(save_filename, {'tsne_encoded_features': tsne_data})
    print(f"Saved: {save_filename}")
