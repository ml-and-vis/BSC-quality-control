# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 16:04:44 2025

@author: Barney
"""

import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Load the data for loc 2
mat_file = 'tsne_encoded_features_2cls_2.mat'
data4 = scipy.io.loadmat(mat_file)
data4 = data4['tsne_encoded_features']

mat_file = 'tsne_encoded_features_9cls_2.mat'
data5 = scipy.io.loadmat(mat_file)
data5 = data5['tsne_encoded_features']

mat_file = 'tsne_encoded_features_4cls_2.mat'
data6 = scipy.io.loadmat(mat_file)
data6 = data6['tsne_encoded_features']

# Load the data for loc 3
mat_file = 'tsne_encoded_features_2cls_3.mat'
data1 = scipy.io.loadmat(mat_file)
data1 = data1['tsne_encoded_features']

mat_file = 'tsne_encoded_features_9cls_3.mat'
data2 = scipy.io.loadmat(mat_file)
data2 = data2['tsne_encoded_features']

mat_file = 'tsne_encoded_features_4cls_3.mat'
data3 = scipy.io.loadmat(mat_file)
data3 = data3['tsne_encoded_features']


# Load the data for loc 5
mat_file = 'tsne_encoded_features_2cls_5.mat'
data7 = scipy.io.loadmat(mat_file)
data7 = data7['tsne_encoded_features']

mat_file = 'tsne_encoded_features_9cls_5.mat'
data8 = scipy.io.loadmat(mat_file)
data8 = data8['tsne_encoded_features']

mat_file = 'tsne_encoded_features_4cls_5.mat'
data9 = scipy.io.loadmat(mat_file)
data9 = data9['tsne_encoded_features']

x = data1[:, 0]  # First column for x-axis
y = data1[:, 1]  # Second column for y-axis
labels = data1[:, 2]  # Third column for class labels (discrete)

# Create the figure and subplots using gridspec
fig = plt.figure(figsize=(17, 13))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])  # 2 rows, 2 columns with different width ratios

# First subplot (scatter plot)
ax4 = fig.add_subplot(gs[0, 0])  # First subplot on the left
ax5 = fig.add_subplot(gs[0, 2])  # Second subplot on the right
ax6 = fig.add_subplot(gs[0, 1])  # Second subplot on the left (second row)

ax1 = fig.add_subplot(gs[1, 0])  # First subplot on the left
ax2 = fig.add_subplot(gs[1, 2])  # Second subplot on the right
ax3 = fig.add_subplot(gs[1, 1])  # Second subplot on the left (second row)

ax7 = fig.add_subplot(gs[2, 0])  # First subplot on the left
ax8 = fig.add_subplot(gs[2, 2])  # Second subplot on the right
ax9 = fig.add_subplot(gs[2, 1])  # Second subplot on the left (second row)

scatter = sns.scatterplot(x=100, y=100, hue=labels, palette='colorblind', s=100, marker='o', ax=ax1)
scatter = sns.scatterplot(x=x, y=y, hue=labels, palette='colorblind', s=20, marker='o', ax=ax1)

# Customize the first subplot (scatter plot)
ax1.set_title('d', fontsize=24, fontweight='bold', loc='left')
ax1.set_xlabel('t-SNE Feature 1', fontsize=20)
ax1.set_ylabel('t-SNE Feature 2', fontsize=20)

# Move the legend below the plot
handles, _ = scatter.get_legend_handles_labels()
ax1.legend().set_visible(False)
#ax1.legend(handles=handles, labels=['Normal', 'Defect'], loc='upper center', fontsize=12, ncol=2, bbox_to_anchor=(0.5, -0.25))

# Second subplot (scatter plot for 9-class data)
x = data2[:, 0]  # First column for x-axis
y = data2[:, 1]  # Second column for y-axis
labels = data2[:, 2]  # Third column for class labels (discrete)
scatter = sns.scatterplot(x=100, y=100, hue=labels, palette='muted', s=100, marker='o', ax=ax2)
scatter = sns.scatterplot(x=x, y=y, hue=labels, palette='muted', s=35, marker='o', ax=ax2)

# Customize the second subplot (scatter plot)
ax2.set_title('f', fontsize=24, fontweight='bold', loc='left')
ax2.set_xlabel('t-SNE Feature 1', fontsize=20)
ax2.set_ylabel('t-SNE Feature 2', fontsize=20)
ax2.legend().set_visible(False)
# Move the legend below the plot
handles, _ = scatter.get_legend_handles_labels()
#ax2.legend(handles=handles, labels=['Normal', 'Sphere_H', 'Cylinder_H', 'Sphere_V', 'Cylinder_V', 'Circle_V', 'Diag_C', 'Ssquare_C', 'Bsquare_C'], 
#           loc='upper center', fontsize=12, ncol=3, bbox_to_anchor=(0.5, -0.09))

# Third subplot (scatter plot for 4-class data)
x = data3[:, 0]  # First column for x-axis
y = data3[:, 1]  # Second column for y-axis
labels = data3[:, 2]  # Third column for class labels (discrete)

colorblind_palette = sns.color_palette("deep")

scatter = sns.scatterplot(x=100, y=100, hue=labels, palette='deep', s=100, marker='o', ax=ax3)
scatter = sns.scatterplot(x=x, y=y, hue=labels, palette=colorblind_palette, s=20, marker='o', ax=ax3)

# Customize the third subplot (scatter plot)
ax3.set_title('e', fontsize=24, fontweight='bold', loc='left')
ax3.set_xlabel('t-SNE Feature 1', fontsize=20)
ax3.set_ylabel('t-SNE Feature 2', fontsize=20)

# Move the legend below the plot
handles, _ = scatter.get_legend_handles_labels()
#ax3.legend(handles=handles, labels=['Normal', 'Defect [H]', 'Defect [V]', 'Defect [C]'], loc='upper center', fontsize=12, ncol=2, bbox_to_anchor=(0.5, -0.25))
ax3.legend().set_visible(False)

# Load the data for the first row (subplots ax4, ax5, ax6)
x = data4[:, 0]  # First column for x-axis
y = data4[:, 1]  # Second column for y-axis
labels = data4[:, 2]  # Third column for class labels (discrete)

# First subplot for the first row (ax4)
scatter = sns.scatterplot(x=100, y=100, hue=labels, palette='colorblind', s=100, marker='o', ax=ax4)
scatter = sns.scatterplot(x=x, y=y, hue=labels, palette='colorblind', s=20, marker='o', ax=ax4)

# Customize the first subplot in the first row
ax4.set_title('a', fontsize=24, fontweight='bold', loc='left')
ax4.set_xlabel('t-SNE Feature 1', fontsize=20)
ax4.set_ylabel('t-SNE Feature 2', fontsize=20)
ax4.legend().set_visible(False)
ax4.set_xlim(-75, 75)
ax4.set_ylim(-75, 75)
ax4.tick_params(axis='both', which='major', labelsize=14)

# Second subplot for the first row (ax5)
x = data5[:, 0]  # First column for x-axis
y = data5[:, 1]  # Second column for y-axis
labels = data5[:, 2]  # Third column for class labels (discrete)

scatter = sns.scatterplot(x=100, y=100, hue=labels, palette='muted', s=100, marker='o', ax=ax5)
scatter = sns.scatterplot(x=x, y=y, hue=labels, palette='muted', s=35, marker='o', ax=ax5)

# Customize the second subplot in the first row
ax5.set_title('c', fontsize=24, fontweight='bold', loc='left')
ax5.set_xlabel('t-SNE Feature 1', fontsize=20)
ax5.set_ylabel('t-SNE Feature 2', fontsize=20)
ax5.legend().set_visible(False)
ax5.set_xlim(-75, 75)
ax5.set_ylim(-75, 75)
ax5.tick_params(axis='both', which='major', labelsize=14)

# Third subplot for the first row (ax6)
x = data6[:, 0]  # First column for x-axis
y = data6[:, 1]  # Second column for y-axis
labels = data6[:, 2]  # Third column for class labels (discrete)

colorblind_palette = sns.color_palette("deep")

scatter = sns.scatterplot(x=100, y=100, hue=labels, palette='deep', s=100, marker='o', ax=ax6)
scatter = sns.scatterplot(x=x, y=y, hue=labels, palette=colorblind_palette, s=20, marker='o', ax=ax6)

# Customize the third subplot in the first row
ax6.set_title('b', fontsize=24, fontweight='bold', loc='left')
ax6.set_xlabel('t-SNE Feature 1', fontsize=20)
ax6.set_ylabel('t-SNE Feature 2', fontsize=20)
ax6.legend().set_visible(False)
ax6.set_xlim(-75, 75)
ax6.set_ylim(-75, 75)
ax6.tick_params(axis='both', which='major', labelsize=14)

# Load the data for the third row (subplots ax7, ax8, ax9)
x = data7[:, 0]  # First column for x-axis
y = data7[:, 1]  # Second column for y-axis
labels = data7[:, 2]  # Third column for class labels (discrete)

# First subplot for the third row (ax7)
scatter = sns.scatterplot(x=100, y=100, hue=labels, palette='colorblind', s=200, marker='o', ax=ax7)
scatter = sns.scatterplot(x=x, y=y, hue=labels, palette='colorblind', s=20, marker='o', ax=ax7)

# Customize the first subplot in the third row
ax7.set_title('g', fontsize=24, fontweight='bold', loc='left')
ax7.set_xlabel('t-SNE Feature 1', fontsize=20)
ax7.set_ylabel('t-SNE Feature 2', fontsize=20)
ax7.set_xlim(-75, 75)
ax7.set_ylim(-75, 75)
ax7.tick_params(axis='both', which='major', labelsize=14)
handles, _ = scatter.get_legend_handles_labels()

ax7.legend(handles=handles, labels=['Normal', 'Defect'], loc='upper center', fontsize=14, ncol=2, bbox_to_anchor=(0.5, -0.25))

# Second subplot for the third row (ax8)
x = data8[:, 0]  # First column for x-axis
y = data8[:, 1]  # Second column for y-axis
labels = data8[:, 2]  # Third column for class labels (discrete)

scatter = sns.scatterplot(x=100, y=100, hue=labels, palette='muted', s=200, marker='o', ax=ax8)
scatter = sns.scatterplot(x=x, y=y, hue=labels, palette='muted', s=35, marker='o', ax=ax8)

# Customize the second subplot in the third row
ax8.set_title('i', fontsize=24, fontweight='bold', loc='left')
ax8.set_xlabel('t-SNE Feature 1', fontsize=20)
ax8.set_ylabel('t-SNE Feature 2', fontsize=20)
ax8.set_xlim(-75, 75)
ax8.set_ylim(-75, 75)
ax8.tick_params(axis='both', which='major', labelsize=14)
handles, _ = scatter.get_legend_handles_labels()

ax8.legend(handles=handles, labels=['Normal', 'Sphere_H', 'Cylinder_H', 'Sphere_V', 'Cylinder_V', 'Circle_V', 'Diag_C', 'Ssquare_C', 'Bsquare_C'], 
           loc='upper center', fontsize=14, ncol=3, bbox_to_anchor=(0.5, -0.20))

# Third subplot for the third row (ax9)
x = data9[:, 0]  # First column for x-axis
y = data9[:, 1]  # Second column for y-axis
labels = data9[:, 2]  # Third column for class labels (discrete)

colorblind_palette = sns.color_palette("deep")

scatter = sns.scatterplot(x=100, y=100, hue=labels, palette='deep', s=200, marker='o', ax=ax9)
scatter = sns.scatterplot(x=x, y=y, hue=labels, palette=colorblind_palette, s=20, marker='o', ax=ax9)

# Customize the third subplot in the third row
ax9.set_title('h', fontsize=24, fontweight='bold', loc='left')
ax9.set_xlabel('t-SNE Feature 1', fontsize=20)
ax9.set_ylabel('t-SNE Feature 2', fontsize=20)
ax9.set_xlim(-75, 75)
ax9.set_ylim(-75, 75)
ax9.tick_params(axis='both', which='major', labelsize=14)
handles, _ = scatter.get_legend_handles_labels()
ax9.legend(handles=handles, labels=['Normal', 'Defect [H]', 'Defect [V]', 'Defect [C]'], loc='upper center', fontsize=14, ncol=2, bbox_to_anchor=(0.5, -0.25))

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)
ax9.spines['top'].set_visible(False)
ax9.spines['right'].set_visible(False)

ax1.set_xlim(-75,75)
ax1.set_ylim(-75,75)
ax2.set_xlim(-75,75)
ax2.set_ylim(-75,75)
ax3.set_xlim(-75,75)
ax3.set_ylim(-75,75)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)

ellipse = Ellipse((-10,-12), 15, 50, angle=50, edgecolor='k', facecolor='none', lw=1)
ax2.add_patch(ellipse)
ax2.arrow(-35, -35, 15, 15, head_width=2, head_length=2, fc='k', ec='k')

ax2.text(-70, -50, 
              'Defects [V]', 
              fontsize=16)  

ellipse = Ellipse((35,-10), 20, 50, angle=105, edgecolor='k', facecolor='none', lw=1)
ax2.add_patch(ellipse)
ax2.arrow(45, -30, -5, 5, head_width=2, head_length=2, fc='k', ec='k')
ax2.text(47, -35, 
              'Defects [H]', 
              fontsize=16)  

ellipse = Ellipse((20, 60), 20, 25, angle=0, edgecolor='k', facecolor='none', lw=1)
ax2.add_patch(ellipse)

ax2.arrow(45, 60, -8, 0, head_width=2, head_length=2, fc='k', ec='k')

ax2.text(50, 55, 
              'Defects [C]', 
              fontsize=16)  

ellipse = Ellipse((27, 0), 15, 65, angle=75, edgecolor='k', facecolor='none', lw=1)
ax5.add_patch(ellipse)
ax5.arrow(60, -25, -5, 10, head_width=2, head_length=2, fc='k', ec='k')
ax5.text(37, -35, 
              'Defects [V]', 
              fontsize=16)  # Add background color with transparency

ellipse = Ellipse((-15, 19), 15, 50, angle=90, edgecolor='k', facecolor='none', lw=1)
ax5.add_patch(ellipse)

ax5.arrow(-2, 35, 0, -5, head_width=2, head_length=2, fc='k', ec='k')

ax5.text(-25, 37, 
              'Defects [C]', 
              fontsize=16)  # Add background color with transparency

ellipse = Ellipse((-57,-15), 25, 55, angle=165, edgecolor='k', facecolor='none', lw=1)
ax5.add_patch(ellipse)

ax5.arrow(-55, 25, 0, -10, head_width=2, head_length=2, fc='k', ec='k')

ax5.text(-74, 30, 
              'Defects [H]', 
              fontsize=16)  # Add background color with transparency

ellipse = Ellipse((-20,30), 25, 90, angle=135, edgecolor='k', facecolor='none', lw=1)
ax8.add_patch(ellipse)
ax8.arrow(-25, 55, 0, -8, head_width=2, head_length=2, fc='k', ec='k')

ax8.text(-60, 60, 
              'Defects [H]', 
              fontsize=16)  # Add background color with transparency

ellipse = Ellipse((-5,-17), 15, 40, angle=70, edgecolor='k', facecolor='none', lw=1)
ax8.add_patch(ellipse)
ax8.arrow(-45, -30, 17, 15, head_width=2, head_length=2, fc='k', ec='k')

ax8.text(-74, -42, 
              'Defects [V]', 
              fontsize=16)  # Add background color with transparency

ellipse = Ellipse((12,0), 15, 60, angle=70, edgecolor='k', facecolor='none', lw=1)
ax8.add_patch(ellipse)

ax8.arrow(45, -25, -5, 8, head_width=2, head_length=2, fc='k', ec='k')

ax8.text(47, -35, 
              'Defects [C]', 
              fontsize=16)  # Add background color with transparency

fig.text(-0.05, 0.85, 'Tap @\nlocation 2', fontsize=20, fontweight='normal', rotation=0, ha='center', va='center',color='midnightblue', bbox=dict(facecolor='white', alpha=0.3))


fig.text(-0.05, 0.55, 'Tap @\nlocation 3', fontsize=20, fontweight='normal', rotation=0, ha='center', va='center',color='midnightblue', bbox=dict(facecolor='white', alpha=0.3))

fig.text(-0.05, 0.24, 'Tap @\nlocation 5', fontsize=20, fontweight='normal', rotation=0, ha='center', va='center',color='midnightblue', bbox=dict(facecolor='white', alpha=0.3))

# Adjust layout to prevent overlap and ensure legends are positioned outside
plt.tight_layout()

# Save the figure
plt.savefig('TSNE_Visualization.png', dpi=300, bbox_inches='tight', pad_inches=0.5)

# Show the plot
plt.show()
