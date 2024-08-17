import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

mpl.rc('font', family='serif', serif='cmr10')
plt.rcParams['axes.unicode_minus'] = False

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb} \usepackage{newtxtext, newtxmath}'

# Ajustar el tamaño de fuente globalmente
plt.rcParams.update({'font.size': 14})

# Load the data
file_path = 'fpv_fwt_output.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Prepare the data for clustering
unique_scenarios = data['k'].unique()
scenario_vectors = []

for scenario in unique_scenarios:
    scenario_data = data[data['k'] == scenario]
    pv_wt_vector = np.hstack((scenario_data['fPV_MCS'].values, scenario_data['fWT_MCS'].values))
    scenario_vectors.append(pv_wt_vector)

scenario_vectors = np.array(scenario_vectors)

# Apply PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(scenario_vectors)

# Apply KMeans clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(reduced_vectors)

# Assign each scenario to a cluster
clusters = kmeans.labels_

# Create a DataFrame with scenario numbers and their corresponding cluster
clustered_scenarios = pd.DataFrame({'Scenario': unique_scenarios, 'Cluster': clusters})

# # Save the clusters to a CSV file
# clustered_scenarios.to_csv('clustered_scenarios.csv', index=False)

# Calculate the mean profile for each cluster
representative_profiles = []

for cluster in range(num_clusters):
    cluster_indices = clustered_scenarios[clustered_scenarios['Cluster'] == cluster].index
    cluster_data = scenario_vectors[cluster_indices]
    
    # Calculate the mean profile for both PV and WT
    mean_profile_pv = cluster_data[:, :8760].mean(axis=0)
    mean_profile_wt = cluster_data[:, 8760:].mean(axis=0)
    
    representative_profiles.append((mean_profile_pv, mean_profile_wt))

# Save the representative profiles to a CSV file
output_data = {}

for i, (mean_profile_pv, mean_profile_wt) in enumerate(representative_profiles):
    output_data[f'Cluster_{i+1}_PV'] = mean_profile_pv
    output_data[f'Cluster_{i+1}_WT'] = mean_profile_wt

output_df = pd.DataFrame(output_data)
output_df.to_csv('representative_profiles.csv', index=False)

# Calculate the number and percentage of scenarios in each cluster
cluster_counts = clustered_scenarios['Cluster'].value_counts().sort_index()
total_scenarios = len(unique_scenarios)
cluster_percentages = (cluster_counts / total_scenarios) * 100

# Print the number and percentage of scenarios in each cluster
for i in range(num_clusters):
    print(f'Cluster {i+1}: {cluster_counts[i]} scenarios ({cluster_percentages[i]:.2f}%)')

# Plot the clusters
plt.figure(figsize=(10, 7))
colors = ['b', 'r', 'g','c']
markers = ['o', 's', '^', 'D']

for i in range(num_clusters):
    cluster_points = clustered_scenarios[clustered_scenarios['Cluster'] == i]
    plt.scatter(reduced_vectors[cluster_points.index, 0], reduced_vectors[cluster_points.index, 1], 
                c=colors[i], label=f'Cluster {i+1}', marker=markers[i])

# Plot cluster centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, alpha=0.75, label='Centroids')

plt.title(f'Two component PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.savefig('PCA.pdf', bbox_inches='tight', pad_inches=0.02)
plt.show()
