import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Plot experiments.
# Similarity measure
measures = ['Similarity measure' ,'js', 'cs', 'dcs']

# N_hashes tested
num_hashes = ['Number of hashes/projections', 100, 120, 150]

# Number of bands tested
num_bands = ['Number of bands', 20, 10, 5] # Increasing number of bands increases the number of false positives. (and runtime)
# Seeds tested
seeds = ['Seeds', 19, 42, 47]

# Timeout in seconds.
timeout = 30 * 60

# Throw error if no results folder is found.
if not os.path.exists('results'):
    raise RuntimeError('No results folder found. Run experiments first.')

# X axis
measure_data = []
num_hash_data = []
num_band_data = []
combined_data = []
seed_data = []

# Y axis
execution_time_data = []
found_pairs_data = []

# Load experiments from results folder.
for root, dirs, files in os.walk('results'):
    for file in files:
        file_parts = file.split('_')
        # Remove .txt extension.
        file_parts[-1] = file_parts[-1][:-4]

        measure_data.append(file_parts[0])
        num_hash_data.append(int(file_parts[1]))
        num_band_data.append(int(file_parts[2]))
        combined_data.append(f'{file_parts[0]} ({file_parts[1]}, {file_parts[2]})')
        
        
        seed_data.append(int(file_parts[3]))

        # Read number of found pairs from file.
        with open(f'{root}/{file}', 'r') as f:
            lines = f.readlines()
            execution_time_data.append(float(lines[-1]))
            found_pairs_data.append(len(lines) - 1)


# Create a DataFrame
df = pd.DataFrame({
    'Similarity measure, number of hashes/projections, number of bands': combined_data,
    'Execution time': execution_time_data,
    'Number of found pairs': found_pairs_data,
    'Seed': seed_data
})

unique_values = df['Similarity measure, number of hashes/projections, number of bands'].unique()
unique_values.sort()


# Create a figure with subplots
plt.figure(figsize=(14, 6))

# Plot Execution Time
plt.subplot(1, 2, 1)
sns.barplot(x='Similarity measure, number of hashes/projections, number of bands',
            y='Execution time', hue='Seed', data=df, palette='muted', order=unique_values)
plt.title('Execution Time')
plt.xticks(rotation=45, ha='right')

# Plot Number of Found Pairs
plt.subplot(1, 2, 2)
sns.barplot(x='Similarity measure, number of hashes/projections, number of bands',
            y='Number of found pairs', hue='Seed', data=df, palette='muted', order=unique_values)
plt.title('Number of Found Pairs')
plt.xticks(rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()