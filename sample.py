import numpy as np

data = np.load('./data/user_movie_rating.npy')
num_records = len(data)

print(f"Number of records: {num_records}")

subset = data[:652255]

print(f"Subset shape: {subset.shape}")

np.save('./data/user_movie_rating_subset.npy', subset)