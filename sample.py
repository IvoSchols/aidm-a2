import numpy as np
from scipy import sparse

data = np.load('./data/user_movie_rating.npy')
num_records = len(data)

users = data[:,0]
movies = data[:,1]
ratings = data[:,2]

n_users = np.max(users)
n_movies = np.max(movies)

rating_matrix = sparse.csc_matrix((ratings, (movies, users)))

print(f"Rating matrix shape: {rating_matrix.shape}")

# print(f"Number of records: {num_records}")

# subset = data[:652255]

# print(f"Subset shape: {subset.shape}")

# np.save('./data/user_movie_rating_subset.npy', subset)