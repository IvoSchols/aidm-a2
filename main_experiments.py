from collections import defaultdict
import time
import numpy as np
from scipy import sparse
import argparse
from sklearn.random_projection import SparseRandomProjection
from itertools import combinations
import os
import subprocess
import multiprocessing
from time import perf_counter



# Load data from file, return a sparse matrix of shape (movies, users)
def load_data(file_path : str, similarity_measure : str):
    # Ratings is a numpy array of shape (n_ratings, 3): [user_id, movie_id, rating]
    ratings_data = np.load(file_path) 
    
    # Create a row based sparse matrix of shape (movies, users) from the ratings for fast access in JS
    # Create a column based sparse matrix of shape (movies, users) from the ratings for fast access in CS/DCS
    users = ratings_data[:,0]
    movies = ratings_data[:,1]
    ratings = ratings_data[:,2]
    
    if similarity_measure == 'js':
        ratings = np.ones(len(ratings))
        rating_matrix = sparse.csr_matrix((ratings, (movies, users)), dtype='?')
    elif similarity_measure == 'cs':
        rating_matrix = sparse.csc_matrix((ratings, (users, movies)), dtype='float')
    elif similarity_measure == 'dcs':
        ratings = np.ones(len(ratings))
        rating_matrix = sparse.csc_matrix((ratings, (users, movies)), dtype='?')

    return rating_matrix

# Append candidate pairs to file
def append_result(candidate_pairs : list, file_name : str):
    with open(file_name, 'a') as f:
        for (user1,user2) in candidate_pairs:
            f.write("%s,%s\n" % (user1, user2))



##
# MinHash
##

# Takes a rating matrix of shape (n_movies, n_users) and returns a signature matrix of shape (n_hashes, n_users)
def minhash_jaccard(rating_matrix, n_hashes : int):
    n_movies, n_users = rating_matrix.shape

    # Generate signature matrix with all values set to infinity
    signature_matrix = np.full((n_hashes,n_users),np.inf)

    for i in range(n_hashes):
        # Generate a random permutation of the rows and swap them
        swapped_matrix = rating_matrix[np.random.permutation(n_movies), :]

        # Find the index of the first True value for each user
        min_hashes = np.argmax(swapped_matrix > 0, axis=0)

        # Store the minimum hash for each user
        signature_matrix[i, :] = np.minimum(min_hashes, signature_matrix[i, :])

    return signature_matrix.astype(int)

##
# LSH
##

# Return candidate pairs of shape (n_candidate_pairs, 2) from the given signature matrix
# Divide the signature matrix into n_bands bands and n_rows rows per band
# If two users are similar, that is:
#   - jaccard_similarity > 0.5
def lsh_jaccard(signature_matrix : np.ndarray, n_bands : int, file_name : str):
    n_hashes, _ = signature_matrix.shape
    rows_per_band = n_hashes // n_bands

    # Apply LSH to find candidate pairs
    for band in range(n_bands):
        # Extract a band from the signature matrix
        band_matrix = signature_matrix[band * rows_per_band: (band + 1) * rows_per_band, :]

        # Collapse the rows of the band into a single value for each user
        dest_bucket = np.sum(band_matrix, axis=0)

        # Find unique buckets and map each user to its bucket
        buckets, bucket_indices = np.unique(dest_bucket, return_inverse=True)
        bucket_users_dict = defaultdict(list)
        for bucket_index in range(len(buckets)):
            bucket_users_dict[bucket_index] = np.argwhere(bucket_indices == bucket_index).flatten()

        similar_users = list()

        for user_bucket in bucket_users_dict.values():
            # Iterate through each pair of users in the bucket
            for user1, user2 in combinations(user_bucket, 2):
                user1_sig = signature_matrix[:, user1]
                user2_sig = signature_matrix[:, user2]

                # Add pair to the set if above the threshold -> calculation is done inline for performance reasons
                intersection_size = np.sum(user1_sig & user2_sig)
                union_size = np.sum(user1_sig | user2_sig)
                similarity = intersection_size / union_size
                if similarity > 0.5:
                    similar_users.append((user1,user2))

        append_result(similar_users, f"{file_name}.txt")


# Return candidate pairs of shape (n_candidate_pairs, 2) by applying LSH to the given projected matrix (users, movies)
# We use random projection to reduce the dimensionality of the rating matrix
# and then bin the users into buckets based on their hash
#   - cosine_similarity > 0.73
#   - discrete_cosine_similarity > 0.73
# @profile
def lsh_cosine(projected_matrix, n_bands, similarity_measure, file_name):
    n_users, n_projections = projected_matrix.shape

    columns_per_band = n_projections // n_bands
    # Divide the hashed vectors into n_bands bands and n_rows rows per band
    for band in range(n_bands):
        #Extract a band from the projected matrix
        band_matrix = projected_matrix[:, band * columns_per_band: (band + 1) * columns_per_band]

        # Hash the band matrix
        hashed_band_matrix = np.sign(band_matrix)
        # Collapse the columns of the band into a single value for each user and calculate its destination bucket
        dest_bucket = np.sum(hashed_band_matrix, axis=1).astype(int)

        # Find unique buckets and map each user to its bucket
        buckets, bucket_indices = np.unique(dest_bucket, return_inverse=True)
        bucket_users_dict = defaultdict(list)
        for bucket_index in range(len(buckets)):
            bucket_users_dict[bucket_index] = np.argwhere(bucket_indices == bucket_index).flatten()

        similar_users = list()

        for user_bucket in bucket_users_dict.values():
            # Iterate through each pair of users in the bucket
            for user1, user2 in combinations(user_bucket, 2):
                user1_vec = projected_matrix[user1]
                user2_vec = projected_matrix[user2]

                norm_user1 = np.linalg.norm(user1_vec)
                norm_user2 = np.linalg.norm(user2_vec)

                cosine = np.dot(user1_vec, user2_vec) / (norm_user1 * norm_user2)
                
                theta = np.arccos(cosine)
                similarity = 1 - theta/180

                if similarity > 0.73:
                    similar_users.append((user1, user2))
        
        append_result(similar_users, f"{file_name}.txt")


def run_experiment(data, similarity_measure, num_hash, num_band, seed, timeout):
    print(f'Running experiment with measure = {similarity_measure}, num_hash = {num_hash}, num_band = {num_band}, seed = {seed}')
    np.random.seed(seed)
    file_name = f'{similarity_measure}_{num_hash}_{num_band}_{seed}'

    start = perf_counter()
 
    if similarity_measure == 'js':
        signature_matrix = minhash_jaccard(rating_matrix, num_hash)
    elif similarity_measure == 'cs' or similarity_measure == 'dcs':
        projection_matrix = SparseRandomProjection(n_components=num_hash, random_state=seed, dense_output=True)
        projected_matrix = projection_matrix.fit_transform(rating_matrix) # Project the rating matrix onto a lower dimensional space
        del projection_matrix # free memory

    del rating_matrix # free memory
    # Stop timer
    stop = time.time()
    print("Time elapsed for minhash/projection matrix: ", stop - start)

    # Compute LSH
    if similarity_measure == 'js':
        lsh_jaccard(signature_matrix, num_band, file_name)
    elif similarity_measure == 'cs' or similarity_measure == 'dcs':
        lsh_cosine(projected_matrix, num_band, similarity_measure, file_name)


    execution_time = perf_counter() - start

    # Append execution time to results file.
    with open(f'{file_name}.txt', 'a') as f:
        f.write(f'{execution_time}\n')

    print(f'Done with experiment: {file_name}')

def main():

    # Arguments that will be passed to main.py.
    measures = ['js', 'cs', 'dcs']
    num_hashes = [100, 120, 150]
    num_projections = [750, 1500, 3000]
    num_bands = [20, 10, 5]
    seeds = [19, 42, 47]
    timeout = 30 * 60

    # Start timer
    start = time.time()

    # Load data
    rating_matrix_js = load_data('data/user_movie_rating.npy', 'js')
    rating_matrix_cs = load_data('data/user_movie_rating.npy', 'cs')
    rating_matrix_dcs = load_data('data/user_movie_rating.npy', 'dcs') 

    # Create a pool of workers
    with multiprocessing.Pool(2) as pool:
        jobs = []
        for measure in measures:
            for num_hash, num_projection in zip(num_hashes, num_projections):
                for num_band in num_bands:
                    for seed in seeds:
                        if measure == 'js':
                            job = pool.apply_async(run_experiment, rating_matrix_js, measure, num_hash, num_band, seed, timeout)
                        elif measure == 'cs':
                            job = pool.apply_async(run_experiment, rating_matrix_cs, measure, num_projection, num_band, seed, timeout)
                        elif measure == 'dcs':
                            job = pool.apply_async(run_experiment, rating_matrix_dcs, measure, num_projection, num_band, seed, timeout)
                        jobs.append(job)

        # Wait for all jobs to complete
        pool.close()
        pool.join()

        # Collect results
        results = [job.get() for job in jobs if job.get() is not None]

    print('All experiments are done!')

    # Stop timer
    end = time.time()
    print("Total time elapsed: ", end - start)

if __name__ == "__main__":
    main()
