from collections import defaultdict
import time
import numpy as np
from scipy import sparse
import os
import argparse
from sklearn.random_projection import SparseRandomProjection
from itertools import combinations

# List of prime numbers to be used for hashing -> prime numbers are used to reduce the number of collisions
# We know that max(user_id) = 103704 -> we need prime numbers up to 103704
# https://prime-numbers.info/list/first-100-primes
prime_numbers = [7417,7433,7451,7457,7459,7459,7477,7481,7487,7489,7499,7507,7517,7523,7529,7537,7541,7547,7549,7559,7561,7573,7577,7583,7589,7591,7603,7607,7621,7639,7643,7649,7669,7673,7681,7687,7691,7699,7703,7717,7723,7727,7741,7753,7757,7759,7789,7793,7817,7823,7829,7841,7853,7867,7873,7877,7879,7883,7901,7907,7919]
# Return hash h(x, a, b, c, n_buckets) = ((ax + b) % c) % n_buckets that maps integer x to a bucket
# x: input value
# a, b, c: parameters of the hash function
# n_buckets: number of buckets
def hash_function_jaccard(x : np.ndarray, a : int, b : int, c : int, n_buckets : int):
    return ((a*x + b) % c) % n_buckets

# Returns hashed vector with values in {-1, 1}
def hash_vector(vec):
    return tuple(1 if bit > 0 else -1 for bit in vec)

###
# Parsing Args, Reading & Writing Data
###

def parse_args():
    argparser = argparse.ArgumentParser()
    # argparser.add_argument("-d", default="data/user_movie_rating_subset.npy", help="specify data file path")
    argparser.add_argument("-d", default="data/user_movie_rating.npy", help="specify data file path")
    argparser.add_argument("-s", default=42, type=int, help="the random seed to be used")
    argparser.add_argument("-m", choices = ['js','cs','dcs'], help="similarity measure: jacard (js), cosine (cs), discrete cosine (dcs)")
    args = argparser.parse_args()
    return args

# Load data from file and transform it into a numpy array
def load_data_jaccard(file_path : str):
    # Ratings is a numpy array of shape (n_ratings, 3): [user_id, movie_id, rating]
    ratings_data = np.load(file_path) 
    
    # Create a column (user) based sparse matrix of shape (users, movies) from the ratings
    users = ratings_data[:,0]
    movies = ratings_data[:,1]
    ratings = ratings_data[:,2]
    
    rating_matrix = sparse.csc_matrix((ratings, (movies, users)))
    
    return rating_matrix

# Load data from file and transform it into a numpy array
def load_data_cosine(file_path : str):
    # Ratings is a numpy array of shape (n_ratings, 3): [user_id, movie_id, rating]
    ratings_data = np.load(file_path) 
    
    # Create a column (user) based sparse matrix of shape (users, movies) from the ratings
    users = ratings_data[:,0]
    movies = ratings_data[:,1]
    ratings = ratings_data[:,2]
    
    rating_matrix = sparse.csr_matrix((ratings, (users, movies)))
    
    return rating_matrix

# Append candidate pairs to file
def append_result(candidate_pairs : list, file_name : str):
    with open(file_name, 'a') as f:
        for (user1,user2) in candidate_pairs:
            f.write("%s,%s\n" % (user1, user2))



##
# MinHash
##

# Takes a rating matrix of shape (n_users, n_movies) and a hash function and returns the minimum hash value for each user
def minhash_jaccard(rating_matrix, n_hashes : int):
    n_movies= rating_matrix.shape[0]
    n_users = rating_matrix.shape[1]

    # generate hash functions by representing them as the coefficients a,b of the hash function h(x,a,b,c,n_buckets)
    # c is a prime number and n_buckets is the number of buckets
    hash_functions = [[np.random.randint(1,1000),np.random.randint(1,1000),prime_numbers[i]] for i in range(n_hashes)]

    # Generate signature matrix with all values set to infinity
    signature_matrix = np.full((n_hashes,n_users),np.inf)

    # Find indices of non-zero entries in the rating matrix
    non_zero_indices = rating_matrix.nonzero()

    # Iterate through each hash function and update the signature matrix
    for i in range(n_hashes):
        # Calculate hash values for all non-zero movies
        hash_values = hash_function_jaccard(non_zero_indices[1], hash_functions[i][0],
                                            hash_functions[i][1], hash_functions[i][2], n_movies)
        # Update the signature matrix with the minimum hash value for each user
        signature_matrix[i, non_zero_indices[1]] = np.minimum(signature_matrix[i, non_zero_indices[1]], hash_values)

    return signature_matrix

##
# Similarity Measures
##

# Return the Jaccard similarity of two sets x and y (this is done inline after measuring performance)
def jaccard_similarity(x : np.ndarray, y : np.ndarray):
    intersection_size = np.sum(np.isin(x,y))
    union_size = len(x) + len(y) - intersection_size
    return intersection_size / union_size

# Return the cosine similarity of two vectors x and y
def cosine_similarity(x : np.ndarray, y : np.ndarray):
    norm_x, norm_y = np.linalg.norm(x), np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0
    theta = np.arccos(np.dot(x,y.T) / (norm_x * norm_y))
    return 1- theta/180

# Return the discrete cosine similarity of two vectors x and y
def discrete_cosine_similarity(x : np.ndarray, y : np.ndarray):
    x = np.sign(x)
    y = np.sign(y)
    norm_x, norm_y = np.linalg.norm(x), np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0
    theta = np.arccos(np.dot(x,y.T) / (norm_x * norm_y))
    return 1- theta/180

##
# LSH
##

# Return candidate pairs of shape (n_candidate_pairs, 2) from the given signature matrix
# Divide the signature matrix into n_bands bands and n_rows rows per band
# If two users are similar, that is:
#   - jaccard_similarity > 0.5
def lsh_jaccard(signature_matrix : np.ndarray, n_bands : int):
    n_hashes, n_users = signature_matrix.shape
    rows_per_band = n_hashes // n_bands

    # Generate hash functions for each band
    hash_functions = [[np.random.randint(1,1000),np.random.randint(1,1000),np.random.choice(prime_numbers)] for _ in range(n_bands)]
    n_buckets = n_users // 2


    # Apply LSH to find candidate pairs
    for band in range(n_bands):
        # Extract a band from the signature matrix
        band_matrix = signature_matrix[band * rows_per_band: (band + 1) * rows_per_band, :]

        # Collapse the rows of the band into a single row for each user and calculate its destination bucket
        dest_bucket = hash_function_jaccard(np.sum(band_matrix, axis=0), hash_functions[band][0], hash_functions[band][1], hash_functions[band][2], n_buckets)

        # Find unique buckets and map each user to its bucket
        buckets, bucket_indices = np.unique(dest_bucket, return_inverse=True)

        similar_users = list()

        for bucket_index in range(len(buckets)):
            bucket_users = np.argwhere(bucket_indices == bucket_index).flatten()

            # Iterate through each pair of users in the bucket
            for user1, user2 in combinations(bucket_users, 2):
                user1_sig = signature_matrix[:, user1]
                user2_sig = signature_matrix[:, user2]

                # Add pair to the set if above the threshold -> calculation is done inline for performance reasons
                intersection_size = np.sum(np.isin(user1_sig,user2_sig))
                union_size = n_hashes + n_hashes - intersection_size # n_hashes = len(user1_sig) = len(user2_sig)
                if intersection_size / union_size > 0.5:
                    similar_users.append((user1,user2))
        
        append_result(similar_users, "js.txt")


# Return candidate pairs of shape (n_candidate_pairs, 2) by applying LSH to the given rating matrix
# We use random projection to reduce the dimensionality of the rating matrix
# and then bin the users into buckets based on their hash
#   - cosine_similarity > 0.73
#   - discrete_cosine_similarity > 0.73
def lsh_cosine(rating_matrix, projection_matrix: SparseRandomProjection, similarity_function, n_bands):
    # Project the rating matrix onto a lower dimensional space
    projected_matrix = projection_matrix.fit_transform(rating_matrix)
    # Hash the projected vectors giving hashed_vectors of shape (users, hash)
    hashed_vectors = np.array([hash_vector(vec) for vec in projected_matrix.A])


    columns_per_band = len(hashed_vectors[0]) // n_bands
    # Divide the hashed vectors into n_bands bands and n_rows rows per band
    for band in range(n_bands):
        #Extract a band from the vector
        band_matrix = hashed_vectors[:, band * columns_per_band: (band + 1) * columns_per_band]

        user_buckets = defaultdict(list)

        # Put the users into buckets based on their hash
        for user, hash in enumerate(band_matrix):
            user_buckets[np.sum(hash)].append(user)

        similar_users = list()

        # Iterate through each bucket and find similar users
        for user_bucket in user_buckets.values():
            # Iterate through each pair of users in the bucket
            for user1, user2 in combinations(user_bucket, 2):
                similarity = similarity_function(rating_matrix[user1].toarray(), rating_matrix[user2].toarray())

                if similarity > 0.73:
                    similar_users.append((user1, user2))
        
        if similarity_function == cosine_similarity:
            append_result(similar_users, "cs.txt")
        elif similarity_function == discrete_cosine_similarity:
            append_result(similar_users, "cs.txt")


def main():
    args = parse_args()
    directory = args.d
    seed = args.s
    similarity_measure = args.m

    # TODO: remove
    if similarity_measure is None:
        similarity_measure = 'cs'

    if similarity_measure != 'js' and similarity_measure != 'cs' and similarity_measure != 'dcs':
        raise Exception("Unknown similarity measure")

    np.random.seed(seed)

    # load data

    # Start timer
    start = time.time()

    # Compute minhash / random projection
    num_projections = 50 # TODO: DO tune this parameter!
    n_hashes = 25     # Can also be tuned -> higher values lead to better results but take longer to compute

    if similarity_measure == 'js':
        rating_matrix = load_data_jaccard(directory)
        signature_matrix = minhash_jaccard(rating_matrix, n_hashes)
        del rating_matrix # free memory
    elif similarity_measure == 'cs' or similarity_measure == 'dcs':
        rating_matrix = load_data_cosine(directory)
        projection_matrix = SparseRandomProjection(n_components=num_projections, dense_output=False, random_state=seed)

    # Stop timer
    stop = time.time()
    print("Time elapsed for minhash/projection matrix: ", stop - start)

    # Compute LSH
    n_bands = 3 # TODO: DO tune this parameter!

    if similarity_measure == 'js':
        lsh_jaccard(signature_matrix, n_bands)
        del signature_matrix # free memory
    elif similarity_measure == 'cs':
        lsh_cosine(rating_matrix, projection_matrix, cosine_similarity, n_bands)
        del rating_matrix # free memory
    elif similarity_measure == 'dcs':
        lsh_cosine(rating_matrix, projection_matrix, discrete_cosine_similarity, n_bands)
        del rating_matrix # free memory

    # Stop timer
    end = time.time()
    print("Total time elapsed: ", end - start)

if __name__ == "__main__":
    main()
