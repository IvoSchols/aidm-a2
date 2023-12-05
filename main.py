from collections import defaultdict
import time
import numpy as np
from scipy import sparse
import os
import argparse
from sklearn.random_projection import SparseRandomProjection
from itertools import combinations

# List of prime numbers to be used for hashing -> prime numbers are used to reduce the number of collisions
# We know that max(user_id) = 103704 -> take 100 random prime numbers till 103704
# prime_numbers= [102259, 62939, 12097, 86501, 23279, 51151, 34267, 44621, 34607, 5807, 29569, 24691, 47237, 49757, 38011, 64033, 55171, 22273, 30059, 42797, 4591, 54907, 10781, 88657, 30557, 54037, 13309, 53197, 87877, 82393, 97, 14867, 41603, 15053, 30643, 56311, 89917, 1889, 99809, 48397, 85159, 34033, 46723, 19, 45979, 9769, 68897, 98711, 84499, 55163, 5507, 91967, 52561, 86131, 27967, 63599, 4801, 68539, 9533, 100019, 27077, 93407, 41621, 1481, 83737, 191, 88079, 90011, 50497, 41953, 102793, 8609, 97259, 457, 41669, 94951, 11161, 46993, 59981, 66403, 39301, 40879, 62213, 29303, 83243, 31139, 26927, 317, 87767, 8009, 86197, 37571, 67219, 40751, 15289, 54539, 44617, 24203, 98909, 94463]
# https://prime-numbers.info/
# prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541]
prime_numbers = [29, 1301, 1237, 659, 859, 937, 829, 281, 1289, 311, 127, 1399, 59, 1297, 673, 1493, 593, 359, 431, 751, 719, 1249, 911, 1531, 269, 643, 1381, 877, 1171, 701, 1069, 683, 1453, 499, 521, 941, 797, 773, 1277, 1223, 487, 1061, 691, 1153, 149, 109, 631, 1319, 97, 523, 1321, 971, 1483, 1151, 137, 977, 181, 379, 409, 179, 1481, 947, 1423, 641, 367, 467, 757, 373, 1097, 1213, 251, 139, 1471, 1093, 239, 1031, 443, 1229, 1217, 919, 1123, 811, 73, 1103, 107, 907, 227, 1433, 839, 101, 1013, 383, 1543, 419, 47, 241, 653, 967, 173, 233]
#
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
    
    # Mandatory
    argparser.add_argument("-d", default="data/user_movie_rating.npy", help="specify data file path")
    argparser.add_argument("-s", default=42, type=int, help="the random seed to be used")
    argparser.add_argument("-m", choices = ['js','cs','dcs'], help="similarity measure: jacard (js), cosine (cs), discrete cosine (dcs)")

    # Tunable
    argparser.add_argument("-n_hashes", default=100, type=int, help="number of hash functions to be used (only for jaccard&max 100)")
    argparser.add_argument("-n_projections", default=100, type=int, help="number of projections to be used")
    argparser.add_argument("-n_bands", default=5, type=int, help="number of bands to be used")

    args = argparser.parse_args()

    # TODO: remove
    args.m = 'js'

    if args.m != 'js' and args.m != 'cs' and args.m != 'dcs':
        raise Exception("Unknown similarity measure")


    return args

# Load data from file and transform it into a numpy array
def load_data_jaccard(file_path : str):
    # Ratings is a numpy array of shape (n_ratings, 3): [user_id, movie_id, rating]
    ratings_data = np.load(file_path) 
    
    # Create a column (user) based sparse matrix of shape (users, movies) from the ratings
    users = ratings_data[:,0]
    movies = ratings_data[:,1]
    ratings = ratings_data[:,2]
    
    rating_matrix = sparse.csr_matrix((ratings, (movies, users)))
    
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

# Takes a rating matrix of shape (n_movies, n_users) and returns a signature matrix of shape (n_hashes, n_users)
# @profile
def minhash_jaccard(rating_matrix, n_hashes : int):
    n_movies, n_users = rating_matrix.shape

    # Generate signature matrix with all values set to infinity
    signature_matrix = np.full((n_hashes,n_users),np.inf)

    for i in range(n_hashes):
        # Generate a random permutation of the rows and swap them
        swapped_matrix = rating_matrix[np.random.permutation(n_movies), :]

        # Find the index of the first non-zero value for each column/user -> by transforming the matrix into a boolean matrix and finding the first True indice
        min_hashes = np.argmax(swapped_matrix != 0, axis=0)

        # Store the minimum hash for each user
        signature_matrix[i, :] = np.minimum(min_hashes, signature_matrix[i, :])

    return signature_matrix

##
# Similarity Measures
##

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
# @profile
def lsh_jaccard(signature_matrix : np.ndarray, n_bands : int):
    n_hashes, n_users = signature_matrix.shape
    rows_per_band = n_hashes // n_bands

    # Generate hash functions for each band
    # hash_functions = [[np.random.randint(1,1000),np.random.randint(1,1000),np.random.choice(prime_numbers)] for _ in range(n_bands)]
    n_buckets = n_users // 35


    # Apply LSH to find candidate pairs
    for band in range(n_bands):
        # Extract a band from the signature matrix
        band_matrix = signature_matrix[band * rows_per_band: (band + 1) * rows_per_band, :]

        # Collapse the rows of the band into a single row for each user and calculate its destination bucket
        dest_bucket = np.sum(band_matrix, axis=0) #% n_buckets

        # Find unique buckets and map each user to its bucket
        buckets, bucket_indices = np.unique(dest_bucket, return_inverse=True)
        bucket_users_dict = defaultdict(list)
        for bucket_index in range(len(buckets)):
            bucket_users_dict[bucket_index] = np.argwhere(bucket_indices == bucket_index).flatten()

        similar_users = list()

        # # Iterate through each pair of users in the bucket
        for user1, user2 in combinations(bucket_users_dict.values(), 2):
            user1_sig = signature_matrix[:, user1]
            user2_sig = signature_matrix[:, user2]

            # Add pair to the set if above the threshold -> calculation is done inline for performance reasons
            intersection_size = np.intersect1d(user1_sig, user2_sig).size
            union_size = np.union1d(user1_sig, user2_sig).size
            similarity = intersection_size / union_size
            if similarity > 0.5:
                # print(similarity)
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
    n_hashes = args.n_hashes
    n_bands = args.n_bands
    n_projections = args.n_projections

    np.random.seed(seed)

    # load data

    # Start timer
    start = time.time()

    if similarity_measure == 'js':
        rating_matrix = load_data_jaccard(directory)
        signature_matrix = minhash_jaccard(rating_matrix, n_hashes)
        del rating_matrix # free memory
    elif similarity_measure == 'cs' or similarity_measure == 'dcs':
        rating_matrix = load_data_cosine(directory)
        projection_matrix = SparseRandomProjection(n_components=n_projections, dense_output=False, random_state=seed)

    # Stop timer
    stop = time.time()
    print("Time elapsed for minhash/projection matrix: ", stop - start)

    # Compute LSH

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
