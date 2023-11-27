from collections import defaultdict
import time
import numpy as np
from scipy import sparse
import os
import argparse

# List of prime numbers to be used for hashing -> prime numbers are used to reduce the number of collisions
# https://prime-numbers.info/list/first-100-primes
prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541]

# Return hash h(x, a, b, c, n_buckets) = ((ax + b) % c) % n_buckets that maps integer x to a bucket
# x: input value
# a, b, c: parameters of the hash function
# n_buckets: number of buckets
def hash_function_jaccard(x : np.ndarray, a : int, b : int, c : int, n_buckets : int):
    return ((a*x + b) % c) % n_buckets

# Return hash h(x, a, b, c, n_buckets) = ((ax + b) % c) % n_buckets that maps integer x to a bucket
def hash_function_cosine(x : np.ndarray, a : int, b : int, n_buckets : int):
    return np.floor((a*x + b) / n_buckets)


###
# Parsing Args, Reading & Writing Data
###

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", default="data/user_movie_rating_subset.npy", help="specify data file path")
    # argparser.add_argument("-d", default="data/user_movie_rating.npy", help="specify data file path")
    argparser.add_argument("-s", default=42, type=int, help="the random seed to be used")
    argparser.add_argument("-m", choices = ['js','cs','dcs'], help="similarity measure: jacard (js), cosine (cs), discrete cosine (dcs)")
    args = argparser.parse_args()
    return args

# Load data from file and transform it into a numpy array
def load_data(file_path : str):
    # Ratings is a numpy array of shape (n_ratings, 3): [user_id, movie_id, rating]
    ratings_data = np.load(file_path) 
    
    # Create a column (user) based sparse matrix of shape (n_users, n_movies) from the ratings
    users = ratings_data[:,0]
    movies = ratings_data[:,1]
    ratings = ratings_data[:,2]

    n_users = np.max(users)
    n_movies = np.max(movies)
    
    rating_matrix = sparse.csc_matrix((ratings, (users, movies)), shape=(n_users+1, n_movies+1))
    
    return rating_matrix

# Write candidate pairs to file
def write_result(candidate_pairs : list, file_name : str):
    with open(file_name, 'w') as f:
        for (user1,user2) in candidate_pairs:
            f.write("%s,%s\n" % (user1, user2))
    f.close()
    


##
# MinHash
##

# Takes a rating matrix of shape (n_users, n_movies) and a hash function and returns the minimum hash value for each user
def minhash_jaccard(rating_matrix : sparse.csc_matrix, n_hashes : int):
    n_users = rating_matrix.shape[0]
    n_movies = rating_matrix.shape[1]

    # generate hash functions by representing them as the coefficients a,b of the hash function h(x,a,b,c,n_buckets)
    # c is a prime number and n_buckets is the number of buckets
    hash_functions = [[np.random.randint(1,1000),np.random.randint(1,1000),prime_numbers[i]] for i in range(n_hashes)]
    
    # Generate signature matrix with all values set to infinity
    signature_matrix = np.full((n_hashes,n_users),np.inf)

    # Find indices of non-zero entries in the rating matrix
    non_zero_indices = rating_matrix.nonzero()

    # Iterate through each hash function and update the signature matrix
    for i in range(n_hashes):
        # Calculate hash values for all non-zero columns in the rating matrix
        hash_values = hash_function_jaccard(non_zero_indices[1], hash_functions[i][0], hash_functions[i][1],
                                    hash_functions[i][2], n_movies)
        
        np.minimum.at(signature_matrix[i], non_zero_indices[0], hash_values)


    return signature_matrix


def minhash_cosine(rating_matrix : sparse.csc_matrix, n_hashes : int):
    n_users = rating_matrix.shape[0]
    n_movies = rating_matrix.shape[1]

    hash_functions = [[np.random.randint(1, 1000), np.random.randint(1, 1000)] for _ in range(n_hashes)]

    signature_matrix = np.full((n_hashes, n_users), np.inf)

    non_zero_indices = rating_matrix.nonzero()

    for i in range(n_hashes):
        hash_values = hash_function_cosine(non_zero_indices[1], hash_functions[i][0], hash_functions[i][1], n_movies)
        np.minimum.at(signature_matrix[i], non_zero_indices[0], hash_values)

    return signature_matrix
  

##
# Similarity Measures
##

# Return the Jaccard similarity of two sets x and y
def jaccard_similarity(x : np.ndarray, y : np.ndarray):
    intersection = np.intersect1d(x,y)
    union = np.union1d(x,y)
    return len(intersection) / len(union)

# Return the cosine similarity of two vectors x and y
def cosine_similarity(x : np.ndarray, y : np.ndarray):
    return np.dot(x.T,y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Return the discrete cosine similarity of two vectors x and y
def discrete_cosine_similarity(x : np.ndarray, y : np.ndarray):
    x = np.sign(x)
    y = np.sign(y)
    return np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))

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

    similar_users = set()

    # Apply LSH to find candidate pairs
    for band in range(n_bands):
        # Extract a band from the signature matrix
        band_matrix = signature_matrix[band * rows_per_band: (band + 1) * rows_per_band, :]

        # Collapse the rows of the band into a single row for each user and calculate its destination bucket
        dest_bucket = hash_function_jaccard(np.sum(band_matrix, axis=0), hash_functions[band][0], hash_functions[band][1], hash_functions[band][2], n_buckets)

        # Find unique buckets and map each user to its bucket
        buckets, bucket_indices = np.unique(dest_bucket, return_inverse=True)

        for bucket_index in range(len(buckets)):
            bucket_users = np.argwhere(bucket_indices == bucket_index).flatten()

            for i in range(len(bucket_users)):
                user1 = bucket_users[i]
                user1_sig = signature_matrix[:, user1]

                for j in range(i + 1, len(bucket_users)):
                    user2 = bucket_users[j]
                    user2_sig = signature_matrix[:, user2]

                    # Add pair to the set if above the threshold
                    if jaccard_similarity(user1_sig, user2_sig) > 0.5:
                        similar_users.add((user1,user2))



    return similar_users

# Return candidate pairs of shape (n_candidate_pairs, 2) from the given signature matrix
#   - cosine_similarity > 0.73
#   - discrete_cosine_similarity > 0.73
def lsh_cosine(signature_matrix : np.ndarray, n_bands : int, similarity_function):
    n_hashes, n_users = signature_matrix.shape
    rows_per_band = n_hashes // n_bands


    similar_users = set()

    for band in range(n_bands):
        band_matrix = signature_matrix[band * rows_per_band: (band + 1) * rows_per_band, :]

        # Find similarities between all users in the band
        similarities = similarity_function(band_matrix, band_matrix)

        # Convert similarities to distances because LSH is typically used with distance measures
        distances = 1 - similarities

        # Use distances to determine destination buckets
        dest_bucket = np.argmin(distances, axis=1)

        buckets, bucket_indices = np.unique(dest_bucket, return_inverse=True)

        for bucket_index in range(len(buckets)):
            bucket_users = np.argwhere(bucket_indices == bucket_index).flatten()

            for i in range(len(bucket_users)):
                user1 = bucket_users[i]
                user1_sig = signature_matrix[:, user1]

                for j in range(i + 1, len(bucket_users)):
                    user2 = bucket_users[j]
                    user2_sig = signature_matrix[:, user2]

                    # Add pair to the set if below the threshold
                    if similarity_function(user1_sig, user2_sig) > 0.73:
                        similar_users.add((user1, user2))

    return similar_users


def main():
    args = parse_args()
    directory = args.d
    seed = args.s
    similarity_measure = args.m

    # TODO: remove
    if similarity_measure is None:
        similarity_measure = 'js'

    if similarity_measure != 'js' and similarity_measure != 'cs' and similarity_measure != 'dcs':
        raise Exception("Unknown similarity measure")

    np.random.seed(seed)
    
    # load data
    rating_matrix = load_data(directory)

    # Start timer
    start = time.time()

    # Compute minhash
    n_hashes = 100

    if similarity_measure == 'js':
        signature_matrix = minhash_jaccard(rating_matrix, n_hashes)
    elif similarity_measure == 'cs' or similarity_measure == 'dcs':
        signature_matrix = minhash_cosine(rating_matrix, n_hashes)

    # Stop timer
    stop = time.time()
    print("Time elapsed for minhash: ", stop - start)

    del rating_matrix # free memory

    # Compute LSH
    n_bands = 20

    if similarity_measure == 'js':
        candidate_pairs = lsh_jaccard(signature_matrix, n_bands)
    elif similarity_measure == 'cs':
        candidate_pairs = lsh_cosine(signature_matrix, n_bands, cosine_similarity)
    elif similarity_measure == 'dcs':
        candidate_pairs = lsh_cosine(signature_matrix, n_bands, discrete_cosine_similarity)

    del signature_matrix # free memory

    # Stop timer
    end = time.time()
    print("Total time elapsed: ", end - start)

    # write results to file corresponding to the similarity measure
    write_result(candidate_pairs, similarity_measure+".txt")
    



    print(candidate_pairs)

if __name__ == "__main__":
    main()
