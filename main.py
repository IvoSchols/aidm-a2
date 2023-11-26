import time
import numpy as np
from scipy import sparse
import os
import argparse

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
def load_data(file_path : str):
    # Ratings is a numpy array of shape (n_ratings, 3): [user_id, movie_id, rating]
    ratings_data = np.load(file_path) 
    
    # Create a column (user) based sparse matrix of shape (n_users, n_movies) from the ratings
    users = ratings_data[:,0]
    movies = ratings_data[:,1]
    ratings = ratings_data[:,2]

    n_users = np.max(users)
    n_movies = np.max(movies)
    
    rating_matrix = sparse.csr_matrix((ratings, (users, movies)), shape=(n_users+1, n_movies+1))
    
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

# Return hash h(x, a, b, c, n_buckets) = ((ax + b) % c) % n_buckets that maps integer x to a bucket
# x: input value
# a, b, c: parameters of the hash function
# n_buckets: number of buckets
def hash_function(x : int, a : int, b : int, c : int, n_buckets : int):
    return ((a*x + b) % c) % n_buckets

# Takes a rating matrix of shape (n_users, n_movies) and a hash function and returns the minimum hash value for each user
def minhash(rating_matrix : sparse.csr_matrix, n_hashes : int):
    n_users = rating_matrix.shape[0]
    n_movies = rating_matrix.shape[1]

    # generate hash functions by representing them as the coefficients a,b,c of the hash function h(x,a,b,c,n_buckets)
    hash_functions = np.random.randint(1,1000,(n_hashes,3))
    
    # generate signature matrix
    signature_matrix = np.full((n_hashes,n_users),np.inf)

    # Update signature matrix for each user in the rating matrix using the hash functions
    # Start with user 1 and iterate over all users
    for user in range(n_users):
        # Find indices of movies that the user has rated
        non_zero_indices = rating_matrix[user].nonzero()[1]

        if len(non_zero_indices) == 0:
            continue

        # Calculate hash value for each hash function for each movie that the user has rated and keep the minimum hash value
        minhash_values = np.array([min(hash_function(non_zero_indices,hash_functions[i][0],hash_functions[i][1],hash_functions[i][2],n_movies)) for i in range(n_hashes)])
        # Update signature matrix
        signature_matrix[:,user] = np.minimum(minhash_values, signature_matrix[:,user])

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
    return np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))

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
#   - cosine_similarity > 0.73
#   - discrete_cosine_similarity > 0.73
def lsh(signature_matrix : np.ndarray, n_bands : int, similarity_function, threshold: float):
    n_hashes, n_users = signature_matrix.shape
    rows_per_band = n_hashes // n_bands    

    # Generate hash functions for each band
    hash_functions = np.random.randint(1,1000,(n_bands,3))
    n_buckets = n_users // 2


    # Initialize a dictionary to store candidate pairs
    candidate_pairs = {}

    # Apply LSH to find candidate pairs
    for band in range(n_bands):
        # Extract a band from the signature matrix
        band_matrix = signature_matrix[band * rows_per_band: (band + 1) * rows_per_band, :]

        # Collapse the rows of the band into a single row for each user
        band_value = np.sum(band_matrix, axis=0)

        # Calculate destination bucket for each user in the band
        hash_values = hash_function(band_value, hash_functions[band][0], hash_functions[band][1], hash_functions[band][2], n_buckets)

        # Map each user to its bucket
        for user in range(n_users):
            hash_value = hash_values[user]
            # Check if the hash value already exists in the dictionary
            if hash_value in candidate_pairs:
                # If it does, add the current user to the candidate list
                candidate_pairs[hash_value].append(user)
            else:
                # If not, create a new entry in the dictionary
                candidate_pairs[hash_value] = [user]

    # Iterate through each bucket and find candidate pairs
    similar_users = set()

    for bucket in candidate_pairs.values():
        # With fewer than 2 users in the bucket, there are no candidate pairs
        if len(bucket) < 2:
            continue

        # Generate all possible pairs of users in the bucket
        for i in range(len(bucket)):
            user1 = bucket[i]
            user1_sig = signature_matrix[:,user1]

            for j in range(i+1, len(bucket)):
                user2 = bucket[j]
                user2_sig = signature_matrix[:,user2]

                similarity = similarity_function(user1_sig, user2_sig)

                if similarity > threshold:
                    similar_users.add((user1, user2))

    return similar_users



def main():
    args = parse_args()
    directory = args.d
    seed = args.s
    similarity_measure = args.m

    # TODO: remove
    similarity_measure = 'js'

    if similarity_measure != 'js' and similarity_measure != 'cs' and similarity_measure != 'dcs':
        raise Exception("Unknown similarity measure")

    if similarity_measure == 'js':
        similarity_function = jaccard_similarity
        threshold = 0.5
    elif similarity_measure == 'cs':
        similarity_function = cosine_similarity
        threshold = 0.73
    elif similarity_measure == 'dcs':
        similarity_function = discrete_cosine_similarity
        threshold = 0.73

    np.random.seed(seed)
    
    # load data
    rating_matrix = load_data(directory)

    # Start timer
    start = time.time()

    # compute minhash
    n_hashes = 100
    signature_matrix = minhash(rating_matrix, n_hashes)

    del rating_matrix

    # compute LSH
    n_bands = 20
    candidate_pairs = lsh(signature_matrix, n_bands, similarity_function, threshold)

    del signature_matrix

    # Stop timer
    end = time.time()
    print("Time elapsed: ", end - start)

    # write results to file corresponding to the similarity measure
    write_result(candidate_pairs, similarity_measure+".txt")
    



    print(candidate_pairs)

if __name__ == "__main__":
    main()
