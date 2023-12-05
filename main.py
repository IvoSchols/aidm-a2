from collections import defaultdict
import time
import numpy as np
from scipy import sparse
import argparse
from sklearn.random_projection import SparseRandomProjection
from itertools import combinations

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

# Load data from file, return a sparse matrix of shape (movies, users)
def load_data(file_path : str, similarity_measure : str):
    # Ratings is a numpy array of shape (n_ratings, 3): [user_id, movie_id, rating]
    ratings_data = np.load(file_path) 
    
    # Create a column based sparse matrix of shape (movies, users) from the ratings
    users = ratings_data[:,0]
    movies = ratings_data[:,1]
    ratings = ratings_data[:,2]
    
    # Replace all ratings with 1
    ratings = np.ones(len(ratings))
    if similarity_measure == 'js':
        rating_matrix = sparse.csr_matrix((ratings, (movies, users)), dtype='?')
    elif similarity_measure == 'cs' or similarity_measure == 'dcs':
        rating_matrix = sparse.csc_matrix((ratings, (movies, users)), dtype='?')

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
@profile
def minhash_jaccard(rating_matrix, n_hashes : int):
    n_movies, n_users = rating_matrix.shape

    # Generate signature matrix with all values set to infinity
    signature_matrix = np.full((n_hashes,n_users),np.inf)

    for i in range(n_hashes):
        # Generate a random permutation of the rows and swap them
        swapped_matrix = rating_matrix[np.random.permutation(n_movies), :]

        # Find the index of the first True value for each user
        min_hashes = np.argmax(swapped_matrix > 0, axis=0).astype(int)

        # Store the minimum hash for each user
        signature_matrix[i, :] = np.minimum(min_hashes, signature_matrix[i, :])

    return signature_matrix.astype(int)

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
@profile
def lsh_jaccard(signature_matrix : np.ndarray, n_bands : int):
    n_hashes, n_users = signature_matrix.shape
    rows_per_band = n_hashes // n_bands

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

        for user_bucket in bucket_users_dict.values():
            # # Iterate through each pair of users in the bucket
            for user1, user2 in combinations(user_bucket, 2):
                user1_sig = signature_matrix[:, user1]
                user2_sig = signature_matrix[:, user2]

                # Add pair to the set if above the threshold -> calculation is done inline for performance reasons
                intersection_size = np.sum(user1_sig & user2_sig)
                union_size = np.sum(user1_sig | user2_sig)
                similarity = intersection_size / union_size
                if similarity > 0.5:
                    # print(similarity)
                    similar_users.append((user1,user2))

        print("Number of similar users in band: ", len(similar_users))
        append_result(similar_users, "js.txt")


# Return candidate pairs of shape (n_candidate_pairs, 2) by applying LSH to the given rating matrix
# We use random projection to reduce the dimensionality of the rating matrix
# and then bin the users into buckets based on their hash
#   - cosine_similarity > 0.73
#   - discrete_cosine_similarity > 0.73
def lsh_cosine(projected_matrix, similarity_function, n_bands):

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
                similarity = similarity_function(projected_matrix[user1], projected_matrix[user2])

                if similarity > 0.73:
                    similar_users.append((user1, user2))
        
        if similarity_function == cosine_similarity:
            append_result(similar_users, "cs.txt")
        elif similarity_function == discrete_cosine_similarity:
            append_result(similar_users, "dcs.txt")


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

    rating_matrix = load_data(directory)

    if similarity_measure == 'js':
        signature_matrix = minhash_jaccard(rating_matrix, n_hashes)
    elif similarity_measure == 'cs' or similarity_measure == 'dcs':
        projection_matrix = SparseRandomProjection(n_components=n_projections, dense_output=False, random_state=seed)
        # Project the rating matrix onto a lower dimensional space
        projected_matrix = projection_matrix.fit_transform(rating_matrix)

    del rating_matrix # free memory
    # Stop timer
    stop = time.time()
    print("Time elapsed for minhash/projection matrix: ", stop - start)

    # Compute LSH
    if similarity_measure == 'js':
        lsh_jaccard(signature_matrix, n_bands)
    elif similarity_measure == 'cs':
        lsh_cosine(projected_matrix, cosine_similarity, n_bands)
    elif similarity_measure == 'dcs':
        lsh_cosine(projected_matrix, discrete_cosine_similarity, n_bands)

    # Stop timer
    end = time.time()
    print("Total time elapsed: ", end - start)

if __name__ == "__main__":
    main()
