import numpy as np
from scipy import sparse
import os
import argparse

###
# Parsing Args, Reading & Writing Data
###

def parse_args():
    argparser = argparse.ArgumentParser()
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
    # TODO: Evaluate coo matrix, csc matrix, csr matrix and lil matrix
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

# Return hash h(x, a, b, c, n_buckets) = ((ax + b) % c) % n_buckets that maps integer x to a bucket
# x: input value
# a, b, c: parameters of the hash function
# n_buckets: number of buckets
def hash_function(x : int, a : int, b : int, c : int, n_buckets : int):
    return ((a*x + b) % c) % n_buckets

# Return signature matrix of shape (n_hashes, n_users) for the given rating matrix
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

        if len(non_zero_indices) == 0: # TODO: check if this is correct -> Is such a user even present or is it error in the coding?
            continue

        # Calculate hash value for each hash function for each movie that the user has rated and keep the minimum hash value
        minhash_values = np.array([min(hash_function(non_zero_indices,hash_functions[i][0],hash_functions[i][1],hash_functions[i][2],n_movies)) for i in range(n_hashes)])
        # Update signature matrix
        signature_matrix[:,user-1] = np.minimum(minhash_values, signature_matrix[:,user-1])

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
def lsh(rating_matrix, signature_matrix : np.ndarray, n_bands : int, n_buckets : int, similarity_measure : str):
    n_hashes_sig = signature_matrix.shape[0]
    n_users = signature_matrix.shape[1]
    n_rows = int(n_hashes_sig / n_bands)    

    # generate hash functions by representing them as the coefficients a,b,c of the hash function h(x,a,b,c,n_buckets)
    hash_functions = np.random.randint(1,1000,(n_buckets,3))
    
    # generate buckets
    buckets = dict()
    for band in range(n_bands):
        buckets[band] = dict()

    # hash each user's signature into buckets
    for user in range(n_users):
        for band in range(n_bands):
            # extract the corresponding rows for the current band
            band_signature = signature_matrix[band * n_rows: (band + 1) * n_rows, user]

            # compute the hash values for the current band
            hash_values = np.array([hash_function(band_signature,hash_functions[i][0],hash_functions[i][1],hash_functions[i][2],n_buckets) for i in range(n_buckets)])

            # add the user to the corresponding bucket
            for hash_value in hash_values:
                if hash_value not in buckets[band]:
                    buckets[band][hash_value] = []
                buckets[band][hash_value].append(user)
            
    # find candidate pairs by comparing users by their ratings
    candidate_pairs = set()

    for band in range(n_bands):
        for bucket in buckets[band].values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i+1,len(bucket)):
                        user1 = bucket[i]
                        user2 = bucket[j]
                        if similarity_measure == 'js':
                            similarity = jaccard_similarity(rating_matrix[user1], rating_matrix[user2])
                        elif similarity_measure == 'cs':
                            similarity = cosine_similarity(rating_matrix[user1], rating_matrix[user2])
                        elif similarity_measure == 'dcs':
                            similarity = discrete_cosine_similarity(rating_matrix[user1], rating_matrix[user2])
                        else:
                            raise Exception("Unknown similarity measure")
    return candidate_pairs


def main():
    args = parse_args()
    directory = args.d
    seed = args.s
    similarity_measure = args.m

    # TODO: remove
    similarity_measure = 'js'

    if similarity_measure != 'js' and similarity_measure != 'cs' and similarity_measure != 'dcs':
        raise Exception("Unknown similarity measure")

    np.random.seed(seed)
    results_directory = "./results"
    
    # load data
    rating_matrix = load_data(directory)

    # compute minhash
    n_hashes = 100
    signature_matrix = minhash(rating_matrix, n_hashes)

    # compute LSH
    n_bands = 20
    n_buckets = 100
    candidate_pairs = lsh(rating_matrix, signature_matrix, n_bands, n_buckets, similarity_measure)

    # write results to file corresponding to the similarity measure
    write_result(candidate_pairs, similarity_measure+".txt")
    



    print(candidate_pairs)

if __name__ == "__main__":
    main()
