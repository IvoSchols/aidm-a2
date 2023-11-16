import numpy as np
import scipy
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("-d", default="data/user_movie_rating.npy", help="specify data file path")
argparser.add_argument("-s", default=42, type=int, help="the random seed to be used")
argparser.add_argument("-m", choices = ['js','cs','dcs'], help="similarity measure: jacard (js), cosine (cs), discrete cosine (dcs)")
args = argparser.parse_args()

np.random.seed(args.s)

results_directory = "./results"

def load_data():
    data = np.load(args.d)
    return data

#takes result [s1,s2] and appends it to results file
def update_results(result : list):
    if not os.path.isdir(results_directory): # create results directory
        os.mkdir(results_directory)

    line = "{}, {}\n".format(result[0],result[1])
    f = open("{}/{}.txt".format(results_directory,args.m),'a')
    f.write(line)
    f.close()










if __name__ == "__main__":
    data = load_data()
