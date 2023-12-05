
# For the experiments, call main.py with the following arguments:
# -m 'js'
# -num_bands = 25, 50, 100
# Each experiment will run with 5 different seeds, and the results will be averaged.
# -seed = 0, 19, 42, 47, 97
# The results will be saved in the folder "results".

# Experiments are cancelled after 30 minutes of runtime.

import os
import time
import subprocess
import argparse
import numpy as np


# Run the experiments.
def run_experiments():
    # Arguments that will be passed to main.py.

    # Similarity measure
    measures = ['-m' ,'js']

    # Number of bands to test.
    num_bands = ['-n_bands', 25, 50, 100]
    # Seeds to test.
    seeds = ['-s', 0, 19, 42, 47, 97]
    # Number of runs per experiment.
    runs = 5
    # Timeout in seconds.
    timeout = 30 * 60

    # Create results folder if it does not exist.
    if not os.path.exists('results'):
        os.makedirs('results')

    # Run experiments.
    for measure in measures[1:]:
        for num_band in num_bands[1:]:
            for seed in seeds[1:]:
                for run in range(runs):
                    print('Running experiment with num_bands = {}, seed = {}, run = {}.'.format(num_band, seed, run))
                    # Create folder for results.
                    folder = 'results/num_bands_{}_seed_{}_run_{}'.format(num_band, seed, run)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    # Call main.py. and wait for it to finish.
                    args = [measures[0], measure, num_bands[0], str(num_band), seeds[0], str(seed)]

                    cmd = ['python', 'main.py']
                    for arg in args:
                        cmd.append(arg)
                    # Call main.py.

                    subprocess.call(cmd, timeout=timeout)

                    # Move results to folder.
                    os.rename(measure+'.txt', folder + '/'+measure+'.txt')

# Run experiments.
run_experiments()
