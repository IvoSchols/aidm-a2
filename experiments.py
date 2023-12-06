
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
from time import perf_counter

# Run the experiments.
def run_experiments():
    # Arguments that will be passed to main.py.

    # Similarity measure
    measures = ['-m' ,'js', 'cs', 'dcs']
    # measures = ['-m' ,'cs']

    # N_hashes to test.
    num_hashes = ['-n_hashes', 100, 120, 150]

    # Number of bands to test.
    num_bands = ['-n_bands', 20, 10, 5] # Increasing number of bands increases the number of false positives. (and runtime)
    # Seeds to test.
    seeds = ['-s', 19, 42, 47]
    
    # Timeout in seconds.
    timeout = 30 * 60

    # Create results folder if it does not exist.
    if not os.path.exists('results'):
        os.makedirs('results')

    # Run experiments.
    for measure in measures[1:]:
        # Create folder for results per measure.
        folder = f'results/{measure}'
        if not os.path.exists(folder):
            os.makedirs(folder)

        for num_hash in num_hashes[1:]:
            for num_band in num_bands[1:]:
                for seed in seeds[1:]:
                        print(f'Running experiment with measure = {measure}, num_hash = {num_hash}, num_band = {num_band}, seed = {seed}')
                        # Call main.py. and wait for it to finish.
                        args = [measures[0], measure, num_bands[0], str(num_band), seeds[0], str(seed)]

                        cmd = ['python', 'main.py']
                        for arg in args:
                            cmd.append(arg)
                        # Run main.py.
                        start = perf_counter()
                        try:
                            subprocess.run(cmd, timeout=timeout)
                        except subprocess.TimeoutExpired:
                            print('Timeout expired. Continuing with next experiment.')
                        execution_time = perf_counter() - start

                        # Append execution time to results file.
                        with open(f'{measure}.txt', 'a') as f:
                            f.write(f'{execution_time}\n')

                        # Move results to folder.
                        os.rename(f'{measure}.txt', f'{folder}/{measure}_{num_hash}_{num_band}_{seed}.txt')

# Run experiments.
run_experiments()
