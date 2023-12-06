import os
import subprocess
import multiprocessing
from time import perf_counter

def run_experiment(measure, num_hash, num_band, seed, timeout):
    print(f'Running experiment with measure = {measure}, num_hash = {num_hash}, num_band = {num_band}, seed = {seed}')
    file_name = f'{measure}_{num_hash}_{num_band}_{seed}'

    # Call main.py and wait for it to finish.
    args = ['-m', measure, '-n_bands', str(num_band), '-s', str(seed), '-n_hashes', str(num_hash), '-file_name', file_name]

    cmd = ['python', 'main_experiments.py'] + args

    start = perf_counter()
    try:
        subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print('Timeout expired. Continuing with next experiment.')

    execution_time = perf_counter() - start

    # Append execution time to results file.
    with open(f'{file_name}.txt', 'a') as f:
        f.write(f'{execution_time}\n')

    print(f'Done with experiment: {file_name}')

def run_experiments_parallel():
    # Arguments that will be passed to main.py.
    measures = ['js', 'cs', 'dcs']
    num_hashes = [100, 120, 150]
    num_bands = [20, 10, 5]
    seeds = [19, 42, 47]
    timeout = 30 * 60

    # Create a pool of workers
    with multiprocessing.Pool(41) as pool:
        jobs = []
        for measure in measures:
            for num_hash in num_hashes:
                for num_band in num_bands:
                    for seed in seeds:
                        job = pool.apply_async(run_experiment, (measure, num_hash, num_band, seed, timeout))
                        jobs.append(job)

        # Wait for all jobs to complete
        pool.close()
        pool.join()

        # Collect results
        results = [job.get() for job in jobs if job.get() is not None]

    print('All experiments are done!')

# Run experiments in parallel
run_experiments_parallel()
