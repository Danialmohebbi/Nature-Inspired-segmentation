import time
from joblib import Parallel, delayed
import numpy as np
import os
import csv

from data.loader import load_image,get_image_paths
from fitness.reconstruction import compute_all_metrics
from segmentation.base_segmenter import run_segmentation

#K = [2,4,6,8,10]
#ITTERATIONS = 10
#ALGORITHMS = ['pso','zoa','woa','hc','gwo','cmaes','woazoa','gwozoa','sa','ga','otsu','kmeans']
CPU_COUNT = -1
OUTPUT_DIR = "results"
CSV_HEADER = [
    "algo","image_id","mode","k","trial","fitness_fn",
    "MSE","PSNR","FSIM","SSIM","QILV","fitness",
    "convergence","runtime_seconds"
]

def get_csv_path(mode,fitness_fn):
    """
    returns the path to the correct output file and make sure the directory exists.
    Args:
        mode (str): 'gray' or 'color' image type.
        fitness_fn (str): denotes the name of the fitness function used.
    Return:
        output_path (str): path to the output file.
    """
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    return os.path.join(OUTPUT_DIR,f"output_{mode}_{fitness_fn}.csv")

def prepare_csv(mode,fitness_fn):
    """
    Prepare the csv file for a given mode if it doesn't exist.

    Args:
        mode (str): 'gray' or 'color' image type.
        fitness_fn (str): denotes the name of the fitness function used.
    """
    output_path = get_csv_path(mode,fitness_fn)
    if not os.path.exists(output_path):
        with open(output_path,'w',newline="") as file:
            writer = csv.DictWriter(file,fieldnames=CSV_HEADER)
            writer.writeheader()

def add_row(row,mode,fitness_fn):
    """
    add a row to a given mode csv output results.
    Args:
        row (dict): contains values corresponding to the header.
        mode (str):  'gray' or 'color' image type.
        fitness_fn (str): denotes the name of the fitness function used.
    """
    output_path = get_csv_path(mode,fitness_fn)
    with open(output_path, 'a', newline="") as file:
        writer = csv.DictWriter(file,fieldnames=CSV_HEADER)
        writer.writerow(row)


def run(algo, image_path, image_id, mode, k, trial,fitness_fn):
    """
    Run one image segmentation trail on an image, returning the result as a row for the csv.

    Args:
        algo (str): Algorithm name.
        image_path (str): Path to the image file used to open the image.
        image_id (str): Image filename which is used an identifier.
        mode (str): 'gray' or 'color' image type.
        k (int): Number of segments.
        trial (int): Trial number.
        fitness_fn (str): denotes the name of the fitness function used.
    """
    try:
        start_time = time.time()
        image = load_image(image_path,mode=mode)
        result = run_segmentation(algo, image,k,mode=mode,fitness_fn=fitness_fn)
        elapsed = time.time() - start_time
        metrics = compute_all_metrics(image,result["segmented_image"],mode=mode)
        
        row = {
            "algo": algo,
            "image_id": image_id,
            "mode": mode,
            "k": k,
            "trial": trial,
            "fitness_fn": fitness_fn,
            "MSE": metrics["MSE"],
            "PSNR": metrics["PSNR"],
            "FSIM": metrics["FSIM"],
            "SSIM": metrics["SSIM"],
            "QILV": metrics["QILV"],
            "fitness": result["fitness"],
            "convergence": result["history"],
            "runtime_seconds": elapsed
        }
        
        return row
    except Exception as e:
        print(f"=====   FAILED  | algo={algo} | image_id={image_id} | mode={mode} | k={k} | trial={trial} | {e} =====")
        return None
    
def run_experiment(algorithms, k_values, n_trials, fitness_fn):
    """
    """
    all_images = get_image_paths("train")

    prepare_csv('gray',fitness_fn)
    prepare_csv('color',fitness_fn)

    tasks = []
    for algo in algorithms:
        for k in k_values:
            for trial in range(n_trials):
                for (path, mode) in all_images:
                    image_id = os.path.basename(path)
                    tasks.append((algo, path, image_id, mode, k, trial,fitness_fn))

    saved = 0
    failed = 0
    for row in Parallel(n_jobs=CPU_COUNT, verbose=0, return_as="generator")(
        delayed(run)(algo, path, image_id, mode, k, trial,fitness_fn)
        for algo, path, image_id, mode, k, trial,fitness_fn in tasks
    ):
        if row is not None:
            add_row(row, row['mode'],fitness_fn)
            saved += 1
        else:
            failed += 1

    print(f"Saved {saved}/{len(tasks)} runs. ({failed} failed)")
    
