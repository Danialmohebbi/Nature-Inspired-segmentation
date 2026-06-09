from fileinput import filename

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
ALGORITHMS = ['zoa','woa','hc','gwo','cmaes']
ITTERATIONS = 1
K = [2]
OUTPUT_DIR = "results"
GRAY_COUNT = 3
COLOR_COUNT = 0
CSV_HEADER = [
    "algo","image_id","mode","k","trial",
    "MSE","PSNR","FSIM","SSIM","QILV","fitness",
    "convergence"
]

def get_csv_path(mode):
    """
    returns the path to the correct output file and make sure the directory exists.
    Args:
        mode (str): 'gray' or 'color' image type.
    Return:
        output_path (str): path to the output file.
    """
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    return os.path.join(OUTPUT_DIR,f"output_{mode}.csv")

def prepare_csv(mode):
    """
    Prepare the csv file for a given mode if it doesn't exist.

    Args:
        mode (str): 'gray' or 'color' image type.
    """
    output_path = get_csv_path(mode)
    if not os.path.exists(output_path):
        with open(output_path,'w',newline="") as file:
            writer = csv.DictWriter(file,fieldnames=CSV_HEADER)
            writer.writeheader()

def add_row(row,mode):
    """
    add a row to a given mode csv output results.
    Args:
        row (dict): contains values corresponding to the header.
        mode (str):  'gray' or 'color' image type.
    """
    output_path = get_csv_path(mode)
    with open(output_path, 'a', newline="") as file:
        writer = csv.DictWriter(file,fieldnames=CSV_HEADER)
        writer.writerow(row)


def run(algo, image_path, image_id, mode, k, trial):
    """
    Run one image segmentation trail on an image, returning the result as a row for the csv.

    Args:
        algo (str): Algorithm name.
        image_path (str): Path to the image file used to open the image.
        image_id (str): Image filename which is used an identifier.
        mode (str): 'gray' or 'color' image type.
        k (int): Number of segments.
        trial (int): Trial number.
    """
    try:
        
        image = load_image(image_path,mode=mode)
        result = run_segmentation(algo, image,k,mode=mode)
        metrics = compute_all_metrics(image,result["segmented_image"],mode=mode)
        
        row = {
            "algo": algo,
            "image_id": image_id,
            "mode": mode,
            "k": k,
            "trial": trial ,
            "MSE": metrics["MSE"],
            "PSNR": metrics["PSNR"],
            "FSIM": metrics["FSIM"],
            "SSIM": metrics["SSIM"],
            "QILV": metrics["QILV"],
            "fitness": result["fitness"],
            "convergence": result["history"]
        }
        
        return row
    except Exception as e:
        print(f"=====   FAILED  | algo={algo} | image_id={image_id} | mode={mode} | k={k} | trial={trial} | {e} =====")
        return None
    
def run_experiment():
    """
    run the full experiement. Loading 150 gray and 150 color images from BSD500 train set.
    """
    all_images = get_image_paths("train")
    gray_paths = all_images[:GRAY_COUNT]
    color_paths = all_images[GRAY_COUNT:COLOR_COUNT]
    
    prepare_csv('gray')
    prepare_csv('color')
    
    tasks = []
    for algo in ALGORITHMS:
        for k in K:
            for trial in range(ITTERATIONS):
                for path in gray_paths:
                    image_id = os.path.basename(path)
                    tasks.append((algo,path,image_id, "gray", k, trial))
                for path in color_paths:
                    image_id = os.path.basename(path)
                    tasks.append((algo,path,image_id, "color", k, trial)) 
    
    results = Parallel(n_jobs=CPU_COUNT,verbose=10)(
        delayed(run)(
            algo,path,image_id, mode, k, trial
        )
        for algo,path,image_id, mode, k, trial in tasks
    )
    
    saved = 0
    for row in results:
        if row is not None:
            add_row(row,row['mode'])
            saved += 1
    
    print(f"Saved {saved}/{len(tasks)} runs.")
    
