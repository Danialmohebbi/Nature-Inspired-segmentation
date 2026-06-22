import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from evaluation.runner import run_experiment

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--algos", type=str, required=True, help="Comma seperated algorithm names")
    parser.add_argument("--kvals", type=str, required=True, help="Comma seperated k values")
    parser.add_argument("--trials", type=int, required=True, help="Number of trials")
    parser.add_argument("--fitness", type=str, required=True, help="Fitness function name")
    
    return parser.parse_args()


if __name__ == "__main__":
    print("===== Starting main experiement... =====")
    args = parse_args()
    
    algos = [algo.strip() for algo in args.algos.split(',')]
    k_values = [int(k) for k in args.kvals.split(',')]
    
    run_experiment(algorithms=algos,k_values=k_values,n_trials=args.trials,fitness_fn=args.fitness)
    print("===== Experiement Completed. Results saved to results/ =====")