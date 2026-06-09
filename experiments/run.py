import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.runner import run_experiment

if __name__ == "__main__":
    print("===== Starting main experiement... =====")
    run_experiment()
    print("===== Experiement Completed. Results saved to results/ =====")