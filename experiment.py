import numpy as np
from solver import Solvers
from glob import glob 

if __name__ == "__main__":
    file_path = glob("example/noise/*.json")
    num_files = len(file_path)

    gauss = np.zeros(4)
    hull = np.zeros(4)

    for f in file_path: 
        print(f)
        solver = Solvers(path=f)

        pred = solver.ComputeGaussianTrajectory()
        # save_pred(pred)
        gauss_metric = solver.computeMetrics(pred)
        gauss += gauss_metric

        pred = solver.computeConvexHullTrajectory()
        # save_pred(pred, name="pred_CH")
        hull_metric = solver.computeMetrics(pred)
        hull += hull_metric

    print("Gaussian Trajectory Average Metrics: ", gauss/num_files)
    print("Convex Hull Trajectory Average Metrics: ", hull/num_files)
