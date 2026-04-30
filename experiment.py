import numpy as np
from solver import Solvers
from glob import glob 
from utils import save_pred

if __name__ == "__main__":
    file_path = glob("example/pois_noise/*.json")
    num_files = len(file_path)

    gauss = np.zeros(4)
    hull = np.zeros(4)

    for f in file_path: 
        solver = Solvers(path=f)

        pred = solver.ComputeGaussianTrajectory()
        # save_pred(pred, "example/gaussian_trajectory")
        gauss_metric = solver.computeMetrics(pred)
        gauss += gauss_metric

        pred = solver.computeConvexHullTrajectory()
        # save_pred(pred, "example/convex_hull_trajectory")
        hull_metric = solver.computeMetrics(pred)
        hull += hull_metric

    print("Gaussian Trajectory Average Metrics: ")
    print(np.array2string(gauss/num_files, precision=10, floatmode='fixed'))
    print("Convex Hull Trajectory Average Metrics: ")
    print(np.array2string(hull/num_files, precision=10, floatmode='fixed'))
