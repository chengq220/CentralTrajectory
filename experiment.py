import numpy as np
from solver import Solvers
from glob import glob 

if __name__ == "__main__":
    file_path = glob("example/pois_noise/*.json")
    num_files = len(file_path)

    gauss = np.zeros(4)
    hull = np.zeros(4)

    for f in file_path: 
        print(f)
        solver = Solvers(path=f)

        pred = solver.ComputeGaussianTrajectory()
        gauss_metric = solver.computeMetrics(pred)
        gauss += gauss_metric

        pred = solver.computeConvexHullTrajectory()
        hull_metric = solver.computeMetrics(pred)
        hull += hull_metric

    print("Gaussian Trajectory Average Metrics: ")
    print(np.array2string(gauss/num_files, precision=10, floatmode='fixed'))
    print("Convex Hull Trajectory Average Metrics: ")
    print(np.array2string(hull/num_files, precision=10, floatmode='fixed'))
