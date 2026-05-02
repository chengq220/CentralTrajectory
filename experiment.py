import numpy as np
from solver import Solvers
from glob import glob 
from utils import save_pred
from time import perf_counter as time

if __name__ == "__main__":
    folders = ["example/no_noise/", "example/gauss_noise/", "example/pois_noise/"]
    file_path = []
    for folder in folders:
        file_path.extend(glob(f"{folder}*.json"))
        num_files = len(file_path)
        
        assert num_files > 0, "No files found in the specified directory."

        gauss = np.zeros(4)
        hull = np.zeros(4)

        gaus_time = 0
        hull_time = 0

        for f in file_path: 
            solver = Solvers(path=f, bin_size=1, interpolate="cubic")

            pred = solver.ComputeGaussianTrajectory()
            # save_pred(pred, "example/gaussian_trajectory")

            start = time()
            gauss_metric = solver.computeMetrics(pred)
            end = time()
            gaus_time += end - start
            gauss += gauss_metric
            
            pred = solver.computeConvexHullTrajectory()
            # save_pred(pred, "example/convex_hull_trajectory")
            start = time()
            hull_metric = solver.computeMetrics(pred)
            end = time()
            hull_time += end - start
            hull += hull_metric

        print("Gaussian Trajectory Average Metrics: ")
        print(np.array2string(gauss/num_files, precision=10, floatmode='fixed'))
        print("Convex Hull Trajectory Average Metrics: ")
        print(np.array2string(hull/num_files, precision=10, floatmode='fixed'))
        print("=========================================")
    # print(np.max(gauss/num_files))
    # print(f"Max Convex Hull Metrics: {np.max(hull/num_files)}")
    # print(f"Gaussian Time: {gaus_time/num_files}")
    # print(f"Convex Hull Time: {hull_time/num_files}")
