import json
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from utils import *

"""
Central Trajectory Solver
"""
class Solvers:
    def __init__(self, path, bin_size = 1):
        self.gt, self.noise_traj = getData(path)
        self.gt_bins = create_bins(self.gt, bin_size)
        self.noise_bins = [create_bins(self.noise_traj[i], bin_size) for i in range(self.noise_traj.shape[0])]    
    
    """
    Compute convex hull given a time bin and the points belonging to the time bin
    """
    def computeConvexHull(self, bin_idx):
        # interpolate the points in the bin at 3 points, the two end points and the middle point
        bin_start, bin_end, int_start, int_end = self.gt_bins[bin_idx]
        t = (bin_start + bin_end) / 2
        bin_pts = []
        cur_noise_bins = [self.noise_bins[noise_idx][bin_idx] for noise_idx in range(len(self.noise_bins))]
        for nt in range(self.noise_traj.shape[0]):
            cur_noise = self.noise_traj[nt]
            bin_start, bin_end, int_start, int_end = cur_noise_bins[nt]
            noise_x0, noise_y0 = cur_noise[int_start][0], cur_noise[int_start][1]
            noise_x1, noise_y1 = cur_noise[int_end][0], cur_noise[int_end][1]
            noise_t0, noise_t1 = cur_noise[int_start][2], cur_noise[int_end][2]
            interpolate_noise = lin_interpolation(noise_t0, noise_t1, noise_x0, noise_x1, noise_y0, noise_y1, t)
            bin_pts.append([interpolate_noise[0], interpolate_noise[1]])
            bin_pts.append([noise_x1, noise_y1])
            bin_pts.append([noise_x0, noise_y0])
        bin_pts = np.array(bin_pts)
        hull = ConvexHull(bin_pts)
        return hull

    """
    Compute the predicted trajectory given a convex hull assumption
    """
    def computeConvexHullTrajectory(self):
        traj_pred = []
        x,y,t = self.gt[0]
        traj_pred.append([int(x), int(y), int(t)])
        for idx in range(self.gt_bins.shape[0]):
            hull = self.computeConvexHull(idx)
            pred_x = 0 
            pred_y = 0
            bin_start, bin_end, int_start, int_end =  self.noise_bins[0][idx]
            t = (bin_start + bin_end) / 2
            for vert in hull.vertices:
                pred_x += hull.points[vert][0]
                pred_y += hull.points[vert][1]
            pred_x /= len(hull.vertices)
            pred_y /= len(hull.vertices)
            traj_pred.append([float(pred_x), float(pred_y), float(t)])
        x,y,t = self.gt[-1]
        traj_pred.append([int(x), int(y), int(t)])
        return traj_pred

    """
    Compute the predicted trajectory given a gaussian distribution assumption
    """
    def ComputeGaussianTrajectory(self):
        pred_traj = []
        x,y,t = self.gt[0]
        pred_traj.append([int(x), int(y), int(t)])
        for idx in range(self.gt_bins.shape[0]):
            count = 1
            x_avg, y_avg = 0, 0
            cur_noise_bins = [self.noise_bins[noise_idx][idx] for noise_idx in range(len(self.noise_bins))]
            bin_start, bin_end, int_start, int_end = self.gt_bins[idx]
            t = (bin_start + bin_end) / 2
            for nt in range(len(cur_noise_bins)):
                cur_noise = self.noise_traj[nt]
                bin_start, bin_end, int_start, int_end = cur_noise_bins[nt]
                noise_x0, noise_y0 = cur_noise[int_start][0], cur_noise[int_start][1]
                noise_x1, noise_y1 = cur_noise[int_end][0], cur_noise[int_end][1]
                noise_t0, noise_t1 = cur_noise[int_start][2], cur_noise[int_end][2]
                interpolate_noise = lin_interpolation(noise_t0, noise_t1, noise_x0, noise_x1, noise_y0, noise_y1, t)
                x_avg += interpolate_noise[0]
                y_avg += interpolate_noise[1]
                count += 1
            x_avg /= count
            y_avg /= count
            pred_traj.append([float(x_avg), float(y_avg), float(t)])
        x,y,t = self.gt[-1]
        pred_traj.append([int(x), int(y), int(t)])
        return pred_traj

    """
    Interpolate the predicted trajectory to the same timestamps as gt and compute the metric
    """
    def computeMetrics(self, pred):
        
        return metric(self.gt, pred)

if __name__ == "__main__":
    solver = Solvers(path="trajectory.json")
    pred = solver.ComputeGaussianTrajectory()
    # pred = solver.computeConvexHullTrajectory()
    # print(pred)
    save_pred(pred)
