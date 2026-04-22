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
        self.interpolated_gt = self.__interpolate_gt()
        self.noise_bins = [create_bins(self.noise_traj[i], bin_size) for i in range(self.noise_traj.shape[0])]        

    def __interpolate_gt(self):
        interpolated_gt = []
        for idx in range(self.gt_bins.shape[0]):
            bin_start, bin_end, int_start, int_end = self.gt_bins[idx]
            t = (bin_start + bin_end) / 2
            x0, y0 = self.gt[int_start][0], self.gt[int_start][1]
            x1, y1 = self.gt[int_end][0], self.gt[int_end][1]
            t0, t1 = self.gt[int_start][2], self.gt[int_end][2]
            interpolate_gt = lin_interpolation(t0, t1, x0, x1, y0, y1, t)
            interpolated_gt.append(interpolate_gt)
        return interpolated_gt

    """
    Compute convex hull given a time bin and the points belonging to the time bin
    """
    def __computeConvexHull(self, bin_idx):
        # interpolate the points in the bin at 3 points, the two end points of the bin and the middle point
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
            interpolate_noise_t = lin_interpolation(noise_t0, noise_t1, noise_x0, noise_x1, noise_y0, noise_y1, t)
            interpolate_noise_t0 = lin_interpolation(noise_t0, noise_t1, noise_x0, noise_x1, noise_y0, noise_y1, t)
            interpolate_noise_t1 = lin_interpolation(noise_t0, noise_t1, noise_x0, noise_x1, noise_y0, noise_y1, t)
            bin_pts.append([interpolate_noise_t[0], interpolate_noise_t[1]])
            bin_pts.append([interpolate_noise_t0[0], interpolate_noise_t0[1]])
            bin_pts.append([interpolate_noise_t1[0], interpolate_noise_t1[1]])
        bin_pts = np.array(bin_pts)
        hull = ConvexHull(bin_pts)
        return hull

    """
    Compute the predicted trajectory given a convex hull assumption
    """
    def computeConvexHullTrajectory(self):
        traj_pred = []
        x,y,t = self.gt[0]
        traj_pred.append([float(x), float(y), float(t)])
        for idx in range(self.gt_bins.shape[0]):
            hull = self.__computeConvexHull(idx)
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
        traj_pred.append([float(x), float(y), float(t)])
        return traj_pred

    """
    Compute the predicted trajectory given a gaussian distribution assumption
    """
    def ComputeGaussianTrajectory(self):
        pred_traj = []
        x,y,t = self.gt[0]
        pred_traj.append([float(x), float(y), float(t)])
        
        # for each time bin, compute the average of the points 
        for idx in range(self.gt_bins.shape[0]):
            count = 0
            x_avg, y_avg = 0, 0
            cur_noise_bins = [self.noise_bins[noise_idx][idx] for noise_idx in range(len(self.noise_bins))]

            # for each noise trajectory, interpolate the noise points at the mid point of the time bin and add to the average 
            for nt in range(len(cur_noise_bins)):
                cur_noise = self.noise_traj[nt]
                bin_start, bin_end, int_start, int_end = cur_noise_bins[nt]
                t = (bin_start + bin_end) / 2
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
        pred_traj.append([float(x), float(y), float(t)])
        return pred_traj

    """
    Interpolate the predicted trajectory to the same timestamps as gt and compute the metric
    """
    def computeMetrics(self, pred):
        return metric(self.interpolated_gt, pred)

if __name__ == "__main__":
    solver = Solvers(path="trajectory.json")
    pred = solver.ComputeGaussianTrajectory()
    save_pred(pred)
    print(solver.computeMetrics(pred))

    pred = solver.computeConvexHullTrajectory()
    # print(pred)
    save_pred(pred, name="pred_CH")
    print(solver.computeMetrics(pred))
