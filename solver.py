import numpy as np
from scipy.spatial import ConvexHull
from utils import *
from utils import save_pred

"""
Central Trajectory Solver
"""
class Solvers:
    def __init__(self, path, bin_size = 1, interpolate = "linear"):
        self.gt, self.noise_traj = getData(path)
        assert len(self.gt) > 0, "Ground truth trajectory cannot be empty"
        assert len(self.noise_traj) > 0, "Noise trajectory cannot be empty"
        if interpolate not in ["linear", "cubic"]:
            raise ValueError("Interpolation method must be 'linear' or 'cubic'")
        self.gt_bins = create_bins(self.gt, bin_size)
        self.interpolated_gt = self.__interpolate_gt(interpolate)
        self.noise_bins = [create_bins(self.noise_traj[i], bin_size) for i in range(self.noise_traj.shape[0])]        

    def __interpolate_gt(self, interpolate):
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
    def __computeConvexHull(self, bin_idx, num_points=5):
        # interpolate the points in the bin at 3 points, the two end points of the bin and the middle point
        bin_start, bin_end, int_start, int_end = self.gt_bins[bin_idx]
        bin_pts = []
        cur_noise_bins = [self.noise_bins[noise_idx][bin_idx] for noise_idx in range(len(self.noise_bins))]
        for nt in range(self.noise_traj.shape[0]):
            cur_noise = self.noise_traj[nt]
            bin_start, bin_end, int_start, int_end = cur_noise_bins[nt]
            noise_x0, noise_y0 = cur_noise[int_start][0], cur_noise[int_start][1]
            noise_x1, noise_y1 = cur_noise[int_end][0], cur_noise[int_end][1]
            noise_t0, noise_t1 = cur_noise[int_start][2], cur_noise[int_end][2]
            for _ in range(num_points):
                t = np.random.uniform(bin_start, bin_end)
                interpolate_noise_t = lin_interpolation(noise_t0, noise_t1, noise_x0, noise_x1, noise_y0, noise_y1, t)
                bin_pts.append([interpolate_noise_t[0], interpolate_noise_t[1]])
        bin_pts = np.array(bin_pts)
        hull = ConvexHull(bin_pts, qhull_options="QJ")
        return hull

    """
    Compute the predicted trajectory given a convex hull assumption
    """
    def computeConvexHullTrajectory(self):
        traj_pred = []
        x,y,t = self.gt[0]
        traj_pred.append([float(x), float(y), float(t)])
        num_bins = min(self.gt_bins.shape[0], 
                min([len(bins) for bins in self.noise_bins]) if self.noise_bins else 0)
        for idx in range(num_bins):
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

        num_bins = min(self.gt_bins.shape[0], 
                min([len(bins) for bins in self.noise_bins]) if self.noise_bins else 0)
        
        # for each time bin, compute the average of the points 
        for idx in range(num_bins):
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

# if __name__ == "__main__":

#     from glob import glob 

#     file_path = glob("example/noise/*.json")
#     num_files = len(file_path)

#     gauss = np.zeros(4)
#     hull = np.zeros(4)

    # for f in file_path: 
    #     print(f)
#         solver = Solvers(path=f)

#         pred = solver.ComputeGaussianTrajectory()
#         # save_pred(pred)
#         gauss_metric = solver.computeMetrics(pred)
#         gauss += gauss_metric

#         pred = solver.computeConvexHullTrajectory()
#         # save_pred(pred, name="pred_CH")
#         hull_metric = solver.computeMetrics(pred)
#         hull += hull_metric

#     print("Gaussian Trajectory Average Metrics: ", gauss/num_files)
#     print("Convex Hull Trajectory Average Metrics: ", hull/num_files)
