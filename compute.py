import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

"""
Return gt_np with shape num_points x 3
       noise_np num_traj x num_points x 3
"""
def getData(path): 
    with open(path, 'r') as file:
        data = json.load(file)
        gt = data['ground_truth']
        gt_np = []
        for i in range(len(gt)):
            c = gt[i]
            gt_np.append([c['x'], c['y'], c['t']])
        gt_np = np.array(gt_np)

        noise_list = data['noisy_trajectories']
        noise_np = []
        for noise in noise_list:
            temp = []
            for i in range(len(noise)):
                c = noise[i]
                temp.append([c['x'], c['y'], c['t']])
            noise_np.append(temp)
        noise_np = np.array(noise_np)
    
    return gt_np, noise_np

"""
Interpolate the trajectory path using linear interpolation

Returns [x, y, t] 
"""
def lin_interpolation(t_0, t_1, x_0, x_1, y_0, y_1, t):
    EPS = 0.01
    assert t_1 > t_0
    assert t <= t_1 - EPS and t >= t_0 + EPS

    m_x = float((x_1 - x_0))/(t_1 - t_0)
    m_y = float((y_1 - y_0))/(t_1 - t_0)

    x_out = x_1 + m_x * (t - t_1)
    y_out = y_1 + m_y * (t - t_1)
    
    return [x_out, y_out, t]

"""
Create a data structure O(N) to enable 
O(1) query for where interval does the bin belong to

[[bin_start, bin_end, interval_start_idx, interval_end_idx]]
"""
def create_bins(traj, bin_size):
    assert len(traj.shape) == 2    
    bins = []
    for i in range(traj.shape[0]-1):
        t_0 = traj[i][2]
        t_1 = traj[i+1][2]
        bin_range = np.arange(t_0, t_1, bin_size, dtype=np.float64)
        bin_range = np.append(bin_range, t_1)
        windows = sliding_window_view(bin_range, window_shape=2)
        i_0 = np.ones((windows.shape[0], 1)) * i
        i_1 = np.ones((windows.shape[0], 1)) * (i + 1)
        stacked = np.hstack((windows, i_0, i_1))
        bins.append(stacked)
    bins = np.vstack(bins).astype(int)
    return bins

# """
# Find the bins that a time stamp t belong to 
# Return [bin_start, bin_end, interval_start_idx, interval_end_idx]
# O(N) using just a brute force method 
# O(logN) using binary search
# """
# def find_bin(bins, t):
#     """
#     Do i even need this function? since bins already sorted by time, i can just iterate through through it as i process points
#     """
#     for bin in bins:
#         if t >= bin[0] and t <= bin[1]:
#             return bin

"""
Relative Error/Absolute Error
Assume 
"""
def metric(gt, pred):
    abs_error_x = 0
    abs_error_y = 0
    rel_error_x = 0
    rel_error_y = 0

    for (x_gt, y_gt, t_gt), (x_pred, y_pred, t_pred) in zip(gt, pred):
        # find the time interval that t_pred belongs to 


        interpolate_x, interpolate_y, t = lin_interpolation(t_0, t_1, x_0, x_1, y_0, y_1, t_pred)

        abs_error_x += (x_pred - interpolate_x)
        abs_error_y += (y_pred - interpolate_y)
        rel_error_x += (x_pred - interpolate_x) / interpolate_x
        rel_error_y += (y_pred - interpolate_y) / interpolate_y

    abs_error_x /= len(gt)
    abs_error_y /= len(gt)  
    rel_error_x /= len(gt)
    rel_error_y /= len(gt)

    return abs_error_x, abs_error_y, rel_error_x, rel_error_y

    
"""
Compute convex hull given a time bin
"""
def computeConvexHull(pts, bin):
    return 0

"""
Compute the predicted trajectory given a gaussian distribution assumption
"""
def ComputeGaussianTrajectory(path):
    gt, noise_traj = getData(path)
    gt_bins = create_bins(gt, 1)
    noise_bins = [create_bins(noise_traj[i], 1) for i in range(noise_traj.shape[0])]
    pred_traj = []
    for idx in range(gt_bins.shape[0]):
        count = 1
        x_avg, y_avg = 0, 0
        cur_noise_bins = [noise_bins[noise_idx][idx] for noise_idx in range(len(noise_bins))]
        bin_start, bin_end, _, _ = gt_bins[idx]
        print(bin_start, bin_end)
        print(gt[4], gt[bin_end])
        print("===============================")
        t = (bin_start + bin_end) / 2
        gt_x0, gt_y0 = gt[bin_start][0], gt[bin_start][1]
        gt_x1, gt_y1 = gt[bin_end][0], gt[bin_end][1]
        interpolate_gt = lin_interpolation(bin_start, bin_end, gt_x0, gt_x1, gt_y0, gt_y1, t)
        x_avg += interpolate_gt[0]
        y_avg += interpolate_gt[1]
        for nt, noise_bin in enumerate(cur_noise_bins):
            cur_noise = noise_traj[nt]
            bin_start, bin_end, _, _ = noise_bin
            noise_x0, noise_y0 = cur_noise[bin_start][0], cur_noise[bin_start][1]
            noise_x1, noise_y1 = cur_noise[bin_end][0], cur_noise[bin_end][1]
            interpolate_noise = lin_interpolation(bin_start, bin_end, noise_x0, noise_x1, noise_y0, noise_y1, t)
            x_avg += interpolate_noise[0]
            y_avg += interpolate_noise[1]
            count += 1
        x_avg /= count
        y_avg /= count
        pred_traj.append([x_avg, y_avg, t])
    return pred_traj

if __name__ == "__main__":
    ComputeGaussianTrajectory(path="trajectory.json")


# gt, noise = getData()
# print(gt)
# out = lin_interpolation(noise[0], 0, 4)
# a = create_bins(gt, 1)
# print(a)
