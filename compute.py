import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import ConvexHull, convex_hull_plot_2d

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
    EPS = 0.1
    # print(t_0, t, t_1)
    assert t_1 > t_0
    # to approximate t within the bin
    t_alter = max(min(t, t_1 - EPS), t_0 + EPS)

    m_x = float((x_1 - x_0))/(t_1 - t_0)
    m_y = float((y_1 - y_0))/(t_1 - t_0)

    x_out = x_1 + m_x * (t_alter - t_1)
    y_out = y_1 + m_y * (t_alter - t_1)

    return [x_out, y_out, t]
"""
Create a data structure O(N) to enable 
O(1) query for where interval does the bin belong to

[[bin_start_t, bin_end_t, interval_start_idx, interval_end_idx]]
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
        # TODO: need to finish

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
Compute convex hull given a time bin and the points belonging to the time bin
"""
def computeConvexHull(trajectories, bin_idx):
    bin_start, bin_end, int_start, int_end = bin
    # interpolate the points in the bin at 3 points, the two end points and the middle point
    bin_pts = []
    for traj in trajectories: 
        
    hull = ConvexHull(bin_pts)
    return hull

"""
Compute the predicted trajectory given a convex hull assumption
"""
def computeConvexHullTrajectory(path, bin_size = 1):
    gt, noise_traj = getData(path)
    gt_bins = create_bins(gt, bin_size)
    return 0

"""
Compute the predicted trajectory given a gaussian distribution assumption
"""
def ComputeGaussianTrajectory(path, bin_size = 1):
    gt, noise_traj = getData(path)
    gt_bins = create_bins(gt, bin_size)
    noise_bins = [create_bins(noise_traj[i], 1) for i in range(noise_traj.shape[0])]
    pred_traj = []
    x,y,t = gt[0]
    pred_traj.append([int(x), int(y), int(t)])
    for idx in range(gt_bins.shape[0]):
        count = 1
        x_avg, y_avg = 0, 0
        cur_noise_bins = [noise_bins[noise_idx][idx] for noise_idx in range(len(noise_bins))]
        bin_start, bin_end, int_start, int_end = gt_bins[idx]
        t = (bin_start + bin_end) / 2
        gt_x0, gt_y0 = gt[int_start][0], gt[int_start][1]
        gt_x1, gt_y1 = gt[int_end][0], gt[int_end][1]
        gt_t0, gt_t1 = gt[int_start][2], gt[int_end][2]
        interpolate_gt = lin_interpolation(gt_t0, gt_t1, gt_x0, gt_x1, gt_y0, gt_y1, t)
        x_avg += interpolate_gt[0]
        y_avg += interpolate_gt[1]
        for nt in range(len(cur_noise_bins)):
            cur_noise = noise_traj[nt]
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
        pred_traj.append([x_avg, y_avg, t])
    x,y,t = gt[-1]
    pred_traj.append([int(x), int(y), int(t)])
    return pred_traj

def save_pred(pred, name="pred"):
    pred_list = []
    for x, y, t in pred:
        pred_list.append({'x': x, 'y': y, 't': t})
    output = {name: pred_list}
    
    filename = f"{name}.json"
    with open(filename, 'w') as file:
        json.dump(output, file, indent=2)

if __name__ == "__main__":
    pred = ComputeGaussianTrajectory(path="example/trajectory.json")
    print(pred)
    save_pred(pred)
