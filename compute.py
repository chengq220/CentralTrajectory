import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

"""
Return gt_np with shape num_points x 3
       noise_np num_traj x num_points x 3
"""
def getData(): 
    with open('trajectory.json', 'r') as file:
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

Returns [x,y,t] 
"""
def lin_interpolation(traj, idx, t):
    EPS = 0.01
    assert len(traj.shape) == 2
    assert idx < traj.shape[0]-1

    t_0 = traj[idx][2]
    t_1 = traj[idx+1][2]
    x_0 = traj[idx][0]
    x_1 = traj[idx+1][0]
    y_0 = traj[idx][1]
    y_1 = traj[idx+1][1]

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
    
    bins = np.vstack(bins)
    return bins

"""
Relative Error/Absolute Error
"""
def metric(gt, pred):
    return 0

    
"""
Compute convex hull given a time bin
"""
def computeConvexHull(pts, bin):
    return 0

"""
Compute the 2D Gaussian Distribution given a time bin 
"""
def computeGassianDist(pts, bin):
    return 0


gt, noise = getData()
print(gt)
out = lin_interpolation(noise[0], 0, 4)
a = create_bins(gt, 1)
print(a)
