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

"""
Relative Error/Absolute Error
Assume that gt and pred have the same timestamps 
"""
def metric(gt, pred):
    abs_error_x = 0
    abs_error_y = 0
    rel_error_x = 0
    rel_error_y = 0

    for (x_gt, y_gt, _), (x_pred, y_pred, _) in zip(gt, pred):
        abs_error_x += (x_pred - x_gt)
        abs_error_y += (y_pred - y_gt)
        rel_error_x += (x_pred - x_gt) / x_gt
        rel_error_y += (y_pred - y_gt) / y_gt

    abs_error_x /= len(gt)
    abs_error_y /= len(gt)  
    rel_error_x /= len(gt)
    rel_error_y /= len(gt)

    return [abs_error_x, abs_error_y, rel_error_x, rel_error_y]

"""
Save trajectories
"""
def save_pred(pred, name="pred"):
    pred_list = []
    for x, y, t in pred:
        pred_list.append({'x': x, 'y': y, 't': t})
    output = {name: pred_list}
    
    filename = f"{name}.json"
    with open(filename, 'w') as file:
        json.dump(output, file, indent=2)
