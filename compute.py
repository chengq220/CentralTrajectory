import json
import numpy as np

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
Relative Error/Absolute Error
"""
def metric(gt, pred):
    return 0


"""
Interpolate the trajectory path using linear interpolation
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
    print(t_0)

    assert t <= t_1 - EPS and t >= t_0 + EPS

    m_x = float((x_1 - x_0))/(t_1 - t_0)
    m_y = float((y_1 - y_0))/(t_1 - t_0)

    x_out = x_1 + m_x * (t - t_1)
    y_out = y_1 + m_y * (t - t_1)
    
    return [x_out, y_out]
    
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
print(noise[0])
out = lin_interpolation(noise[0], 0, 4)
print(out)
