import numpy as np
import csv

K = "694.71558391 0.0 449.3751854; 0.0 695.54967688 258.64722075; 0.0 0.0 1.0"
K = np.matrix(K)
R = "0.0007963 0.007963 -0.9999993; -0.9999997 0.0000006 -0.0007963; 0.0 0.9999997 0.0007963"
R = np.matrix(R)

# no translation

def pix_to_world(u,v):
    p = np.array([u,v,1]).reshape((3,1))

    p_camera = np.linalg.inv(K) @ p
    p_car = R.T @ p_camera

    return p_car


def convert_waypoints(waypoints):
    car_coords = []
    for (u,v) in waypoints:
        car_frame = pix_to_world(u,v)
        car_coords.append(car_frame[:2].ravel())
    return np.array(car_coords)

def save_waypoints_as_csv(car_coords, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(car_coords)


print("Converting waypoints from camera frame to car frame...")
waypoints = [(212, 240), (211, 209), (211, 179), (211, 149), (211, 119)]
car_coords = convert_waypoints(waypoints)
file = r"/home/nithasree/learning_ws/project/Autonomous-Navigation-for-Unstructured-Environments/scripts/waypoints.csv"
save_waypoints_as_csv(car_coords, file)
print("Waypoints saved successfully!")