from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


class CreateWaypoints():
    def __init__(self, img_path):
        self.img_path = img_path
        self.tif_img = None
    
    def tif2numpy(self):
        self.tif_img = Image.open(self.img_path)
        tif_image_np = np.array(self.tif_img)
        # Normalize the numpy image array
        tif_image_np = (tif_image_np - np.min(tif_image_np)) / (np.max(tif_image_np) - np.min(tif_image_np))
        # Return normalized numpy array of image
        return tif_image_np

    def createContour(self):
        src = cv.imread(self.img_path)
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]
        contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        blank = np.zeros(thresh.shape, dtype='uint8')
        cv.drawContours(blank, contours, -1, (255,0,0), 1)
        return contours, thresh, src
    
    def createCentroids(self):
        centroids = []
        contours, _, src = self.createContour()
        # Append first point of image to centroids
        centroids.append((212, 240))
        for i in contours:
            M = cv.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv.drawContours(src, [i], -1, (0, 255, 0), 2)
                cv.circle(src, (cx, cy), 7, (0, 0, 255), -1)
                cv.putText(src, "center", (cx - 20, cy - 20),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                centroids.append((cx, cy))
        return centroids
    
    def create_waypoints(self, point1, point2, num_waypoints = 5):
        x1, y1 = point1
        x2, y2 = point2
        x_values = np.linspace(x1, x2, num_waypoints)
        y_values = np.linspace(y1, y2, num_waypoints)

        # Create a list of tuples representing the waypoints
        waypoints = [(int(x), int(y)) for x, y in zip(x_values, y_values)]
        return waypoints
    
    def implement(self):
        centroids = self.createCentroids()
        for i in range(len(centroids)-1):
            waypoints = self.create_waypoints(centroids[i], centroids[i+1])
            # print(waypoints)
            # Plot the waypoints as stars
            for point in waypoints:
                plt.scatter(point[0], point[1], marker='*', color='red', s=100)

if __name__ == "__main__":
    path = r"C:/Users/shrey/Documents/LIR_Project/ANEU/mu_ts_2021_11_09_16h15m31s_000000.tif"
    obj = CreateWaypoints(path)
    obj.implement()
