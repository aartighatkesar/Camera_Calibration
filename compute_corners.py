import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

DEBUG = 0  # TODO: Replace with logger

class GetCorners():
    def __init__(self, results_dir, num_horiz, num_vert, dist):
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)

        self.results_dir = results_dir
        self.num_horiz = num_horiz
        self.num_vert = num_vert
        self.dist = dist

    def remove_multiple_lines_1(self, lines, reqd_num_lines, thresh=15):
        """
        Function to get unique <reqd_num_lines> lines
        :param lines: N x 3, col_1 -> rho, col_2 -> theta, col_3 -> projection on x or y axis
        :param reqd_num_lines: total no of unique lines needed
        :return: out_lines : <reqd_num_lines> x 3
        """
        id_s = np.argsort(lines[:, 2])
        lines = lines[id_s]
        if DEBUG == 1:
            print("input lines:{}".format(lines))

        out_id = []
        invalid_id = []
        d_cal = []
        d_min = []
        d_max = []

        for i, (rho, theta, proj) in enumerate(lines):
            if not out_id:
                out_id.append(i)
            elif proj - lines[out_id[-1], 2] > thresh:
                out_id.append(i)
            else:
                invalid_id.append(i)

        if len(out_id) < reqd_num_lines:
            for id in invalid_id:
                out_min = lines[out_id][:, 2] - lines[id][2]
                if np.any(out_min>0):
                    d_min.append(np.min(out_min[out_min>0]))
                else:
                    d_min.append(0)
                if np.any(out_min < 0):
                    d_max.append(np.abs(np.max(out_min[out_min < 0])))
                else:
                    d_max.append(0)

            for i in range(len(d_min)):
                d_cal.append(abs(d_min[i] - d_max[i]))

            id = np.argsort(np.array(d_cal))

            invalid_id = np.array(invalid_id)[id]

            num_extra = reqd_num_lines - len(out_id)

            out_id.extend(invalid_id[0:num_extra].tolist())

        if DEBUG == 1:
            print("=========")
            print("selected:{}".format(lines[out_id]))
            print("=========")
            print("discarded:{}".format(lines[invalid_id]))
            print("@@@@@@@@@@@@@@@@@@@@@")

        # Sort according to projection. In case of horizontal lines, it would be left to right and vertical,
        # it would be top to bottom
        out = lines[out_id]
        id = np.argsort(out[:, 2])
        out = out[id]

        return out

    def draw_lines(self, lines, img, color):

        img_cpy = np.copy(img)

        def _get_line_ends(rho, theta):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            return x1, y1, x2, y2

        for i in range(lines.shape[0]):
            rho = lines[i][0]
            theta = lines[i][1]
            x1, y1, x2, y2 = _get_line_ends(rho, theta)
            cv2.line(img_cpy, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            if DEBUG == 1:
                cv2.putText(img_cpy, "{}_{}".format(rho, theta), (int((x1+x2)/2)+100, int((y1+y2)/2)), 0, 0.25, 255)

        return img_cpy


    def generate_world_crd(self, num_horiz, num_vert, dist):
        """
        Generate the world coordinates in 3D, with the same ordering as corner points. Consider plane in z=0 plane
        :param num_horiz: Number of horizontal lines
        :param num_vert: Number of vertical lines
        :param dist: Distance between squares on grid. Each metric unit is considered as 1 pixel
        :return:
        """
        world_crd_hc = [[] for _ in range(num_vert * num_horiz)]
        for i in range(num_horiz):
            for j in range(num_vert):
                world_crd_hc[i * num_vert + j] = [j * dist, i * dist, 0, 1]

        return np.array(world_crd_hc)


    def get_horiz_vert_lines(self, img, outimg_name):

        # Get Hough Lines
        edges = cv2.Canny(img, 300, 500, None, 3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)  # lines is N x 1 x 2(rho, theta)
        lines = np.squeeze(lines)  # N x 2

        horizontal = lines[np.logical_and(lines[:, 1] >= np.pi/6, lines[:, 1] <= 5*np.pi/6)]  # theta ~ 90deg
        horizontal = np.hstack((horizontal, (horizontal[:, 0:1] * np.sin(horizontal[:, 1:2]))))  # Get y axis projection

        vertical = lines[np.logical_not(np.logical_and(lines[:, 1] >= np.pi/6, lines[:, 1] <= 5*np.pi/6))]  # Theta ~ 0deg
        vertical = np.hstack((vertical, (vertical[:, 0:1] * np.cos(vertical[:, 1:2]))))  # Get x axis projection

        out_fldr = os.path.join(self.results_dir, "lines")
        if not os.path.exists(out_fldr):
            os.makedirs(out_fldr)

        # Draw lines before processing

        img_1 = self.draw_lines(horizontal, img, (0, 255, 0))  # draw horizontal lines
        img_1 = self.draw_lines(vertical, img_1, (0, 0, 255))  # draw vertical lines

        concat = np.concatenate((img_1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)

        plt.figure()
        plt.subplot(211)
        plt.imshow(concat, interpolation='bilinear')
        plt.title("Horizontal and Vertical lines")

        ax1 = plt.subplot(212)
        plt.scatter(lines[:, 1].flatten(), np.arange(lines.shape[0]))
        ax1.set_title('Scatter Plot of theta')
        ax1.set_xlabel("theta in radians")
        ax1.set_yticks([])

        fig = plt.gcf()
        fig.set_size_inches((4, 6), forward=False)
        fig.savefig(os.path.join(out_fldr, outimg_name), dpi=500)
        plt.close()

        # Process lines to discard duplicate lines
        horizontal = self.remove_multiple_lines_1(horizontal, reqd_num_lines=self.num_horiz, thresh=13)
        vertical = self.remove_multiple_lines_1(vertical, reqd_num_lines=self.num_vert)

        img = self.draw_lines(horizontal, img, (0, 255, 0))  # draw horizontal lines
        img = self.draw_lines(vertical, img, (0, 0, 255))  # draw vertical lines

        cv2.imwrite(os.path.join(out_fldr, "processed_" + outimg_name.split('.')[0] + ".jpg"), img)

        if DEBUG == 1:
            cv2.imshow("lines_{}".format(outimg_name), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return horizontal, vertical


    def get_intersection_of_lines(self, horizontal, vertical):
        """
        Function to get intersection between horizontal and vertical lines
        :param horizontal: N x 2( or M) with col0 -> rho, col1-> theta
        :param vertical: N x 2(or M) with col0 -> rho, col1 -> theta
        :return: corners rows of x, y Homogenous coordinates N x 3
        """
        # Number points from left to right, top to bottom


        horizonal_HC = self.generate_line_eqns(horizontal)  # rows of [a, b, c]
        vertical_HC = self.generate_line_eqns(vertical)

        corners_hc = np.zeros((1, 3))

        for i in range(horizontal.shape[0]):
            crs = np.cross(vertical_HC, horizonal_HC[i:i+1, :])
            corners_hc = np.vstack((corners_hc, crs))

        corners_hc = corners_hc[1:, :]
        corners_hc = corners_hc.T/corners_hc[:, 2]
        corners_hc = corners_hc.T

        return corners_hc

    def generate_line_eqns(self, lines):

        """
        Generate homogenous representation of line
        :param lines: N x 3 where col0 -> rho, col1 -> theta
        :return:
        """

        # x*cos(theta) + y*sin(theta) - rho
        line_eqn = [np.cos(lines[:, 1]), np.sin(lines[:, 1]), -1*lines[:, 0]]
        line_eqn = np.array(line_eqn)

        return line_eqn.T

    def plot_points(self, pts, img, label_pts=False):

        for i in range(pts.shape[0]):
            cv2.circle(img, (int(pts[i][0]), int(pts[i][1])), 2, (255, 0, 255), -1)
            if label_pts:
                cv2.putText(img, "{}".format(i), (int(pts[i][0])-5, int(pts[i][1])-5), 0, 0.5, (255, 255, 0))

        return img

    def run(self, img_path):
        """
        MAIN Function to get corners in a image using intersection of hough lines
        :param img_path: Full path to image
        :return:
        """
        print("Processing corners for {}".format(img_path))
        fname = os.path.basename(img_path)
        fname = fname.split('.')[0]

        img = cv2.imread(img_path)
        horizontal, vertical = self.get_horiz_vert_lines(img, outimg_name='lines_'+ fname + '.png')

        corners_hc = self.get_intersection_of_lines(horizontal, vertical)

        world_crd_hc = self.generate_world_crd(num_horiz=self.num_horiz, num_vert=self.num_vert, dist=self.dist)

        #####

        img = self.plot_points(corners_hc, img, label_pts=True)

        crnr_fldr = os.path.join(self.results_dir, 'corners')
        if not os.path.exists(crnr_fldr):
            os.makedirs(crnr_fldr)

        cv2.imwrite(os.path.join(crnr_fldr, 'corners_'+ fname + '.jpg'), img)

        ######

        print("Processing corners for {} ----------------------- Done! ".format(img_path))

        return corners_hc, world_crd_hc


if __name__ == "__main__":

    data_fldr = "/Users/aartighatkesar/Documents/Camera_Calibration/Dataset_1"
    results_fldr = os.path.join(data_fldr, 'results')

    corner_obj = GetCorners(results_fldr, num_horiz=10, num_vert=8, dist=25)

    for x in os.listdir(data_fldr):
        if x.endswith('.jpg'):
            corner_obj.run(os.path.join(data_fldr, x))

    # corner_obj.run(os.path.join(data_fldr, 'Pic_1.jpg'))

