import estimate_homography
from compute_corners import GetCorners

import os
import numpy as np

class CalibrateCamera:

    def __init__(self, dataset_dir, num_horiz, num_vert, dist):

        self.results_dir = os.path.join(dataset_dir, "results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.dataset_dir = dataset_dir

        self.num_horiz = num_horiz
        self.num_vert = num_vert
        self.dist = dist

        self.imgs_dataset = [x for x in os.listdir(self.dataset_dir) if x.endswith('.jpg')]

        self.num_imgs_datset = len(self.imgs_dataset)

        self.H = {}


    def _calculate_H_per_image(self, img_crds, world_crds):
        """
        Calculate the Homography between image plane and world plane
        :param img_crds: N x 3 HC cordinates
        :param world_crds: N x 4 HC coordinates
        :return: H : 3 x 3 matrix

        img_crd = [p1, p2, p3, p4] * [x, y, 0, w].T =>
        img_crd = [p1, p2, p4] * x_w =>
        img_crd = H * x_w
        x_w is HC 3-vector representation of a world point in the Z = 0 world plane
        """

        H = estimate_homography.calculate_homography(img_crds[:, 0:2], world_crds[:, 0:2])

        return H


    def calculate_H_dataset(self):
        """
        Function to calculate the Homography between plane in 3D and image plane

        Stores H in self.H dictionary (key is image name)
        :return:
        """

        corner_dir = os.path.join(self.results_dir, "output_corners")

        corner_Obj = GetCorners(results_dir=corner_dir, num_horiz=self.num_horiz, num_vert=self.num_vert, dist=self.dist)

        for img_name in self.imgs_dataset:
            img_path = os.path.join(self.dataset_dir, img_name)

            corners_hc, world_crd_hc = corner_Obj.run(img_path)
            print("Calculating Homography")
            self.H[img_name] = self._calculate_H_per_image(corners_hc, world_crd_hc)
            print("Calculating Homography -------------- Done !")
            print("----------------------------------------------")



    def _build_equtions_abs_conic_img(self):
        pass

    def _calculate_image_abs_conic(self):
        pass

    def calculate_camera_intrinsic_params_K(self):

        self.calculate_H_dataset()

        # print(self.H)
        pass



if __name__ == "__main__":

    dataset_dir = "/Users/aartighatkesar/Documents/Camera_Calibration/Dataset_1"
    num_horiz = 10
    num_vert = 8
    dist = 25

    calibration_obj = CalibrateCamera(dataset_dir, num_horiz, num_vert, dist)
    calibration_obj.calculate_camera_intrinsic_params_K()

