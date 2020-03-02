import estimate_homography
from compute_corners import GetCorners

import os
import numpy as np
import cv2
from scipy import optimize


class CalibrateCamera:

    def __init__(self, dataset_dir, num_horiz, num_vert, dist, radial_dist=True, fixed_img="Pic_11.jpg"):

        self.results_dir = os.path.join(dataset_dir, "results_{}".format(fixed_img.split(".jpg")[0].split('_')[1]))
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.dataset_dir = dataset_dir

        self.num_horiz = num_horiz
        self.num_vert = num_vert
        self.dist = dist

        self.fixed_img = fixed_img

        self.imgs_dataset = [x for x in os.listdir(self.dataset_dir) if x.endswith('.jpg')]

        self.fix_id = self.imgs_dataset.index(fixed_img)

        self.num_imgs_datset = len(self.imgs_dataset)

        self.H = {}

        self.Rt = [[] for _ in range(self.num_imgs_datset)]  # ordered according to imgs_dataset list

        self.K = 0

        self.corners_hc = [[] for _ in range(self.num_imgs_datset)]

        self.world_hc = 0

        self.K_final = []

        self.Rt_final = []

        self.k1_k2 = np.array([0, 0])

        self.k1_k2_final = []

        self.radial_dist = radial_dist


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

        for i, img_name in enumerate(self.imgs_dataset):
            img_path = os.path.join(self.dataset_dir, img_name)

            self.corners_hc[i], world_crd_hc = corner_Obj.run(img_path)
            print("Calculating Homography")
            self.H[img_name] = self._calculate_H_per_image(self.corners_hc[i], world_crd_hc)
            print("Calculating Homography -------------- Done !")
            print("----------------------------------------------")

        self.world_hc = world_crd_hc


    def _build_equations_img_abs_conic(self):

        def _build_matrix_v_ij(H, i, j):
            """
            Function to build matrix v_ij (Eqn 8 in paper) # Warning: the numbers and i,j should be flipped in equation
            :param H: 3 x 3 homography matrix
            :return: v based on eqn 8. (6,) ndarray
            """
            v = [
                H[0][i - 1] * H[0][j - 1],
                H[0][i - 1] * H[1][j - 1] + H[1][i - 1] * H[0][j - 1],
                H[1][i - 1] * H[1][j - 1],
                H[2][i - 1] * H[0][j - 1] + H[0][i - 1] * H[2][j - 1],
                H[2][i - 1] * H[1][j - 1] + H[1][i - 1] * H[2][j - 1],
                H[2][i - 1] * H[2][j - 1]
            ]

            return np.array(v)

        V = []

        for key in self.H.keys():
            V.append(_build_matrix_v_ij(self.H[key], 1, 2))
            V.append(_build_matrix_v_ij(self.H[key], 1, 1) - _build_matrix_v_ij(self.H[key], 2, 2))

        V = np.array(V)

        assert V.shape[0] == 2*self.num_imgs_datset and V.shape[1] == 6, 'Incorrect dimensions for matrix V'

        return V

    def _calculate_img_abs_conic(self):
        """
        Function to build equations for solving for omega (image of absolute conic)
        :return:
        """
        print("Calculating image of absolute conic")

        mat_V = self._build_equations_img_abs_conic()  # Eqn 8 in paper

        U, sig, Vt = np.linalg.svd(mat_V)  # Solution to mat_V*b is the last col of V
        # which corresponds to smallest eigen value of mat_V based on Linear Least Squares solution

        V = Vt.T

        b = V[:, -1]  # [w11, w12, w22, w13, w23, w33]

        def _build_matrix_w(b):
            W = np.zeros((3,3))
            W[0][0] = b[0]  # w11
            W[0][1] = W[1][0] = b[1]  # w12, w21
            W[1][1] = b[2] # w22
            W[0][2] = W[2][0] = b[3] # w13, w31
            W[1][2] = W[2][1] = b[4]  # w23, w32
            W[2][2] = b[5] # w33

            return W

        omega = _build_matrix_w(b)

        return omega


    def _calculate_K_from_img_abs_conic(self, omega):

        print("Calculating K from image of absolute conic")

        v0 = (omega[0][1]*omega[0][2] - omega[0][0]*omega[1][2])/(omega[0][0]*omega[1][1] - omega[0][1]**2)

        lmda = omega[2][2] - ((omega[0][2]**2 + v0*(omega[0][1]*omega[0][2] - omega[0][0]*omega[1][2]))/omega[0][0])

        alpha = np.sqrt((lmda/omega[0][0]))

        beta = np.sqrt((lmda*omega[0][0])/(omega[0][0]*omega[1][1] - omega[0][1]**2))

        gamma = -1 * (omega[0][1]*(alpha**2)*beta)/lmda

        u0 = (gamma*v0/beta) - (omega[0][2]*(alpha**2)/lmda)

        K = [[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]

        return np.array(K)


    def _calculate_R_t_init(self):
        """
        Function to calculate initial R and T matrix
        :return:
        """

        def _build_initial_R_t(K, H):
            R = np.zeros((3, 3))
            t = np.zeros((3, 1))

            K_inv = np.linalg.inv(K)

            R[:, 0] = np.dot(K_inv, H[:, 0])  # r1 = Kinv*h1
            R[:, 1] = np.dot(K_inv, H[:, 1])  # r2 = Kinv *h2
            R[:, 2] = np.cross(R[:, 0], R[:, 1])  # r3 = r1 x r2
            t = np.dot(K_inv, H[:, 2])  # t = Kinv * h3

            scaling = 1 / np.linalg.norm(R[:, 0])

            R = R * scaling
            t = t * scaling

            return R, t

        Rt = [[] for _ in range(self.num_imgs_datset)]


        for i, img_name in enumerate(self.imgs_dataset):
            print("Processing initial R and t values for {} -> {}".format(img_name, i))
            R_out, t_out = _build_initial_R_t(self.K, self.H[img_name])

            R_out = self._condition_R_mat(R_out)

            Rt[i] = np.hstack((R_out, t_out[:, np.newaxis]))

        return Rt

    @staticmethod
    def _convert_R_mat_to_vec(R_mat):
        """
        Use rodiriguez formula to convert R matrix to R vector with 3 DoF
        :return:
        """
        phi = np.arccos((np.trace(R_mat)-1)/2)

        R_vec = np.array([R_mat[2][1] - R_mat[1][2], R_mat[0][2] - R_mat[2][0], R_mat[1][0] - R_mat[0][1]])

        R_vec = R_vec * (phi/(2*np.sin(phi)))

        # print("-------- R mat to vec-----------")
        # print(cv2.Rodrigues(R_mat)[0])
        # print(w)
        # print("-------- R mat to vec-----------")

        return R_vec


    @staticmethod
    def _convert_R_vec_to_mat(R_vec):
        """
        Function to convert R vector computed using Rodriguez formula back to a mtrix
        R_vec = [wx, wy, wz]
        :return:
        """

        phi = np.linalg.norm(R_vec)
        Wx = np.zeros((3,3))

        Wx[0][1] = -1*R_vec[2]
        Wx[0][2] = R_vec[1]

        Wx[1][0] = R_vec[2]
        Wx[1][2] = -1*R_vec[0]

        Wx[2][0] = -1*R_vec[1]
        Wx[2][1] = R_vec[0]

        R_mat = np.eye(3) + (np.sin(phi)/phi) * Wx + ((1-np.cos(phi))/phi**2)*np.dot(Wx, Wx)

        # print("-------- R vec to mat-----------")
        # print(cv2.Rodrigues(R_vec[:, np.newaxis])[0])
        # print(R_mat)
        # print("-------- R vec to mat-----------")

        return R_mat

    def _condition_R_mat(self, R):
        """
        Function to normalize computed matrix R
        :param R:
        :return:
        """
        U, sig, Vt = np.linalg.svd(R)

        return np.dot(U, Vt)

    def optimize_params(self):
        pass

    @staticmethod
    def compute_residuals(x, img_hc, world_hc, radial_dist=False):
        """

        :param x: np array - variables to optimize
        0:5 (K -> u0, v0, alpha, beta, gamma)
        Then 3 entries of R ,followed by 3 entries for t for each img
        If radial dist is True, then followed by k1, k2 for each img

        x = [alpha, gamma, u0, beta, v0,  r1_1, r2_1, r3_1, t1_1, t2_1, t3_1, r1_2, r2_2, r3_2, t1_2, t2_2, t3_2...... k1_1, k2_1, k1_2, k2_2,....]

        :param img_hc: List of lists of all actual img pts
        :param world_hc: np array of rows of world pts
        :param radial_dist: bool, whether to take radial distortion into account
        :return:
        """
        num_corners = world_hc.shape[0]
        num_imgs = len(img_hc)
        K_val = x[0:5]

        K = np.zeros((3,3))
        #  Build K
        K[0][0] = x[0]
        K[0][1] = x[1]
        K[0][2] = x[2]
        K[1][1] = x[3]
        K[1][2] = x[4]
        K[2][2] = 1


        # Build R, t for each img and compute P = K[R|t]
        P = [[] for _ in range(num_imgs)]

        for i in range(num_imgs):
            Rt_vec = x[5+i*6: 5+(i+1)*6]

            R = CalibrateCamera._convert_R_vec_to_mat(Rt_vec[0:3])
            Rt = np.hstack((R, Rt_vec[3:].reshape(3, 1)))

            P[i] = np.matmul(K, Rt)

        # if radial_dist:
        #
        #     # Build k1, k2 for each image
        #     num_params_krt = 5 + (6 * num_imgs)
        #     k1_k2 = [[] for _ in range(num_imgs)]
        #
        #     for i in range(num_imgs):
        #         k1_k2[i] = x[num_params_krt + i*2: num_params_krt + (i+1)*2]
        #
        #     k1_k2 = np.array(k1_k2)  # Num_imgs x 2
        #     CalibrateCamera.k1_k2 = k1_k2

        if radial_dist:
            # Build k1_k2
            num_params_krt = 5 + (6 * num_imgs)
            k1_k2 = x[num_params_krt:]


        # Compute projections per image, per corner

        # world_hc = <num_corners> rows of x, y, z, w
        # image = list of nd arrays. Each nd array has rows of [x, y, z]

        img_hc = np.array(img_hc)  #shape = Num_imgs, num_corners, 3
        assert img_hc.shape == (num_imgs, num_corners, 3)

        img_hc = np.swapaxes(img_hc, 1, 2)  # shape = num_imgs, 3, num_corners
        img_hc = img_hc[:, 0:2, :]


        P = np.array(P)  # shape = num_imgs, 3, 4

        proj_crd = np.matmul(P, world_hc.T)  # Proj_crd shape = num_imgs, 3, num_corners

        proj_crd = proj_crd/ proj_crd[:, 2:3, :]  # normalizing last crd

        proj_crd = proj_crd[:, 0:2, :]  # Getting physical coordinates


        if radial_dist:
            # Compute radial distortion
            princ_pt = np.array([K[0][2], K[1][2]]).reshape(2,1)

            radius_sq = np.sum((proj_crd-princ_pt)**2, axis=1)  # num_imgs x num_corners

            mul_term = radius_sq * k1_k2[0] + (radius_sq **2) * k1_k2[1]  # num_imgs x num_corners

            mul_term = mul_term[:, np.newaxis, :] # num_imgs x 1 x num_corners

            proj_crd = proj_crd + (proj_crd - princ_pt) * mul_term


        # compute residual
        residual = img_hc.ravel() - proj_crd.ravel()

        return residual

    def calculate_camera_intrinsic_params_K(self):

        print("---------------------------------------")
        print("Calculating Camera Intrinsic parameters")
        print("---------------------------------------")

        self.calculate_H_dataset()

        omega = self._calculate_img_abs_conic()

        self.K = self._calculate_K_from_img_abs_conic(omega)

        print("Calculating Camera Intrinsic parameters ------------  Done!")

        print("-----------------------------------------------------------------------------------------")

    def calculate_camera_extrinsic_params_R_t(self, radial_dist=False):

        print("---------------------------------------")
        print("Calculating Camera Extrinsic parameters")
        print("---------------------------------------")

        num_params = 5 + self.num_imgs_datset * 6  # 5 params for K and (3 DOF for R and 3 DOF for t)for each image

        x_init = np.zeros(num_params)

        #####  Initialize x_init with K values ######
        # K = [[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]
        x_init[0] = self.K[0][0]
        x_init[1] = self.K[0][1]
        x_init[2] = self.K[0][2]
        x_init[3] = self.K[1][1]
        x_init[4] = self.K[1][2]

        self.Rt = self._calculate_R_t_init()

        for i in range(self.num_imgs_datset):
            r_vec = self._convert_R_mat_to_vec(self.Rt[i][:, 0:3])
            x_init[5+i*6: 5+(i+1)*6] = np.hstack((r_vec, self.Rt[i][:, -1]))  # assign R vector and t of each image to init values


        if radial_dist:
            self.k1_k2 = np.zeros(2)
            x_init = np.hstack((x_init, self.k1_k2))

        sol = optimize.least_squares(CalibrateCamera.compute_residuals, x_init, args=(self.corners_hc, self.world_hc), kwargs={'radial_dist':radial_dist},
                                     method='lm',
                                     xtol=1e-15, ftol=1e-15)

        ### Build K, R, t from solution

        # K = [[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]
        self.K_final = np.zeros_like(self.K)
        self.K_final[0][0] = sol.x[0]
        self.K_final[0][1] = sol.x[1]
        self.K_final[0][2] = sol.x[2]
        self.K_final[1][1] = sol.x[3]
        self.K_final[1][2] = sol.x[4]
        self.K_final[2][2] = 1

        self.Rt_final = [[] for _ in range(self.num_imgs_datset)]

        for i in range(self.num_imgs_datset):

            Rt_i = sol.x[5+i*6: 5+(i+1)*6]
            R_i = self._convert_R_vec_to_mat(Rt_i[0:3])
            self.Rt_final[i] = np.hstack((R_i, Rt_i[3:].reshape(3,1)))

        # if radial_dist:
        #     self.k1_k2_final = [[] for _ in range(self.num_imgs_datset)]
        #     num_params_krt = 5 + self.num_imgs_datset * 6
        #     for i in range(self.num_imgs_datset):
        #         self.k1_k2_final[i] = sol.x[num_params_krt + i*2: num_params_krt + (i+1)*2]

        if radial_dist:
            num_params_krt = 5 + self.num_imgs_datset * 6
            self.k1_k2_final = sol.x[num_params_krt:]

        print("-------------------------------------")

        print("Inital_K: {}".format(self.K))
        print("final_K: {}".format(self.K_final))

        print("------")

        print("Inital_K1_k2: {}".format(self.k1_k2))
        print("final_K1_k2: {}".format(self.k1_k2_final))

        print("------")

        print("Inital_Rt: {}".format(self.Rt[0]))
        print("final_Rt: {}".format(self.Rt_final[0]))

        print("-------------------------------------")

        print("Calculating Camera Extrinsic parameters ------------  Done!")

        print("-----------------------------------------------------------------------------------------")

    def project_points_on_fixed_img(self, out_dir, radial_dist):

        if not radial_dist:
            project_fix_dir = os.path.join(out_dir, "fixed_img_pt_proj")

        else:
            project_fix_dir = os.path.join(out_dir, "fixed_img_pt_radial_dist")


        if not os.path.exists(project_fix_dir):
            os.makedirs(project_fix_dir)

        img = cv2.imread(os.path.join(self.dataset_dir, fixed_img))

        fix_orig_pts = self.corners_hc[self.fix_id]

        for i in range(fix_orig_pts.shape[0]):
            cv2.circle(img, (int(fix_orig_pts[i][0]), int(fix_orig_pts[i][1])), 2, (0, 255, 0), -1, cv2.LINE_AA)

        P_fix_id = np.matmul(self.K, self.Rt[self.fix_id])

        H_fix = np.zeros((3, 3))

        H_fix[:, 0] = self.Rt[self.fix_id][:, 0]
        H_fix[:, 1] = self.Rt[self.fix_id][:, 1]
        H_fix[:, 2] = self.Rt[self.fix_id][:, 3]

        H_fix = np.matmul(self.K, H_fix)

        ###

        H_fix_af = np.zeros((3, 3))

        H_fix_af[:, 0] = self.Rt_final[self.fix_id][:, 0]
        H_fix_af[:, 1] = self.Rt_final[self.fix_id][:, 1]
        H_fix_af[:, 2] = self.Rt_final[self.fix_id][:, 3]

        H_fix_af = np.matmul(self.K_final, H_fix_af)


        for id, img_name in enumerate(self.imgs_dataset):

            img_1 = np.copy(img)

            ## Before LM

            P_img = np.matmul(self.K, self.Rt[id])

            H = np.zeros((3, 3))

            H[:, 0] = self.Rt[id][:, 0]
            H[:, 1] = self.Rt[id][:, 1]
            H[:, 2] = self.Rt[id][:, 3]

            H = np.matmul(self.K, H)


            # proj_crd_init = np.matmul(np.matmul(P_fix_id, np.linalg.pinv(P_img)), self.corners_hc[id].T)  # 3 x num_corners

            proj_crd_init = np.matmul(np.matmul(H_fix, np.linalg.pinv(H)), self.corners_hc[id].T)

            proj_crd_init = proj_crd_init/proj_crd_init[-1, :]

            proj_crd_init = proj_crd_init[0:2, :]

            img_2 = np.copy(img_1)

            for i in range(proj_crd_init.shape[1]):
                cv2.circle(img_1, (int(proj_crd_init[0][i]), int(proj_crd_init[1][i])), 2, (255, 0, 0), -1, cv2.LINE_AA)

            ## Projected corners after LM
            # Compute projected corners after LM

            P_img = np.matmul(self.K, self.Rt[id])

            H = np.zeros((3, 3))

            H[:, 0] = self.Rt_final[id][:, 0]
            H[:, 1] = self.Rt_final[id][:, 1]
            H[:, 2] = self.Rt_final[id][:, 3]

            H = np.matmul(self.K_final, H)

            P_img_fin = np.matmul(self.K, self.Rt_final[id])

            proj_crd_fin = np.matmul(np.matmul(H_fix_af, np.linalg.pinv(H)), self.corners_hc[id].T)

            # proj_crd_fin = np.matmul(np.matmul(P_fix_id, np.linalg.pinv(P_img_fin)), self.corners_hc[id].T)  # 3 x num_corners

            proj_crd_fin = (proj_crd_fin / proj_crd_fin[-1, :])

            proj_crd_fin = proj_crd_fin[0:2, :]  # 2 x num_corners

            if radial_dist:
                princ = np.array([self.K_final[0][2], self.K_final[1][2]]).reshape(2, 1)  # 2 x 1

                rad_sq = np.sum((proj_crd_fin - princ) ** 2, axis=0, keepdims=True)  # 1 x num_corners

                mult_term = self.k1_k2_final[0] * rad_sq + self.k1_k2_final[1] * rad_sq ** 2

                proj_crd_fin = proj_crd_fin + (proj_crd_fin - princ) * mult_term

            for i in range(proj_crd_fin.shape[1]):
                cv2.circle(img_2, (int(proj_crd_fin[0][i]), int(proj_crd_fin[1][i])), 2, (255, 0, 0), -1,
                           cv2.LINE_AA)

            out_img_name = os.path.join(project_fix_dir, "{}_{}".format(img_name.split('.')[0], self.fixed_img))

            cv2.imwrite(out_img_name, np.hstack((img_1, img_2)))

    def project_world_pts(self, out_dir, radial_dist):

        if not radial_dist:
            project_wrld_dir = os.path.join(out_dir, "world_pt_proj")

        else:
            project_wrld_dir = os.path.join(out_dir, "world_pt_proj_radial_dist")


        if not os.path.exists(project_wrld_dir):
            os.makedirs(project_wrld_dir)

        for id, img_name in enumerate(self.imgs_dataset):

            img = cv2.imread(os.path.join(self.dataset_dir, img_name))

            # Draw expected corners

            actual_pts = self.corners_hc[id]

            for i in range(actual_pts.shape[0]):
                cv2.circle(img, (int(actual_pts[i][0]), int(actual_pts[i][1])), 2, (0, 255, 0), -1, cv2.LINE_AA)

            # Compute projected corners before LM
            P_init = np.matmul(self.K, self.Rt[id])

            proj_crd_init = np.matmul(P_init, self.world_hc.T)

            proj_crd_init = (proj_crd_init/proj_crd_init[-1, :])  #  3 x num_corners

            proj_crd_init = proj_crd_init[0:2, :]  #  2 x num_corners

            img_2 = np.copy(img)

            for i in range(proj_crd_init.shape[1]):
                cv2.circle(img, (int(proj_crd_init[0][i]), int(proj_crd_init[1][i])), 2, (255, 0, 0), -1, cv2.LINE_AA)

            # Projected corners after LM

            # Compute projected corners before LM
            P_fin = np.matmul(self.K_final, self.Rt_final[id])

            proj_crd_fin = np.matmul(P_fin, self.world_hc.T)

            proj_crd_fin = (proj_crd_fin / proj_crd_fin[-1, :])

            proj_crd_fin = proj_crd_fin[0:2, :]  # 2 x num_corners

            if radial_dist:
                princ = np.array([self.K_final[0][2], self.K_final[1][2]]).reshape(2, 1)  # 2 x 1

                rad_sq = np.sum((proj_crd_fin - princ) ** 2, axis=0, keepdims=True)  # 1 x num_corners

                mult_term = self.k1_k2_final[0] * rad_sq + self.k1_k2_final[1] * rad_sq ** 2

                proj_crd_fin = proj_crd_fin + (proj_crd_fin - princ) * mult_term

            for i in range(proj_crd_fin.shape[1]):
                cv2.circle(img_2, (int(proj_crd_fin[0][i]), int(proj_crd_fin[1][i])), 2, (255, 0, 0), -1,
                           cv2.LINE_AA)

            fin_img = os.path.join(project_wrld_dir, "world_proj_{}".format(img_name))
            cv2.imwrite(fin_img, np.hstack((img, img_2)))


    def run(self):
        self.calculate_camera_intrinsic_params_K()

        self.calculate_camera_extrinsic_params_R_t(radial_dist=self.radial_dist)

        out_dir = os.path.join(self.results_dir, "self_proj")

        self.project_world_pts(out_dir, radial_dist=self.radial_dist)

        out_dir = os.path.join(self.results_dir, "{}_proj".format(self.fixed_img.split('.')[0]))

        self.project_points_on_fixed_img(out_dir, radial_dist=self.radial_dist)


if __name__ == "__main__":

    dataset_dir = "/Users/aartighatkesar/Documents/Camera_Calibration/Dataset_1"
    num_horiz = 10
    num_vert = 8
    dist = 25
    fixed_img = "Pic_28.jpg"

    calibration_obj = CalibrateCamera(dataset_dir, num_horiz, num_vert, dist, radial_dist=False, fixed_img=fixed_img)
    calibration_obj.run()


