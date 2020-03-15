# Camera Calibration - Estimation of Camera Extrinsic and Intrinsic Parameters

#### Description:
Implementation of Zhang's Camera Calibration algorithm to estimate a camera's extrinsic parameter matrices "R", "t" and intrinsic parameter matrix "K". It uses a Calibration pattern of checkerboard to estimate these parameters
```
0. Generate dataset of images by using the Calibration Pattern in different views
1. Generate Hough lines and separate horizontal from vertical
2. From the hough lines, get rid of duplicate lines and then find intersection of horizontal and vertical lines to get corners corrdinates on image plane
3. Generate world coordinates of the corresponding corners in the image plane
4. Find Homography H between world plane Z=0 and image plane
5. Using this homography and the image of absolute conic, solve equations to estimate the image (omega) of the absolute conic.
6. Using the relationship between omega and K, obtain the intrinsic parameter matrix K. This is a estimate and will be refined further
7. Now, build an initial estimate of R and t matrices using K and H
8. Condition matrix R to be orthonormal
9. Convert the matrix R to a vector R_vec using Rodrigues formula
10. Build initial estimate of all the parameters, 5 for K, 3*no_of_images in dataset for R and 3*no_of_images_in_dataset for t
11. If incorporating radial distortion, the also include parameters k1 and k2
12. Optimize the geometric error between the actual corners and projected coordinates of world coordinates in the image plane using Levenberg - Marquardt algorithm
13. Rebuild the matrices R, t for each image from the 1D vector obtained after optimization. Note that the R_vect should be converted to R matrix using Rodrigues formula
14. Rebuild estimated K matrix from 1D vector after optimization
```

#### Original Paper: [**Zhang_technical_report.pdf**](./Zhang_technical_report.pdf)

Refer [here](./My%20Notes) for the mathematical formulation and technical explanation


#### Dependencies

- OpenCV
- NumPy
- SciPy (for Levenberg-Marquardt implementation)

## Scripts
- [**calibrate_camera.py**](./calibrate_camera.py): **_MAIN_** file to run. Pass the correct parent folder and image list in the script. Result folder with all intermediate images are generated in /<parent_folder>/results_<fixed_img_no>
```python
python calibrate_camera.py
```

###### Supporting scripts
- [**compute_corners.py**](./compute_corners.py): Script to get coordinates of corners by intersection of lines obtained using Hough transform
- [**estimate_homography.py**](./estimate_homography.py): Helper functions which help in bilinear interpolation and projecting images to a canvas using Homography matrix

## Results

- [Input dataset - click click](./Dataset_1)
- [Results for Input Dataset - more clicks](./results)

#### Input Images
|<img src="https://github.com/aartighatkesar/Camera_Calibration/blob/in-progress/images_for_readme/input_imgs/Pic_28.jpg" alt="Pic_28.jpg" width="320" height="240" /> Pic_28.jpg
|<img src="https://github.com/aartighatkesar/Camera_Calibration/blob/in-progress/images_for_readme/input_imgs/Pic_40.jpg" alt="Pic_40.jpg" width="320" height="240" /> Pic_40.jpg  

#### Hough Lines and Processing

|<img src="https://github.com/aartighatkesar/Camera_Calibration/blob/in-progress/images_for_readme/corners_lines/lines_Pic_28.png" alt="Lines before processing" width="400" height="600"/>

|<img src="https://github.com/aartighatkesar/Camera_Calibration/blob/in-progress/images_for_readme/corners_lines/processed_lines_Pic_28.jpg" alt="Lines after processing" width="320" height="240" />


#### Corners and refinement

Corners before (found by intersection of hough lines) and after refinement 
|<img src="https://github.com/aartighatkesar/Camera_Calibration/blob/in-progress/images_for_readme/corners_lines/corners_afterPic_28.jpg" alt="Corners after processing" width="320" height="240" />
