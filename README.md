---
layout: post
title:  "Revisiting Data Normalization for Appearance-Based Gaze Estimation"
date:   2022-09-06 10:45:31 +0900
category: Paper Review
tags: [Deep Learning, Computer Vision, Gaze, Normalization]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

![Paper Front Page](/assets/posts/paper_review/5.gazenormalization/FrontPage.png "Paper Front Page")

**The Cauchy-Schwarz Inequality**

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

## 1. Intro (Context)
Revisiting Data Normalization for Appearance-Based Gaze Estimation ([Source Paper](https://perceptualui.org/publications/zhang18_etra.pdf))

### 1-1) Appearance-based gaze estimation  
1. What is appearance-based gaze estimation?  
- **3 main types**
  - 3D eye model recovery-based method
  - 2D eye feature regression-based method
  - appearance based method
- **Characteristics of Alternatives**
  - Require personal calibration (for 3D eye model recovery)
  - Require dedicated devices (infrared cameras)
  - Directly use the detected geometric eye features (pupil center and glint) to regress the point of gaze
- **Characteristics of Appearance-based Methods**
  - Do not require dedicated devices (web cameras)
  - Feature extractor to extract gaze features from high-dimensional raw image data
  - Robust regression function to learn the mappings from appearance to human gaze
  - Large number of training samples to train the regression model

2. Challenges
  - Variability in head pose
  - user-camera distance  

3. Normalization
  - Cancel out geometric variability: Map input images to normalized space

**Paper states**
> Role and importance of data normalization remains unclear  

And claims to study for the first time using principled evaluations on both simulated and real data. Proposes modification to remove the scaling factor and the new formulation performs significantly better (between 9.5% and 32.7%). 


### 1-2) Recent development of appearance-based gaze estimation  
Fixed head pose -> fixed distance between user and camera -> Real-world environment

> Large data contains sufficient variability but manual collection and annotation is costly


### 1-3) Image resizing is equivalent to 3D scaling rather than 3D shifting.
![3D Scaling](/assets/posts/paper_review/5.gazenormalization/fig1.distance_resize_scale.png "3D Scaling")  
*Although there is some difference between (b), taken at distance 2d compared to (d), taken at distance d and resized, there is very little difference between (d) and (c) which was scaled while being taken at distance d*



## 2. Methodology
> The key idea is to standardize the translation and rotation between camera and face coordinate system via camera rotation and scaling.

### 2-1) Eye Image Normalization
![Normalization Overview](/assets/posts/paper_review/5.gazenormalization/fig2.normalization_overview.png "Normalization Overview")

Starts from an arbitrary pose of the target face.  
x-axis: line connecting midpoints of the two eyes from right eye to left eye  
y-axis: perpendicular to the x-axis inside the triangle plane from the eye to the mouth  
z-axis: perpendicular to the triangle and pointing backwards from the face.  

$$e_{r}$$: The midpoint of the right eye as the origin of the head coordinate system.  
$$R_{r}$$: Translation and rotation from the camera coordinate system to the head coordinate system.  

**Normalized image must meet three conditions**
1. The normalized camera looks at the origin of the head coordinate system and the center of the eye is located at the center of the normalized image.
2. The x-axes of the head and camera coordinate systems are on the same plane (x-axis of the head coordinate system appears horizontal)
3. The normalized camera is located at a fixed distance $d_{n} from the eye center and the eye always has the same size.

$$z_{c}$$: Rotated z-axis of the camera coordinate system  
$$x_{r}$$: x-axis of the head coordinate system  

$$z_{c}$$ has to be $$e_{r}$$  
$$y_{c} = z_{c} \times x_{r} $$   
$$y_{c} \perp z_{c}$$, $$y_{c} \perp x_{r}$$  
$$x_{c} = y_{c} \times z_{c} $$  

![R Equation](/assets/posts/paper_review/5.gazenormalization/fig3.requation.png "R Equation")  
$$S = \begin{bmatrix} 1 & 0 & 0\\0 & 1 & 0\\0 & 0 & \frac{d_{n}}{\lVert e_{r}\rVert}\end{bmatrix}$$  
$$M = SR $$  

-> This is for a 3D face mesh

**If the input is a 2D face image,**  
Image normalization is achieved via perspective warping  
$$W = C_{n}MC^{-1}_{r}$$  
$$C_{r}$$: Original camera projection matrix from camera calibration  
$$C_{n}$$: Camera projection matrix for the normalized camera

### 2-2) Modified Data Normalization  
It is important to handle the geometric transformation caused by the eye image normalization and apply the same transformation to the gaze direction vector.

$$g_{r}$$: ground-truth gaze direction vector  
$$g_{n}$$: normalized gaze vector  

For 3D space, this was previously proposed:  
$$g_{n} = Mg_{r}$$  

$$R_{n}$$: Head rotation matrix  
Since scaling does not affect the rotation matrix, the head rotation matrix after normalization is computed only with rotation as
$$R_{n} = RR_{r}$$  

**A modified version of data normalization for 2d images is proposed**  
Propose to only rotate the original gaze vector to obtain the normalized gaze vector **without scaling matrix**.  
$$g_{n} = Rg_{r}$$  

Which can be interpreted as applying the S matrix to $$C_{r}$$ instead of the 3D coordinate system.  
$$W = (C_{n}S) (RC^{-1}_{r})$$

## 3. Performance
Since this paper emphasizes itself for being the first to quantize the effects of normalization, the boosted performance is an essential component of this paper.

### 3-1) Evaluation on Synthetic Images
> real-world datasets inevitably have limited head pose variations due to device constraints. To fully evaluate the effect of data normalization on gaze estimation performance, we first use synthetic eye images with controlled head pose variations.

![Result with Synthetic Data](/assets/posts/paper_review/5.gazenormalization/fig8.synthetic_result.png)  
*The modified normalization method performs 5.9 degrees better in Mean error*

![Result with Synthetic Data with Different Distances](/assets/posts/paper_review/5.gazenormalization/fig9.syntehtic_result_distance.png)  
*The modified normalization method is stable across different distances between the eye and the camera*

### 3-2) Evaluation on Real Images
![Result with Real Data](/assets/posts/paper_review/5.gazenormalization/fig10.real_result.png)  
*Network is fine-tuned and tested on MPIIGaze. Although the modified version performs 0.5 degrees better in Mean error, the difference is much smaller compared to the results evaluated with synthetic data*

## 4. Code
### 4-1) Pseudo Code
Building a program that takes in images and applies the normalization technique requires multiple steps.   

1. Pre-defined data & Hyperparameters  
There are some data that we need pre-defined in order to test the program  
- **3D Coordinates of face** (n, 3): n is the number of 3D points of the face. The coordinates should mimic the normal human face in 3D
- **Camera Matrix** (3, 3): This is $$C_{r}$$, the original camera projection matrix from camera calibration
- **Camera Distortion** (1,4): This is the camera distortion to later use in [solvePnP](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d) to estimate the head pose. The length could be different, check the [official docs](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d) for more reference.
- **Template Landmark Index** (m; m < n): Indexes in the landmark provided by MediaPipe's [FaceMesh API](https://google.github.io/mediapipe/solutions/face_mesh) to later use for drawing critical points on the face and building the 3D face coordinates.
- **Face Key Landmark Index** (l ; l <  m): Key indexes in the face landmark that will be used to calculate the center of the face (both in 2D and 3D) 
- **Hyperparameters**
  - focal_norm (int): The focal values that will be used in $$C_{n}$$ (the camera projection matrix for the normalized camera)
  - distance_norm (int): The normalized distance for the normalized processes to be used in S (scaling matrix)
  - roiSize (2): the size of the normalized image (used in $$C_{n}$$ to center the image)

2. Code Architecture  
- **Preparation Process**
  - Video is captured using [OpenCV](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a5d5f5dacb77bbebdcbfb341e3d4355c1) and an image is read
  - The image is processed with MediaPipe's FaceMesh API  
  - The resulting landmarks will be filtered with our pre-defined "Temple Landmark Index" and scaled to represent 2D pixel coordinates  
  - The key landmark points in the face is drawn  
  - The 2D landmark points are resized to 3D and processed with the 3D face using SolvePnP to get the rotation and translation vector  
  > **solvePnP**  
  > - World coordinates $$X_{w}, Y_{w}, Z_{w}$$ are projected to image plane (u,v)  
  > $$\begin{align*} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} &= \bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{T}_w \begin{bmatrix} X_{w} \\ Y_{w} \\ Z_{w} \\ 1 \end{bmatrix} \\ \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} &= \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_{w} \\ Y_{w} \\ Z_{w} \\ 1 \end{bmatrix} \end{align*}$$
  > - rotational vector (rvec) and translational vector (tvec) are computed that transforms a 3D object from world coordinate system to the camera coordinate system  
  > $$\begin{align*} \begin{bmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{bmatrix} &= \hspace{0.2em} ^{c}\bf{T}_w \begin{bmatrix} X_{w} \\ Y_{w} \\ Z_{w} \\ 1 \end{bmatrix} \\ \begin{bmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{bmatrix} &= \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_{w} \\ Y_{w} \\ Z_{w} \\ 1 \end{bmatrix} \end{align*}$$  
- **Normalization Process**
  - The rotation vector is transformed to a rotation matrix using [rodrigues](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac) and the 3D face coordinates are generated and the 3D coordinates of the center of the face is calculated  
  > **Rodrigues Transformation**  
  > $$\theta := \lVert r \rVert$$  
  > $$r := \frac{r}{\theta} $$   
  > A := [0, -rz, ry; rz, 0, -rx; -ry, rx, 0]  
  > $$R := cos(\theta)I + (1-cos(\theta))rr^T + sin(\theta)A$$  
  > * The rotation matrix is given with the rotation vector
  - The distance between the eye and the camera is calculated with normalization and the scaling matrix is calculated  
  - The R matrix is calculated starting with $$Z_{c}$$ with a series of cross multiplication  
  - The transformation matrix W is calculated with $$C_{n}MC^{-1}_{r}$$
  - The final normalized image is generated using [warpPerspective](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87)
  > **Perspective Transformation**
  ![Perspective Transformation Overview](/assets/posts/paper_review/5.gazenormalization/fig11.warp_overview.png)  
  ![Perspective Transformation Vectors](/assets/posts/paper_review/5.gazenormalization/fig12.warp_vectors.png)  
  ![Perspective Transformation Illustration](/assets/posts/paper_review/5.gazenormalization/fig13.warp_display.png)  



### 4-2) Sample Code
- Preparation Process
1.  Video is captured and processed using OpenCV and MediaPipe  
```python
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        results = process_image(image, face_mesh)
```
This is a pretty straightforward process where the front camera of the computer starts recording video and captures on image, which is then processed with MediaPipe's FaceMesh.  
The result is a hashmap, which has a property *multi_face_landmarks* with size (n) where n is the number of faces.  
Each face in *multi_face_landmarks* has property *landmark* that stores the landmarks size (n, 2) where n is the number of landmarks. 
2. Landmarks are converted to 2D pixel coordinates  
```python
def face_detect(image_shape, multi_face_landmark):
    height = image_shape[0]
    width = image_shape[1]
    landmarks = np.empty((0,2), dtype=np.float64)
    for index in TEMPLATE_LANDMARK_INDEX:
        landmark = multi_face_landmark.landmark[index]
        landmarks = np.append(landmarks, np.array([[min(width, landmark.x*width), min(height, landmark.y*height)]]), axis=0)
    return landmarks, face_center(landmarks)
```
We are only extracting the landmarks we are interested in, which is sorted in *TEMPLATE_LANDMARK_INDEX*.  
The 2D pixel coordinates are calculated by multiplying the width and height of the image.  
All of the landmarks are appended in an array *landmarks*   
3. The head pose is estimated  
```python
landmarks = landmarks.astype(np.float32)
landmarks = landmarks.reshape(num_pts, 1, 2)
def estimateHeadPose(landmarks, face_model, camera, distortion, iteration=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    if iteration:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)
    return rvec, tvec
``` 
The landmarks are first reshaped to (n, 1, 2) where n is the number of points in the 3D face model.  
With the 3D face model(n, 1, 3), key landmarks(n, 1, 2), camera matrix(3, 3), distortion(5,) the rotational vector and translational vector are returned.


- Normalization Process
1. Get the 3D coordinates of the face with the camera coordinate system  
```python
# Pose translation matrix
ht = ht.reshape((3, 1))
# Pose rotation vector converted to a rotation matrix
hR = cv2.Rodrigues(hr)[0]
Fc = np.dot(hR, face) + ht
```
2. Calculate the distance from the eye to the camera and the scaling matrix S  
```python
# actual distance bcenterween eye and original camera
distance = np.linalg.norm(center)
z_scale = distance_norm/distance
# scaling matrix
S = np.array([ 
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, z_scale],
])
```
3. Calculate the rotational matrix R  
```python
# z-axis
forward = (center/distance).reshape(3)
# x_r: x-axis of the head coordinate system
hRx = hR[:,0]
# y-axis
down = np.cross(forward, hRx)
down /= np.linalg.norm(down)
# x-axis
right = np.cross(down, forward)
right /= np.linalg.norm(right)
# rotation matrix R
R = np.c_[right, down, forward].T
```
4. Calculate the transformation matrix W, warp the image to get the final normalized image  
```python
# transformation matrix
W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix))) 
# image normalization
img_warped = cv2.warpPerspective(img, W, roiSize)
return img_warped
```

The complete code can be found [here]().

### 4-3) Results  
1. Image With Landmark Annotations  
![Sample Image with Landmark Annotations](/assets/posts/paper_review/5.gazenormalization/samples/sample_original.png)  
2. Normalized Image  
![Normalized Image](/assets/posts/paper_review/5.gazenormalization/samples/sample_normalized.png)    
3. Sample Video  
<iframe width="560" height="315" src="https://www.youtube.com/embed/UfjoRIEpn7s" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>  

The model successfully annotates the original image and normalizes it by rotating the roll and maintaining a constant distance resulting in a consistent image size.

### 4-4) Tests
1. Distance
- Image from a far distance
![Distant Image](/assets/posts/paper_review/5.gazenormalization/tests/1.distance.png)   
- Normalized image
![Distant Normalized Image](/assets/posts/paper_review/5.gazenormalization/tests/2.distance_normalized.png)   
2. MaskSample Video  
<iframe width="560" height="315" src="https://www.youtube.com/embed/qfxD4_KBv9k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
The normalized image is a little unstable (due to occluded landmarks from the mask)
3. Occlusion
<iframe width="560" height="315" src="https://www.youtube.com/embed/5alYwgWpxB4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
There is great instability when there is occlusion of the face. I tested which part of the face being occluded resulted in the most distortion, affecting the normalization.

## 5. Challenging Aspects
1. Method
Vector, Camera Coordinates & World Coordinates
Matrix Multiplication
2. Code
MediaPipe, Conversion of 2D Landmark points to Pixel Points

## 6. Thoughts
There could be more ways to normalize image data.
Besides learning, data pre-processing is also very important.
Intuition for neural networks to learn well is to restrict the differences between the training data to learn.

## 7. References
1. [Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark](https://arxiv.org/abs/2104.12668)
2. [PnP Pose Computation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
3. [Rodrigues](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac)
4. [Perspective Transformation](https://theailearner.com/tag/cv2-warpperspective/)
5. [Cross Product](https://mathinsight.org/cross_product)
6. [Image Formation and Cameras](https://courses.cs.washington.edu/courses/csep576/11sp/pdf/ImageFormation.pdf)
7. [Camera Matrix](https://en.wikipedia.org/wiki/Camera_matrix)
