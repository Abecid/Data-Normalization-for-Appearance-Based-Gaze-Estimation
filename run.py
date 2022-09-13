import numpy as np
import cv2
import mediapipe as mp

from template.template_data import CAMERA_MATRIX, CAMERA_DISTORTION, TEMPLATE_LANDMARK_INDEX, TEMPLATE_FACE_3D_V1

FACE_KEY_LANDMARK_INDEX = [0, 9, 20]

camera_matrix = np.array(CAMERA_MATRIX) 
camera_distortion = np.array(CAMERA_DISTORTION)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Drawing const parameters
point_drawing = {'radius': 2, 'rgb': [0, 255, 0], 'thickness': 1}
point_center_drawing = {'radius': 5, 'rgb': [0, 0, 255], 'thickness': 2}

def draw_point(image, mat, spec):
    rgb = spec['rgb']
    for arr in mat:
        x = int(arr[0])
        y = int(arr[1])
        # Image is in BGR, flip the RGB order
        cv2.circle(image, (x, y), radius=spec['radius'], color=(rgb[2], rgb[1], rgb[0]), thickness=spec['thickness'])

def generate_3d_face():
    face = np.empty((0, 3), dtype=np.float64)
    for index in TEMPLATE_LANDMARK_INDEX:
        face = np.append(face, np.array([TEMPLATE_FACE_3D_V1[index]]), axis=0)
    return face

def process_image(image, face_mesh):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return face_mesh.process(image)

def estimateHeadPose(landmarks, face_model, camera, distortion, iteration=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    if iteration:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def normalizeFace(img, face, hr, ht, camera_matrix):
    focal_norm = 960
    distance_norm = 130
    roiSize=(300, 300)

    # Pose translation matrix
    ht = ht.reshape((3, 1))

    # Pose rotation vector converted to a rotation matrix
    hR = cv2.Rodrigues(hr)[0]
    Fc = np.dot(hR, face) + ht
    
    center = np.zeros(np.array(Fc[:,0]).shape)
    for index in FACE_KEY_LANDMARK_INDEX:
        center += np.array(Fc[:,index])
    center = np.array([ center / len(FACE_KEY_LANDMARK_INDEX) ]).reshape((3,1))

    # actual distance bcenterween eye and original camera
    distance = np.linalg.norm(center)
    z_scale = distance_norm/distance

    # C_n: camera projection matrix for the normalized camera
    cam_norm = np.array([
       [focal_norm, 0, roiSize[0]/2],
       [0, focal_norm, roiSize[1]/2],
       [0, 0, 1.0],
    ])

    # scaling matrix
    S = np.array([ 
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, z_scale],
    ])
    
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

    # transformation matrix
    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix))) 
        
    # image normalization
    img_warped = cv2.warpPerspective(img, W, roiSize)
    return img_warped

def face_detect(image_shape, multi_face_landmark):
    height = image_shape[0]
    width = image_shape[1]

    landmarks = np.empty((0,2), dtype=np.float64)
    for index in TEMPLATE_LANDMARK_INDEX:
        landmark = multi_face_landmark.landmark[index]
        landmarks = np.append(landmarks, np.array([[min(width, landmark.x*width), min(height, landmark.y*height)]]), axis=0)
    return landmarks, face_center(landmarks)

def face_center(landmarks):
    center = np.zeros(np.array(landmarks[0]).shape, dtype=np.float64)
    for index in FACE_KEY_LANDMARK_INDEX:
        center += np.array(landmarks[index])
    return np.array([ center / len(FACE_KEY_LANDMARK_INDEX) ])

def main():
    # Find 3D Standard Face Points
    face = generate_3d_face()
    num_pts = face.shape[0]
    facePts = face.T.reshape(num_pts, 1, 3)

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            results = process_image(image, face_mesh)

            if results.multi_face_landmarks:
                # Get key face landmarks in 2D pixels
                landmarks, center = face_detect(image.shape, results.multi_face_landmarks[0])
                draw_point(image, landmarks, point_drawing)
                draw_point(image, center, point_center_drawing)
                
                # Convert 2D landmark pixels to 3D
                landmarks = landmarks.astype(np.float32)
                landmarks = landmarks.reshape(num_pts, 1, 2)

                # Get rotational/translation shift
                hr, ht = estimateHeadPose(landmarks, facePts, camera_matrix, camera_distortion)
                # Normalization
                processed_face = normalizeFace(image, face.T, hr, ht, camera_matrix)

                # Show camera image with landmarks
                cv2.imshow("Cam image", image)
                # Show normalized image
                cv2.imshow("normalized Image", processed_face)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == '__main__':
    main()
