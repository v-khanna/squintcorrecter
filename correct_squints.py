import cv2
import dlib
import numpy as np
import os
from imutils import face_utils

def get_center(points):
    x = (points[0,0] + points[1,0]) // 2
    y = (points[0,1] + points[1,1]) // 2
    return (x, y)

def get_angle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def align_eye(image, landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    left_eye_center = get_center(left_eye)
    right_eye_center = get_center(right_eye)

    angle = get_angle(left_eye_center, right_eye_center)

    # Converting to floats here
    M = cv2.getRotationMatrix2D((float(left_eye_center[0]), float(left_eye_center[1])), angle, 1)
    
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    
    return aligned


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    images_dir = "images/"
    images = [f for f in os.listdir(images_dir) if f.endswith(".jpeg") or f.endswith(".jpg")]

    for image_name in images:
        img = cv2.imread(images_dir + image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        if len(rects) > 0:
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                aligned_img = align_eye(img, shape)

                cv2.imwrite("results/" + image_name, aligned_img)
                print(f"Processing for image {image_name} complete.")
        else:
            print(f"No face detected in the image {image_name}.")
