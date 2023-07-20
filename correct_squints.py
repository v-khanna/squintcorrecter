import cv2
import dlib
import numpy as np
from skimage import transform
import os

# Paths
images_path = './images'
results_path = './results'
landmarks_model_path = 'shape_predictor_68_face_landmarks.dat'

# Load the Haar cascade xml file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the facial landmarks predictor
predictor = dlib.shape_predictor(landmarks_model_path)

# For each image in the images directory
for filename in os.listdir(images_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Read the image
        img = cv2.imread(os.path.join(images_path, filename))

        # Convert color style from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform face detection
        faces = face_cascade.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # For each detected face
        for (x, y, w, h) in faces:
            # Get the facial landmarks
            landmarks = predictor(img_rgb, dlib.rectangle(x, y, x + w, y + h))

            # Get the coordinates of the corners of the left eye
            left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)], np.int32)

            # Compute the aspect ratio of the left eye to estimate if it's squinting
            eye_aspect_ratio = (np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])) / (2.0 * np.linalg.norm(left_eye[0] - left_eye[3]))

            # If the eye aspect ratio is less than a certain threshold, it might be a squint
            if eye_aspect_ratio < 0.2:  # you may need to experiment to find a suitable threshold
                # Stretch the eye vertically a bit
                eye_img = img_rgb[y:y+h, x:x+w]
                stretched_eye = transform.resize(eye_img, (h + 10, w))  # Increase height by 10 pixels
                img_rgb[y:y+h+10, x:x+w] = stretched_eye

        # Save the image to the results directory
        result_image_path = os.path.join(results_path, 'corrected_' + filename)
        cv2.imwrite(result_image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

