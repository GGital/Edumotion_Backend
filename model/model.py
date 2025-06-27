import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from visualization_utils.visualization import display_image_with_gestures_and_hand_landmarks
import numpy as np
import cv2

def initialize_gesture_recognition_model(model_path):
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO)
    gesture_recognizer = vision.GestureRecognizer.create_from_options(options)
    return gesture_recognizer

def recognize_gestures_video(video_path, model_path='models_weights/gesture_recognizer.task'):
    gesture_recognizer = initialize_gesture_recognition_model(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    coordinates_array = []
    frame_indices = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_idx = 0
    while True : 
        ret, frame = cap.read()
        if not ret:
            break
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = gesture_recognizer.recognize_for_video(image , 
                                                         timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        if results.gestures:
            image_result, array = display_image_with_gestures_and_hand_landmarks(image, results)
            array = np.array(array).reshape(21, 3)
            wrist = array[0]
            array -= wrist
            norm_factor = np.linalg.norm(array[9]) + 1e-6
            array /= norm_factor
            coordinates_array.append(array.flatten())
            frame_indices.append(current_frame_idx)
            cv2.imshow('Gesture Recognition', image_result)
        else:
            cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        current_frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()
    return coordinates_array, frame_indices
