from model_utils.model import recognize_gestures_video
from interpolation_utils.interpolation import interpolate_missing_frames
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

if __name__ == "__main__":
    video1_coords, video1_indices = recognize_gestures_video('Test_medias/hand1.mp4')
    video2_coords, video2_indices = recognize_gestures_video('Test_medias/hand2.mp4')

    if video1_indices:
        video1_array = interpolate_missing_frames(video1_coords, video1_indices)
    else:
        video1_array = []
    if video2_indices:
        video2_array = interpolate_missing_frames(video2_coords, video2_indices)
    else:
        video2_array = []

    video1_array = np.array(video1_array)
    video2_array = np.array(video2_array)

    print(video1_array.shape, video2_array.shape)

    distance, path = fastdtw(video1_array, video2_array, dist=euclidean)
    print(f"DTW distance between the two videos: {distance}")
