from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import numpy as np
from model_utils.model import recognize_gestures_video
from interpolation_utils.interpolation import interpolate_missing_frames
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

app = FastAPI()

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/compare-gestures/")
def compare_gestures(
    video1: UploadFile = File(...),
    video2: UploadFile = File(...)
):
    # Save uploaded files
    video1_path = os.path.join(UPLOAD_DIR, video1.filename)
    video2_path = os.path.join(UPLOAD_DIR, video2.filename)
    with open(video1_path, "wb") as buffer1:
        shutil.copyfileobj(video1.file, buffer1)
    with open(video2_path, "wb") as buffer2:
        shutil.copyfileobj(video2.file, buffer2)

    # Process videos
    video1_coords, video1_indices = recognize_gestures_video(video1_path , visualize=False)
    video2_coords, video2_indices = recognize_gestures_video(video2_path , visualize=False)

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

    if video1_array.size == 0 or video2_array.size == 0:
        return JSONResponse(status_code=400, content={"error": "No hand detected in one or both videos."})

    distance, path = fastdtw(video1_array, video2_array, dist=euclidean)

    # Optionally, remove uploaded files after processing
    os.remove(video1_path)
    os.remove(video2_path)

    return {"dtw_distance": distance, "video1_shape": video1_array.shape, "video2_shape": video2_array.shape}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
