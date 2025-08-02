from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import cv2
import glob
from datetime import datetime
import tempfile
from typing import Optional
from PIL import Image
import base64
from io import BytesIO
from pydantic import BaseModel
import requests
import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import scipy.io.wavfile
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Import object recognition functions
from model_utils.object_recognition import (
    InitializeObjectRecognitionModel, 
    recognize_objects_in_image, 
    display_recognition_results,
    is_iou_above_threshold,
    compare_images_iou,
    compare_boxes_iou
)

# Import gesture recognition functions
from model_utils.model import recognize_gestures_video
from interpolation_utils.interpolation import interpolate_missing_frames

# Initialize models at startup
model = None
processor = None
tts_tokenizer = None
tts_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, processor, tts_tokenizer, tts_model
    model, processor = InitializeObjectRecognitionModel()
    print("Object recognition model initialized successfully")
    
    # Initialize TTS model
    try:
        tts_tokenizer = VitsTokenizer.from_pretrained("VIZINTZOR/MMS-TTS-THAI-MALEV2", cache_dir="./mms")
        tts_model = VitsModel.from_pretrained("VIZINTZOR/MMS-TTS-THAI-MALEV2", cache_dir="./mms").to("cuda")
        print("TTS model initialized successfully")
    except Exception as e:
        print(f"Warning: TTS model initialization failed: {e}")
        print("TTS endpoint will not be available")
    
    yield
    # Shutdown (if needed)
    print("Shutting down...")

app = FastAPI(title="Edumotion Backend API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = "uploaded_images"
CAPTURED_FRAMES_DIR = "captured_frames"
OUTPUT_DIR = "output_videos"

for directory in [UPLOAD_DIR, CAPTURED_FRAMES_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Pydantic models for request bodies
class TTSRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Edumotion Backend API is running"}

@app.post("/object-recognition/")
async def object_recognition_api(
    image: UploadFile = File(...),
    object_name: str = Query(..., description="Name of the object to recognize")
):
    """
    Object recognition endpoint that processes an uploaded image and detects specified objects.
    """
    import time
    start_time = time.time()
    
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        print(f"DEBUG: Starting single object recognition for '{object_name}'")
        
        # Save uploaded image temporarily
        file_save_start = time.time()
        temp_image_path = os.path.join(UPLOAD_DIR, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}")
        
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        file_save_time = time.time() - file_save_start
        print(f"DEBUG: File saving took {file_save_time:.2f} seconds")
        
        # Perform object recognition
        inference_start = time.time()
        print(f"DEBUG: Starting object recognition inference...")
        results = recognize_objects_in_image(temp_image_path, object_name, model, processor)
        inference_time = time.time() - inference_start
        print(f"DEBUG: Inference took {inference_time:.2f} seconds")
        
        # Process results for JSON response
        processing_start = time.time()
        response_data = {
            "object_name": object_name,
            "image_filename": image.filename,
            "detections": []
        }
        
        if results and len(results) > 0:
            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]
            
            for box, score, label in zip(boxes, scores, labels):
                detection = {
                    "bounding_box": [round(float(coord), 2) for coord in box.tolist()],
                    "confidence": round(float(score), 3),
                    "label": int(label)
                }
                response_data["detections"].append(detection)
        
        processing_time = time.time() - processing_start
        total_time = time.time() - start_time
        
        print(f"DEBUG: Found {len(response_data['detections'])} objects")
        print(f"DEBUG: Response processing took {processing_time:.2f} seconds")
        print(f"DEBUG: Total time: {total_time:.2f} seconds")
        
        # Add performance stats to response
        response_data["performance_stats"] = {
            "total_time_seconds": round(total_time, 2),
            "file_save_time_seconds": round(file_save_time, 2),
            "inference_time_seconds": round(inference_time, 2),
            "processing_time_seconds": round(processing_time, 2)
        }
        
        # Clean up temporary file
        cleanup_start = time.time()
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        cleanup_time = time.time() - cleanup_start
        print(f"DEBUG: Cleanup took {cleanup_time:.2f} seconds")
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/compare-objects/")
async def compare_objects_api(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    object_name: str = Query(..., description="Name of the object to compare"),
    threshold: float = Query(0.5, description="IoU threshold for comparison")
):
    """
    Compare objects in two images using IoU threshold.
    """
    import time
    start_time = time.time()
    
    try:
        # Validate file types
        if not image1.content_type.startswith('image/') or not image2.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Both files must be images")
        
        print(f"DEBUG: Starting object comparison for '{object_name}' with threshold {threshold}")
        
        # Save uploaded images temporarily
        file_save_start = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_image1_path = os.path.join(UPLOAD_DIR, f"temp1_{timestamp}_{image1.filename}")
        temp_image2_path = os.path.join(UPLOAD_DIR, f"temp2_{timestamp}_{image2.filename}")
        
        with open(temp_image1_path, "wb") as buffer1:
            shutil.copyfileobj(image1.file, buffer1)
        with open(temp_image2_path, "wb") as buffer2:
            shutil.copyfileobj(image2.file, buffer2)
        
        file_save_time = time.time() - file_save_start
        print(f"DEBUG: File saving took {file_save_time:.2f} seconds")
        
        # Process first image
        inference1_start = time.time()
        print(f"DEBUG: Starting object recognition for image 1...")
        results1 = recognize_objects_in_image(temp_image1_path, object_name, model, processor)
        inference1_time = time.time() - inference1_start
        print(f"DEBUG: Image 1 inference took {inference1_time:.2f} seconds")
        print(f"DEBUG: Image 1 found {len(results1[0]['boxes']) if results1 and len(results1) > 0 else 0} objects")
        
        # Process second image
        inference2_start = time.time()
        print(f"DEBUG: Starting object recognition for image 2...")
        results2 = recognize_objects_in_image(temp_image2_path, object_name, model, processor)
        inference2_time = time.time() - inference2_start
        print(f"DEBUG: Image 2 inference took {inference2_time:.2f} seconds")
        print(f"DEBUG: Image 2 found {len(results2[0]['boxes']) if results2 and len(results2) > 0 else 0} objects")
        
        # Extract and display detection results
        def extract_detections(results, image_name):
            detections = []
            if results and len(results) > 0:
                boxes = results[0]["boxes"]
                scores = results[0]["scores"]
                labels = results[0]["labels"]
                
                for box, score, label in zip(boxes, scores, labels):
                    detection = {
                        "bounding_box": [round(float(coord), 2) for coord in box.tolist()],
                        "confidence": round(float(score), 3),
                        "label": int(label),
                        "description": f"Detected a photo of a {object_name} with confidence {round(float(score), 3)} at location {[round(float(coord), 2) for coord in box.tolist()]}"
                    }
                    detections.append(detection)
                    print(f"DEBUG: {image_name} - {detection['description']}")
            return detections
        
        image1_detections = extract_detections(results1, "Image1")
        image2_detections = extract_detections(results2, "Image2")
        
        # Compare results with detailed IoU analysis
        comparison_start = time.time()
        print(f"DEBUG: Starting detailed IoU comparison...")
        
        # Get the maximum IoU score between any boxes
        max_iou_score = compare_boxes_iou(results1, results2, object_name)
        if max_iou_score is None:
            max_iou_score = 0.0
        
        # Calculate similarity percentage (0-1 scale)
        similarity_percentage = round(max_iou_score, 3)
        
        # Determine YES/NO based on simple threshold comparison
        is_above_threshold = "YES" if similarity_percentage > threshold else "NO"
        
        # Also get the complex IoU match for backward compatibility
        is_match = is_iou_above_threshold(results1, results2, object_name, threshold)
        
        comparison_time = time.time() - comparison_start
        print(f"DEBUG: IoU comparison took {comparison_time:.2f} seconds")
        print(f"DEBUG: Maximum IoU score: {max_iou_score:.3f}")
        print(f"DEBUG: Similarity percentage: {similarity_percentage:.3f}")
        print(f"DEBUG: Is above threshold ({threshold}): {is_above_threshold}")
        
        total_time = time.time() - start_time
        print(f"DEBUG: Total comparison time: {total_time:.2f} seconds")
        print(f"DEBUG: Breakdown - File save: {file_save_time:.2f}s, Image1: {inference1_time:.2f}s, Image2: {inference2_time:.2f}s, Comparison: {comparison_time:.2f}s")
        
        response_data = {
            "object_name": object_name,
            "threshold": threshold,
            "images_match": is_match,
            "is_above_threshold": is_above_threshold,
            "similarity_percentage": similarity_percentage,
            "max_iou_score": round(max_iou_score, 3),
            "image1_filename": image1.filename,
            "image2_filename": image2.filename,
            "image1_detections": image1_detections,
            "image2_detections": image2_detections,
            "comparison_summary": {
                "total_boxes_image1": len(image1_detections),
                "total_boxes_image2": len(image2_detections),
                "best_match_iou": round(max_iou_score, 3),
                "similarity_description": f"The bounding boxes are {round(max_iou_score * 100, 1)}% similar"
            },
            "performance_stats": {
                "total_time_seconds": round(total_time, 2),
                "file_save_time_seconds": round(file_save_time, 2),
                "image1_inference_seconds": round(inference1_time, 2),
                "image2_inference_seconds": round(inference2_time, 2),
                "comparison_time_seconds": round(comparison_time, 2)
            }
        }
        
        # Clean up temporary files
        cleanup_start = time.time()
        for temp_path in [temp_image1_path, temp_image2_path]:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        cleanup_time = time.time() - cleanup_start
        print(f"DEBUG: Cleanup took {cleanup_time:.2f} seconds")
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        # Clean up temporary files in case of error
        for temp_path in [temp_image1_path, temp_image2_path]:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        
        raise HTTPException(status_code=500, detail=f"Error comparing images: {str(e)}")

@app.post("/create-video-from-frames/")
async def create_video_from_latest_frames(
    fps: int = Query(30, description="Frames per second for the output video"),
    duration_seconds: int = Query(10, description="Duration of the video in seconds")
):
    """
    Create a video from the latest captured frames (last 10 seconds by default).
    """
    try:
        # Get all frame files from captured_frames directory
        frame_patterns = [
            os.path.join(CAPTURED_FRAMES_DIR, "*.jpg"),
            os.path.join(CAPTURED_FRAMES_DIR, "*.jpeg"),
            os.path.join(CAPTURED_FRAMES_DIR, "*.png"),
            os.path.join(CAPTURED_FRAMES_DIR, "*.bmp")
        ]
        
        all_frames = []
        for pattern in frame_patterns:
            all_frames.extend(glob.glob(pattern))
        
        if not all_frames:
            raise HTTPException(status_code=404, detail="No frames found in captured_frames directory")
        
        # Sort frames by modification time (latest first)
        all_frames.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Calculate how many frames we need for the specified duration
        total_frames_needed = fps * duration_seconds
        
        # Take the latest frames (up to the required number)
        latest_frames = all_frames[:min(total_frames_needed, len(all_frames))]
        
        if len(latest_frames) < fps:  # Need at least 1 second worth of frames
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough frames available. Found {len(latest_frames)}, need at least {fps} for 1 second"
            )
        
        # Reverse to get chronological order (oldest to newest of the selected frames)
        latest_frames.reverse()
        
        # Read the first frame to get dimensions
        first_frame = cv2.imread(latest_frames[0])
        if first_frame is None:
            raise HTTPException(status_code=400, detail="Could not read the first frame")
        
        height, width, channels = first_frame.shape
        
        # Create output video path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"latest_frames_video_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise HTTPException(status_code=500, detail="Could not initialize video writer")
        
        frames_written = 0
        # Write frames to video
        for frame_path in latest_frames:
            frame = cv2.imread(frame_path)
            if frame is not None:
                # Resize frame if necessary to match the first frame dimensions
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                video_writer.write(frame)
                frames_written += 1
        
        # Release video writer
        video_writer.release()
        
        # Verify the video file was created
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Failed to create video file")
        
        response_data = {
            "message": "Video created successfully",
            "video_filename": output_filename,
            "video_path": output_path,
            "frames_used": frames_written,
            "total_frames_available": len(all_frames),
            "duration_seconds": frames_written / fps,
            "fps": fps,
            "resolution": f"{width}x{height}"
        }
        
        return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating video: {str(e)}")

@app.get("/download-video/{filename}")
async def download_video(filename: str):
    """
    Download a generated video file.
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='video/mp4'
    )

@app.get("/list-videos/")
async def list_videos():
    """
    List all generated video files.
    """
    try:
        video_files = []
        if os.path.exists(OUTPUT_DIR):
            for filename in os.listdir(OUTPUT_DIR):
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    file_stats = os.stat(file_path)
                    video_files.append({
                        "filename": filename,
                        "size_bytes": file_stats.st_size,
                        "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                        "download_url": f"/download-video/{filename}"
                    })
        
        return {"videos": video_files}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing videos: {str(e)}")

@app.get("/frames-info/")
async def get_frames_info():
    """
    Get information about available frames in the captured_frames directory.
    """
    try:
        frame_patterns = [
            os.path.join(CAPTURED_FRAMES_DIR, "*.jpg"),
            os.path.join(CAPTURED_FRAMES_DIR, "*.jpeg"),
            os.path.join(CAPTURED_FRAMES_DIR, "*.png"),
            os.path.join(CAPTURED_FRAMES_DIR, "*.bmp")
        ]
        
        all_frames = []
        for pattern in frame_patterns:
            all_frames.extend(glob.glob(pattern))
        
        if not all_frames:
            return {"total_frames": 0, "frames": []}
        
        # Sort frames by modification time (latest first)
        all_frames.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        frames_info = []
        for frame_path in all_frames[:50]:  # Limit to latest 50 frames for response size
            file_stats = os.stat(frame_path)
            frames_info.append({
                "filename": os.path.basename(frame_path),
                "path": frame_path,
                "size_bytes": file_stats.st_size,
                "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            })
        
        return {
            "total_frames": len(all_frames),
            "frames": frames_info,
            "showing_latest": min(50, len(all_frames))
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting frames info: {str(e)}")

@app.post("/vlm-inference-comparison/")
async def vlm_inference_comparison(
    video: UploadFile = File(..., description="Video file to compare against combined frames"),
    threshold: float = Query(..., description="Threshold value for comparison")
):
    """
    VLM Inference Comparison Endpoint
    
    Combines the latest 10 seconds of captured frames into a video, then sends both
    the combined video (videoA) and uploaded video (videoB) to the external VLM inference API.
    
    Parameters:
    - video: Video file to compare
    - threshold: Threshold value for comparison
    
    Returns:
    - Response from the external VLM inference API
    """
    try:
        # Validate uploaded file
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        print(f"DEBUG: Starting VLM inference comparison with threshold: {threshold}")
        
        # Step 1: Create combined video from latest 10 seconds of frames
        print("DEBUG: Creating combined video from latest frames...")
        
        # Get all frame files from captured_frames directory
        frame_patterns = [
            os.path.join(CAPTURED_FRAMES_DIR, "*.jpg"),
            os.path.join(CAPTURED_FRAMES_DIR, "*.jpeg"),
            os.path.join(CAPTURED_FRAMES_DIR, "*.png"),
            os.path.join(CAPTURED_FRAMES_DIR, "*.bmp")
        ]
        
        all_frames = []
        for pattern in frame_patterns:
            all_frames.extend(glob.glob(pattern))
        
        if not all_frames:
            raise HTTPException(status_code=404, detail="No frames found in captured_frames directory")
        
        # Sort frames by modification time (latest first)
        all_frames.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Calculate frames needed for 10 seconds at 30 FPS
        fps = 30
        duration_seconds = 10
        total_frames_needed = fps * duration_seconds
        
        # Take the latest frames (up to 300 frames for 10 seconds)
        latest_frames = all_frames[:min(total_frames_needed, len(all_frames))]
        
        if len(latest_frames) < fps:  # Need at least 1 second worth of frames
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough frames available. Found {len(latest_frames)}, need at least {fps} for 1 second"
            )
        
        # Reverse to get chronological order (oldest to newest of the selected frames)
        latest_frames.reverse()
        
        # Read the first frame to get dimensions
        first_frame = cv2.imread(latest_frames[0])
        if first_frame is None:
            raise HTTPException(status_code=400, detail="Could not read the first frame")
        
        height, width, channels = first_frame.shape
        
        # Create combined video path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_video_filename = f"combined_frames_{timestamp}.mp4"
        combined_video_path = os.path.join(OUTPUT_DIR, combined_video_filename)
        
        # Initialize video writer for combined video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(combined_video_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise HTTPException(status_code=500, detail="Could not initialize video writer")
        
        # Write frames to combined video
        frames_written = 0
        print(f"DEBUG: Writing {len(latest_frames)} frames to combined video...")
        
        for frame_path in latest_frames:
            frame = cv2.imread(frame_path)
            if frame is not None:
                # Resize frame if necessary to match the first frame dimensions
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                video_writer.write(frame)
                frames_written += 1
            else:
                print(f"DEBUG: Failed to read frame: {frame_path}")
        
        video_writer.release()
        print(f"DEBUG: Combined video created with {frames_written} frames")
        
        # Verify the combined video file was created
        if not os.path.exists(combined_video_path):
            raise HTTPException(status_code=500, detail="Failed to create combined video file")
        
        # Step 2: Save uploaded video temporarily
        uploaded_video_filename = f"uploaded_{timestamp}_{video.filename}"
        uploaded_video_path = os.path.join(OUTPUT_DIR, uploaded_video_filename)
        
        with open(uploaded_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        print(f"DEBUG: Uploaded video saved as: {uploaded_video_filename}")
        
        # Step 3: Send both videos to external VLM inference API
        external_api_url = "https://oj42uymbfcas62-7860.proxy.runpod.net/vlm_inference_comp"
        
        try:
            # Prepare files and data for the external API (all as form-data)
            with open(combined_video_path, 'rb') as videoA_file, open(uploaded_video_path, 'rb') as videoB_file:
                # Prepare form-data exactly as required by the external API
                files = {
                    'videoA': (combined_video_filename, videoA_file, 'video/mp4'),
                    'videoB': (uploaded_video_filename, videoB_file, 'video/mp4'),
                    'threshold': (None, str(threshold))  # Send threshold as form-data string
                }
                
                print(f"DEBUG: Sending POST request to: {external_api_url}")
                print(f"DEBUG: Form-data fields:")
                print(f"  - videoA: {combined_video_filename} (video file)")
                print(f"  - videoB: {uploaded_video_filename} (video file)")
                print(f"  - threshold: '{threshold}' (string)")
                
                # Send POST request to external API with all data as form-data
                # No query parameters, no additional headers, just pure form-data
                response = requests.post(
                    external_api_url,
                    files=files,
                    timeout=300  # 5 minute timeout
                )
                
                print(f"DEBUG: Request sent successfully")
                print(f"DEBUG: Response status code: {response.status_code}")
                print(f"DEBUG: Response headers: {dict(response.headers)}")
                if response.status_code != 200:
                    print(f"DEBUG: Response content: {response.text[:500]}...")
                
                print(f"DEBUG: External API response status: {response.status_code}")
                
                if response.status_code == 200:
                    # Return the response from external API
                    try:
                        external_response = response.json()
                        print(f"DEBUG: External API JSON response received")
                    except:
                        external_response = {"response": response.text}
                        print(f"DEBUG: External API text response received")
                    
                    # Add metadata about our processing
                    result = {
                        "status": "success",
                        "combined_video_info": {
                            "filename": combined_video_filename,
                            "frames_used": frames_written,
                            "duration_seconds": frames_written / fps,
                            "fps": fps,
                            "resolution": f"{width}x{height}"
                        },
                        "uploaded_video_info": {
                            "filename": video.filename,
                            "saved_as": uploaded_video_filename
                        },
                        "threshold": threshold,
                        "external_api_response": external_response
                    }
                    
                    return JSONResponse(content=result)
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"External API error: {response.text}"
                    )
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to external API: {str(e)}"
            )
        
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(combined_video_path):
                    os.remove(combined_video_path)
                    print(f"DEBUG: Cleaned up combined video: {combined_video_filename}")
                if os.path.exists(uploaded_video_path):
                    os.remove(uploaded_video_path)
                    print(f"DEBUG: Cleaned up uploaded video: {uploaded_video_filename}")
            except Exception as cleanup_error:
                print(f"DEBUG: Error during cleanup: {cleanup_error}")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in VLM inference comparison: {str(e)}")

@app.post("/compare-gestures/")
async def compare_gestures_api(
    video1: UploadFile = File(..., description="First video file for gesture comparison"),
    video2: UploadFile = File(..., description="Second video file for gesture comparison")
):
    """
    Compare Gestures Between Two Videos
    
    Analyzes hand gestures in two video files and computes their similarity using DTW (Dynamic Time Warping).
    
    Parameters:
    - video1: First video file containing hand gestures
    - video2: Second video file containing hand gestures
    
    Returns:
    - DTW distance (lower = more similar)
    - Video analysis information
    - Performance statistics
    """
    import time
    start_time = time.time()
    
    try:
        # Validate file types
        if not video1.content_type.startswith('video/') or not video2.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Both files must be videos")
        
        print(f"DEBUG: Starting gesture comparison between '{video1.filename}' and '{video2.filename}'")
        
        # Save uploaded files
        file_save_start = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video1_path = os.path.join(UPLOAD_DIR, f"gesture1_{timestamp}_{video1.filename}")
        video2_path = os.path.join(UPLOAD_DIR, f"gesture2_{timestamp}_{video2.filename}")
        
        with open(video1_path, "wb") as buffer1:
            shutil.copyfileobj(video1.file, buffer1)
        with open(video2_path, "wb") as buffer2:
            shutil.copyfileobj(video2.file, buffer2)
        
        file_save_time = time.time() - file_save_start
        print(f"DEBUG: File saving took {file_save_time:.2f} seconds")
        
        # Process video 1
        video1_start = time.time()
        print(f"DEBUG: Processing video 1 for gesture recognition...")
        video1_coords, video1_indices = recognize_gestures_video(video1_path, visualize=False)
        video1_time = time.time() - video1_start
        print(f"DEBUG: Video 1 processing took {video1_time:.2f} seconds")
        print(f"DEBUG: Video 1 detected {len(video1_indices) if video1_indices else 0} gesture frames")
        
        # Process video 2
        video2_start = time.time()
        print(f"DEBUG: Processing video 2 for gesture recognition...")
        video2_coords, video2_indices = recognize_gestures_video(video2_path, visualize=False)
        video2_time = time.time() - video2_start
        print(f"DEBUG: Video 2 processing took {video2_time:.2f} seconds")
        print(f"DEBUG: Video 2 detected {len(video2_indices) if video2_indices else 0} gesture frames")
        
        # Interpolate missing frames
        interpolation_start = time.time()
        print(f"DEBUG: Starting frame interpolation...")
        
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
        
        interpolation_time = time.time() - interpolation_start
        print(f"DEBUG: Interpolation took {interpolation_time:.2f} seconds")
        print(f"DEBUG: Video 1 array shape: {video1_array.shape}")
        print(f"DEBUG: Video 2 array shape: {video2_array.shape}")
        
        # Check if gestures were detected
        if video1_array.size == 0 or video2_array.size == 0:
            error_msg = "No hand detected in "
            if video1_array.size == 0 and video2_array.size == 0:
                error_msg += "both videos"
            elif video1_array.size == 0:
                error_msg += "video 1"
            else:
                error_msg += "video 2"
            
            # Clean up files
            for video_path in [video1_path, video2_path]:
                if os.path.exists(video_path):
                    os.remove(video_path)
            
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Calculate DTW distance
        dtw_start = time.time()
        print(f"DEBUG: Calculating DTW distance...")
        distance, path = fastdtw(video1_array, video2_array, dist=euclidean)
        dtw_time = time.time() - dtw_start
        print(f"DEBUG: DTW calculation took {dtw_time:.2f} seconds")
        print(f"DEBUG: DTW distance: {distance:.3f}")
        
        total_time = time.time() - start_time
        print(f"DEBUG: Total gesture comparison time: {total_time:.2f} seconds")
        
        # Prepare response
        response_data = {
            "dtw_distance": round(float(distance), 3),
            "video1_info": {
                "filename": video1.filename,
                "gesture_frames_detected": len(video1_indices) if video1_indices else 0,
                "array_shape": list(video1_array.shape),
                "has_gestures": video1_array.size > 0
            },
            "video2_info": {
                "filename": video2.filename,
                "gesture_frames_detected": len(video2_indices) if video2_indices else 0,
                "array_shape": list(video2_array.shape),
                "has_gestures": video2_array.size > 0
            },
            "similarity_analysis": {
                "dtw_distance": round(float(distance), 3),
                "similarity_description": f"DTW distance of {distance:.3f} (lower values indicate more similar gestures)"
            },
            "performance_stats": {
                "total_time_seconds": round(total_time, 2),
                "file_save_time_seconds": round(file_save_time, 2),
                "video1_processing_seconds": round(video1_time, 2),
                "video2_processing_seconds": round(video2_time, 2),
                "interpolation_time_seconds": round(interpolation_time, 2),
                "dtw_calculation_seconds": round(dtw_time, 2)
            }
        }
        
        # Clean up uploaded files
        cleanup_start = time.time()
        for video_path in [video1_path, video2_path]:
            if os.path.exists(video_path):
                os.remove(video_path)
        cleanup_time = time.time() - cleanup_start
        print(f"DEBUG: Cleanup took {cleanup_time:.2f} seconds")
        
        return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        try:
            for video_path in [video1_path, video2_path]:
                if 'video_path' in locals() and os.path.exists(video_path):
                    os.remove(video_path)
        except:
            pass
        
        print(f"DEBUG: Error in gesture comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing gestures: {str(e)}")

@app.post("/tts/")
async def text_to_speech_json(request: TTSRequest):
    """
    Text-to-Speech endpoint that sends text to external TTS API and returns audio file.
    
    Request Body:
    - text: The text to convert to speech
    
    Returns:
    - Audio file from external TTS API
    """
    try:
        # Validate input
        if not request.text or request.text.strip() == "":
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # External TTS API URL
        external_tts_url = "https://oj42uymbfcas62-7860.proxy.runpod.net/tts"
        
        # Prepare request body for external API
        tts_payload = {
            "text": request.text.strip()
        }
        
        print(f"DEBUG: Sending TTS request to: {external_tts_url}")
        print(f"DEBUG: Request payload: {tts_payload}")
        
        # Send POST request to external TTS API
        response = requests.post(
            external_tts_url,
            json=tts_payload,
            timeout=60,  # 1 minute timeout
            headers={"Content-Type": "application/json"}
        )
        
        print(f"DEBUG: TTS API response status: {response.status_code}")
        print(f"DEBUG: TTS API response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            # Check if response is audio content
            content_type = response.headers.get('content-type', '')
            print(f"DEBUG: Response content-type: {content_type}")
            
            if 'audio' in content_type or 'wav' in content_type or 'mp3' in content_type:
                # Save the audio content to a temporary file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                audio_filename = f"tts_output_{timestamp}.wav"
                audio_path = os.path.join(OUTPUT_DIR, audio_filename)
                
                # Write audio content to file
                with open(audio_path, 'wb') as audio_file:
                    audio_file.write(response.content)
                
                print(f"DEBUG: Audio file saved as: {audio_filename}")
                
                # Return the audio file
                return FileResponse(
                    path=audio_path,
                    filename=audio_filename,
                    media_type='audio/wav',
                    headers={"Content-Disposition": f"attachment; filename={audio_filename}"}
                )
            else:
                # If not audio, return the response as JSON
                try:
                    return response.json()
                except:
                    return {"response": response.text}
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"External TTS API error: {response.text}"
            )
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to external TTS API: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in TTS processing: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=True)
