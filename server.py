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

# Import object recognition functions
from model_utils.object_recognition import (
    InitializeObjectRecognitionModel, 
    recognize_objects_in_image, 
    display_recognition_results,
    is_iou_above_threshold,
    compare_images_iou
)

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
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded image temporarily
        temp_image_path = os.path.join(UPLOAD_DIR, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}")
        
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Perform object recognition
        results = recognize_objects_in_image(temp_image_path, object_name, model, processor)
        
        # Process results for JSON response
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
        
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
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
    try:
        # Validate file types
        if not image1.content_type.startswith('image/') or not image2.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Both files must be images")
        
        # Save uploaded images temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_image1_path = os.path.join(UPLOAD_DIR, f"temp1_{timestamp}_{image1.filename}")
        temp_image2_path = os.path.join(UPLOAD_DIR, f"temp2_{timestamp}_{image2.filename}")
        
        with open(temp_image1_path, "wb") as buffer1:
            shutil.copyfileobj(image1.file, buffer1)
        with open(temp_image2_path, "wb") as buffer2:
            shutil.copyfileobj(image2.file, buffer2)
        
        # Compare images
        is_match = compare_images_iou(temp_image1_path, temp_image2_path, object_name, model, processor, threshold)
        
        response_data = {
            "object_name": object_name,
            "threshold": threshold,
            "images_match": is_match,
            "image1_filename": image1.filename,
            "image2_filename": image2.filename
        }
        
        # Clean up temporary files
        for temp_path in [temp_image1_path, temp_image2_path]:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
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
        external_api_url = "https://nwljdfqrcl8o1p-7860.proxy.runpod.net/"
        
        try:
            # Prepare files for the external API
            with open(combined_video_path, 'rb') as videoA_file, open(uploaded_video_path, 'rb') as videoB_file:
                files = {
                    'videoA': (combined_video_filename, videoA_file, 'video/mp4'),
                    'videoB': (uploaded_video_filename, videoB_file, 'video/mp4')
                }
                
                data = {
                    'threshold': threshold
                }
                
                print(f"DEBUG: Sending request to external API: {external_api_url}")
                print(f"DEBUG: videoA: {combined_video_filename}, videoB: {uploaded_video_filename}, threshold: {threshold}")
                
                # Send POST request to external API
                response = requests.post(
                    external_api_url,
                    files=files,
                    data=data,
                    timeout=300  # 5 minute timeout
                )
                
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

@app.post("/tts/")
async def text_to_speech_json(request: TTSRequest):
    """
    Text-to-Speech endpoint that accepts JSON body with text to convert to audio using Thai TTS model.
    
    Request Body:
    - text: The text to convert to speech
    
    Returns:
    - Audio file as response
    """
    try:
        # Check if TTS model is available
        if tts_tokenizer is None or tts_model is None:
            raise HTTPException(
                status_code=503, 
                detail="TTS model is not available. Please check server logs for initialization errors."
            )
        
        # Validate input
        if not request.text or request.text.strip() == "":
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Tokenize the input text
        inputs = tts_tokenizer(text=request.text.strip(), return_tensors="pt").to("cuda")
        
        # Set seed for deterministic output
        set_seed(456)
        
        # Generate audio
        with torch.no_grad():
            outputs = tts_model(**inputs)
        
        waveform = outputs.waveform[0]
        
        # Convert PyTorch tensor to NumPy array
        waveform_array = waveform.cpu().numpy()
        
        # Create temporary file for the audio
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_filename = f"tts_output_{timestamp}.wav"
        audio_path = os.path.join(OUTPUT_DIR, audio_filename)
        
        # Save audio file
        scipy.io.wavfile.write(
            audio_path, 
            rate=tts_model.config.sampling_rate, 
            data=waveform_array
        )
        
        # Verify the audio file was created
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Failed to create audio file")
        
        # Return the audio file
        return FileResponse(
            path=audio_path,
            filename=audio_filename,
            media_type='audio/wav',
            headers={"Content-Disposition": f"attachment; filename={audio_filename}"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=True)
