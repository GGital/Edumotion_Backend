
from fastapi import FastAPI, UploadFile, File, Form
from model_utils.VLM import *
from model_utils.post_processor import *
import shutil
import os
import torch

app = FastAPI()

model , processor = initialize_vlm_model()

@app.post("/vlm_inference_comp")
async def vlm_inference_comp_endpoint(
    videoA: UploadFile = File(...),
    videoB: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    videoA_path = f"temp_{videoA.filename}"
    videoB_path = f"temp_{videoB.filename}"
    with open(videoA_path, "wb") as buffer:
        shutil.copyfileobj(videoA.file, buffer)
    with open(videoB_path, "wb") as buffer:
        shutil.copyfileobj(videoB.file, buffer)
    result = vlm_inference_comp(model, processor, videoA_path, videoB_path, sys_prompt=sys_prompt_comp, threshold=threshold)
    os.remove(videoA_path)
    os.remove(videoB_path)
    result = convert_escaped_newlines(result[0])
    sections = parse_video_comparison_output(result, threshold)
    torch.cuda.empty_cache()
    return sections

@app.post("/vlm_inference_ask")
async def vlm_inference_ask_endpoint(
    video: UploadFile = File(...),
):
    video_path = f"temp_{video.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    result = vlm_inference_ask(model, processor, video_path, sys_prompt=sys_prompt_ask, user_prompt=user_prompt_ask)
    os.remove(video_path)
    torch.cuda.empty_cache()
    return {"result": result, "score": extract_score_percentage(result)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)