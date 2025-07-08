import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

Sys_Prompt = """ You are a video reasoning assistant specialized in analyzing and comparing motion in videos.
You must reason step by step about the motion occurring in each video, and then compare the motion patterns.
Always focus primarily on human and object movement, action sequence, and timing.
Your final answer should clearly summarize the key differences in motion and assign a similarity score out of 100.
"""

User_Prompt = """ You are given two videos: Video A and Video B.

Please follow these steps:

1. Describe the motion in **Video A** step-by-step. Focus on what the subject does, the direction and intensity of movement, and the action sequence.
2. Do the same for **Video B**, using the same level of detail.
3. Compare the two videos specifically in terms of motion. Highlight:
  - Any differences in action types
  - Timing or duration of actions
  - Number of people or moving objects
  - Direction, speed, or style of movement
4. Based on your comparison, provide a **similarity score** out of 100 that reflects how similar the two videos are in terms of motion only.

**Output Format**:
--- Video A Motion Description ---
[step-by-step motion description]

--- Video B Motion Description ---
[step-by-step motion description]

--- Motion Comparison ---
[detailed analysis of key differences in movement, style, duration, direction, etc.]

--- Similarity Score (Motion Only) ---
Score: [XXX]/100
"""

def initialize_vlm_model(model_name='llava-hf/LLaVA-NeXT-Video-7B-hf') :
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        ).to(0).eval()
    processor = LlavaNextVideoProcessor.from_pretrained(model_name)
    return model, processor


def vlm_inference(model, processor, videoA_path, videoB_path, sys_prompt=Sys_Prompt, user_prompt=User_Prompt):
    messages = [
        {
            "role": "system",
            "content" : [
                {"type" : "text" , "text" : sys_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "path": videoA_path},
                {"type": "video", "path": videoB_path},
                {"type": "text", "text": user_prompt },
            ],
        },
    ]

    inputs = processor.apply_chat_template(messages, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=1024)
    text = processor.decode(output[0], skip_special_tokens=True)
    return text
