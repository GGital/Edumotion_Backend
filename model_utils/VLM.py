import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

sys_prompt_comp = """ You are a video reasoning assistant specialized in analyzing and comparing motion in videos.
You must reason step by step about the motion occurring in each video, and then compare the motion patterns.
Always focus primarily on human and object movement, action sequence, and timing.
Your final answer should clearly summarize the key differences in motion and assign a similarity score out of 100.
"""

user_prompt_comp = """ You are given two videos: Video A and Video B.

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

sys_prompt_ask = """ You are a vision-language assistant specialized in analyzing video quality and content clarity.
You will watch a video and extract fine-grained details, such as small object interactions, subtle movements, facial expressions, or precise gestures.
You must assess how clearly these minor details are represented and assign a score based on how well the video conveys them.
Be fair, detailed, and reasoned in your grading.
"""

user_prompt_ask = """ Please analyze the following video for minor or subtle visual details.

1. Identify any small or subtle elements you notice in the video. These may include:
   - Hand gestures, facial expressions, or posture changes
   - Object handling or minor movements
   - Visual cues in the environment (e.g., signs, symbols, props)
2. Explain whether each identified detail is **clear and easy to observe**, or **unclear/ambiguous** due to motion, angle, or visibility.
3. Based on your analysis, assign a score out of 100 for how well the video presents these minor details.

**Output Format**:
--- Minor Detail Observations ---
[List of small or subtle visual elements you identified]

--- Representation Assessment ---
[For each, explain how clearly it is shown and why — including visibility, motion clarity, and framing]

--- Minor Detail Clarity Score ---
Score: [XX]/100
"""


def initialize_vlm_model(model_name='llava-hf/LLaVA-NeXT-Video-7B-hf') :
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        ).to(0).eval()
    processor = LlavaNextVideoProcessor.from_pretrained(model_name)
    return model, processor


def vlm_inference_comp(model, processor, videoA_path, videoB_path, sys_prompt=Sys_Prompt_Comp, user_prompt=User_Prompt_Comp):
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

def vlm_inference_ask(model, processor, video_path, sys_prompt=sys_prompt_ask, user_prompt=user_prompt_ask):
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
                {"type": "video", "path": video_path},
                {"type": "text", "text": user_prompt },
            ],
        },
    ]

    inputs = processor.apply_chat_template(messages, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=1024)
    text = processor.decode(output[0], skip_special_tokens=True)
    return text
