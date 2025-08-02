from model_utils.VLM import *
from model_utils.post_processor import extract_score_percentage

model , processor = initialize_vlm_model()
videoA_path = 'Test_medias/hand1.mp4'
videoB_path = 'Test_medias/hand2.mp4'

result = vlm_inference_comp(model, processor, videoA_path, videoB_path, sys_prompt=sys_prompt_comp, user_prompt=user_prompt_comp)
print(result)

result = vlm_inference_ask(model , processor, videoA_path, sys_prompt=sys_prompt_ask, user_prompt=user_prompt_ask)
print(result)

print(extract_score_percentage(result))