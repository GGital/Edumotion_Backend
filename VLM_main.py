from model_utils.VLM import initialize_vlm_model, vlm_inference , Sys_Prompt, User_Prompt

model , processor = initialize_vlm_model()
videoA_path = 'Test_medias/hand1.mp4'
videoB_path = 'Test_medias/hand2.mp4'

result = vlm_inference(model, processor, videoA_path, videoB_path, sys_prompt=Sys_Prompt, user_prompt=User_Prompt)
print(result)