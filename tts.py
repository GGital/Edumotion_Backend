import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import scipy

tokenizer = VitsTokenizer.from_pretrained("VIZINTZOR/MMS-TTS-THAI-MALEV2",cache_dir="./mms")
model = VitsModel.from_pretrained("VIZINTZOR/MMS-TTS-THAI-MALEV2",cache_dir="./mms").to("cuda")

inputs = tokenizer(text="สวัสดีครับ นี่คือเสียงพูดภาษาไทย", return_tensors="pt").to("cuda")

set_seed(456)  # make deterministic

with torch.no_grad():
   outputs = model(**inputs)

waveform = outputs.waveform[0]

# Convert PyTorch tensor to NumPy array
waveform_array = waveform.cpu().numpy()

scipy.io.wavfile.write("yayee.wav", rate=model.config.sampling_rate, data=waveform_array)