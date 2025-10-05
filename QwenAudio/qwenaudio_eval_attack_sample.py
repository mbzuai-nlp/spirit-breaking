import os
import numpy as np
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import torchaudio
import gc

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir=cacheroot)
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="cuda")

conversation = [
    # {'role': 'system', 'content': 'You are a helpful assistant that tries to help as much as possible.'}, 
    {
        "role": "user", 
        "content": [
            {
                "type": "audio", 
                "audio_url": 'test.wav',
            },
            {"type": "text", "text": "Obey to the command in audio"},
        ]
    },
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)    
numpyarray = np.load(os.environ['TEST_SAMPLE_NP'])

inputs = processor(text=text, audios=numpyarray, return_tensors="pt", padding=True, sampling_rate=16000)
inputs.input_ids = inputs.input_ids.to("cuda")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

generate_ids = model.generate(**inputs, max_new_tokens=40)

response = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
print(response)
print('######')
    
