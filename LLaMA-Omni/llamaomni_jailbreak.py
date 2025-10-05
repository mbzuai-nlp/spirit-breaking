
import os
import pandas as pd

from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset

from omni_speech.model.builder import load_pretrained_model
from torchattacks.attacks.pgd_uap_v1_speech import *

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class CustomAudioModelWithNoise(nn.Module):
    def __init__(self, base_model, processor, target_response, input_response=None):
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        
        self.target_ids = processor(target_response[:-1].replace('Sure, ', '').capitalize(), return_tensors="pt").input_ids.to("cuda")
        self.targetlen = self.target_ids.shape[1]

        self.target_response = target_response
        self.input_response = input_response

class DataSetSpeechAttack(Dataset):
    def __init__(self, attack_prompts_file, audioroot, transform=None, target_transform=None):
        self.attack_prompt = pd.read_csv(attack_prompts_file, sep=',"', header=None, skiprows=1)
        self.attack_prompt.columns = [0, 1]
        self.audioroot = audioroot
        
        self.dataset_length = len(self.attack_prompt)
        self.target_transform = target_transform
            
    def conversation_wrapper(self, speech_path):
        conversation = [
            {
                "speech": speech_path,
                "conversations": [{"from": "human", "value": f"<speech>.\nPlease directly answer the questions in the user's speech."}],
            }
        ]
        
        return conversation

    def __len__(self):
        return len(self.attack_prompt)

    def __getitem__(self, idx):
        speech_path = os.path.join(self.audioroot, f'output_{idx}.wav')
        label = self.attack_prompt.loc[idx, 1]
        prompt = self.attack_prompt.loc[idx, 0]
    
        return {
            'inputs': self.conversation_wrapper(speech_path),
            'label': label,
            'prompt': prompt
        }
        
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--attack-prompts-file", type=str, required=True)
parser.add_argument("--audioroot", type=str, required=True)
parser.add_argument("--audiosaveroot", type=str, required=True)
parser.add_argument("--category", type=str, required=True)

parser.add_argument("--eps", type=float, default=0.05)
parser.add_argument("--alpha", type=float, default=0.0005)

args = parser.parse_args()

# Assign arguments to variables
attack_prompts_file = args.attack_prompts_file
audioroot = args.audioroot
category = args.category

audiosaveroot = args.audiosaveroot


MODEL_CACHE = ".cache"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/ictnlp/LLaMA-Omni/{MODEL_CACHE}.tar"
)
tokenizer, base_model, _ = load_pretrained_model(
    f"{MODEL_CACHE}/Llama-3.1-8B-Omni", model_base=None, s2s=False
)

dataset = DataSetSpeechAttack(
    attack_prompts_file,
    audioroot,
    target_transform=tokenizer
)

counter = 0
for i, sample in tqdm(enumerate(iter(dataset))):
    if i == 0 or i >= 9:
        continue

    custom_model = CustomAudioModelWithNoise(
        base_model, 
        tokenizer, 
        sample['label'],
        sample['prompt']
    ).to("cuda")
    attack_iters = 1000
    attack = PGDSpeech(
        custom_model, 
        eps=args.eps,  # values in noise max and min
        alpha=args.alpha,  # learning power 001
        steps=attack_iters, 
        nprompt=1,
        saveroot=audiosaveroot,
        category=category
    )
    
    adv_noise, success, message = attack(sample['inputs'], None, index=i)
    
    if success is True:
        counter += 1
    
print('accuracy', counter / len(dataset))