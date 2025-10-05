import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import torch
import requests
import librosa
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read
import noisereduce as nr

from noise_defender import run_with_intervention
from noise_analyzer import detect_noise_sensitive_neurons, get_activations

ds_collections = {
    'airbench_level3': {'path': 'airbench_level_3_eval.jsonl'}
}


class AudioChatDataset(torch.utils.data.Dataset):
    def __init__(self, ds, sample_ratio=1.0, seed=42):
        path = ds['path']
        all_data = open(path).readlines()
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)
            
        # Sample the data if ratio is less than 1
        if sample_ratio < 1.0:
            num_samples = max(1, int(len(all_data) * sample_ratio))
            self.datas = random.sample(all_data, num_samples)
        else:
            self.datas = all_data

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio = data['audio']
        data_idx = data['id']
        query = data['query']
        clean_audio = data.get('clean_audio', audio)  # Assuming clean audio path is provided

        return {
            'audio': audio,
            'clean_audio': clean_audio,
            'data_idx': data_idx,
            'query': query,
        }

def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs

def collate_fn(inputs, processor):
    text_list = []
    for _ in inputs:
        query = _['query']
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': query}
        ]
        text = processor.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors='pt',
            tokenize=False
        )
        text_list.append(text)

    audio_paths = [_['audio'] for _ in inputs]
    data_idxs = [_['data_idx'] for _ in inputs]
    
    # Assuming you have noisereduce imported as nr
    input_audios = [ffmpeg_read(read_audio(_['audio']), sampling_rate=processor.feature_extractor.sampling_rate) for _ in inputs]
    denoised_audios = [nr.reduce_noise(y=audio, sr=processor.feature_extractor.sampling_rate) for audio in input_audios]

    inputs = processor(text=text_list, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    inputs_denoised = processor(text=text_list, audios=denoised_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    
    return inputs, inputs_denoised, audio_paths, data_idxs

def run_defense(model, noisy_inputs, clean_inputs, processor, defense_type='prune', top_k=0.5, activation_choice='lm'):
    """Apply defense mechanism and generate output"""
    noise_neurons, clean_activations, noisy_activations = detect_noise_sensitive_neurons(
        model, 
        clean_inputs,
        noisy_inputs,
        top_k_percent=top_k,
        activations=activation_choice
    )
    
    # if defense_type == 'patch':
    #     clean_activations = get_activations(model, clean_inputs, activation_choice).to(noisy_inputs['input_ids'].device)
    # else:
    #     clean_activations = None
    breakpoint()
    output = run_with_intervention(
        model=model,
        input_data=noisy_inputs,
        neurons=noise_neurons,
        intervention_type=defense_type,
        clean_activations=clean_activations,
        activations_choice=activation_choice,
        mode='all',
        bias=0.5
    )
    
    return output, len(noise_neurons)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--defense-type', type=str, default='prune', choices=['prune', 'bias', 'patch'])
    parser.add_argument('--top-k', type=float, default=0.5)
    parser.add_argument('--activation-choice', type=str, default='lm', choices=['lm', 'audio'])
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample_ratio', type=float, default=1)
    args = parser.parse_args()

    # Set CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint, device_map='cuda', torch_dtype='auto', trust_remote_code=True)

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.tokenizer.padding_side = 'left'

    random.seed(args.seed)
    dataset = AudioChatDataset(ds=ds_collections[args.dataset], sample_ratio=args.sample_ratio)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
        shuffle=False  # Keep order deterministic
    )

    results = []
    count = 0

    for inputs_noisy, inputs_clean, audio_path, data_idxs in tqdm(data_loader, total=len(data_loader)):
    # Process each item in batch individually
        for i in range(len(data_idxs)):
            # Extract single sample from batch
            single_noisy = {k: v[i:i+1].to(device) for k, v in inputs_noisy.items()}
            single_clean = {k: v[i:i+1].to(device) for k, v in inputs_clean.items()}
            
            # Apply defense and generate output
            output, num_neurons = run_defense(
                model, 
                single_noisy, 
                single_clean, 
                processor,
                defense_type=args.defense_type,
                top_k=args.top_k,
                activation_choice=args.activation_choice
            )
            
            # Process output
            output_ids = output.sequences[:, single_noisy['input_ids'].size(1):]
            output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]  # [0] since we're processing single items
            
            # Store results
            results.append({
                'idx': data_idxs[i],
                'response': output_text,
                'audio_path': audio_path[i],
                'num_neurons': num_neurons,
                'defense_type': args.defense_type,
                'top_k': args.top_k,
                'activation_choice': args.activation_choice
            })

    # Save results
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = f'{args.dataset}_{args.defense_type}_{args.top_k}_{args.activation_choice}_{time_prefix}.json'
    json.dump(results, open(results_file, 'w'))