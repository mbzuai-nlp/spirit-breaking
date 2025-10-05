import argparse
import json
import os
import random
import time
from functools import partial
import torch
import requests

from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read
from sklearn.metrics import accuracy_score
import noisereduce as nr
import sys

# Assuming the custom modules are in this path
sys.path.append('activation_patching')
from noise_defender import run_with_intervention
from noise_analyzer import detect_noise_sensitive_neurons

ds_collections = {
    'meld': {'path': 'meld_eval.jsonl'}
}

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        path = ds['path']
        self.datas = open(path).readlines()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio = data['audio']
        source = data['source']
        prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>" + data['prompt']
        gt = data['gt']

        return {
            'audio': audio,
            'prompt': prompt,
            'source': source,
            'gt': gt
        }

def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs

def collate_fn(inputs, processor):
    input_texts = [_['prompt'] for _ in inputs]
    source = [_['source'] for _ in inputs]
    gt = [_['gt'] for _ in inputs]
    audio_paths = [_['audio'] for _ in inputs]
    
    original_audios = []
    denoised_audios = []
    for audio_file in audio_paths:
        # The original audio is treated as the "noisy" input
        original_audio_data = ffmpeg_read(read_audio(audio_file), sampling_rate=processor.feature_extractor.sampling_rate)
        # The "clean" reference is created by denoising the original audio
        denoised_audio_data = nr.reduce_noise(y=original_audio_data, sr=processor.feature_extractor.sampling_rate)
        
        original_audios.append(original_audio_data)
        denoised_audios.append(denoised_audio_data)

    # `inputs_noisy` are from the original files
    inputs_noisy = processor(text=input_texts, audios=original_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    # `inputs_clean` are from the denoised versions
    inputs_clean = processor(text=input_texts, audios=denoised_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    
    return inputs_clean, inputs_noisy, audio_paths, source, gt

def run_defense(model, clean_inputs, noisy_inputs, defense_type='prune', top_k=0.5, activation_choice='lm'):
    """Apply defense mechanism and generate output"""
    noise_neurons, clean_activations, _ = detect_noise_sensitive_neurons(
        model, 
        clean_inputs,
        noisy_inputs,
        top_k_percent=top_k,
        activations=activation_choice
    )
    
    output = run_with_intervention(
        model=model,
        input_data=noisy_inputs, # Run intervention on the original (noisy) input
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
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B')
    parser.add_argument('--dataset', type=str, default='meld')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--defense-type', type=str, default='prune', choices=['prune', 'bias', 'patch'])
    parser.add_argument('--top-k', type=float, default=0.1, help="Percentage of neurons to select (e.g., 0.1 for 10%)")
    parser.add_argument('--activation-choice', type=str, default='lm', choices=['lm', 'audio'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint, device_map=device, torch_dtype='auto', trust_remote_code=True).eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.tokenizer.padding_side = 'left'

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    dataset = AudioDataset(ds=ds_collections[args.dataset])
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
        shuffle=False
    )

    gts, responses, sources, audio_paths, neurons_used = [], [], [], [], []
    
    print(f"Processing {args.dataset} with defense: {args.defense_type}, top_k={args.top_k}")
    for inputs_clean, inputs_noisy, audio_path, source, gt in tqdm(data_loader, total=len(data_loader)):
        for i in range(len(gt)):
            single_clean = {k: v[i:i+1].to(device) for k, v in inputs_clean.items()}
            single_noisy = {k: v[i:i+1].to(device) for k, v in inputs_noisy.items()}
            
            output, num_neurons = run_defense(
                model, single_clean, single_noisy,
                defense_type=args.defense_type,
                top_k=args.top_k,
                activation_choice=args.activation_choice
            )
            
            output_ids = output.sequences[:, single_noisy['input_ids'].size(1):]
            output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            gts.append(gt[i])
            responses.append(output_text)
            sources.append(source[i])
            audio_paths.append(audio_path[i])
            neurons_used.append(num_neurons)

    print(f"Evaluating {args.dataset} ...")

    results = []
    for gt, response, source, audio_path, num_neurons in zip(gts, responses, sources, audio_paths, neurons_used):
        results.append({
            'gt': gt, 'response': response, 'source': source, 'audio_path': audio_path,
            'num_neurons': num_neurons, 'defense_type': args.defense_type,
            'top_k': args.top_k, 'activation_choice': args.activation_choice
        })
    
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = f'{args.dataset}_defense_{args.defense_type}_k{args.top_k}_{time_prefix}.json'
    json.dump(results, open(results_file, 'w'), indent=2)
    print(f"Results saved to: {results_file}")
    
    results_dict = {}
    for item in tqdm(results):
        results_dict.setdefault(item["source"], []).append(item)
    
    print(f"\nAccuracy Results (defense: {args.defense_type}, top_k={args.top_k}):")
    overall_refs, overall_hyps = [], []
    
    for source in results_dict:
        refs, hyps = [], []
        results_list = results_dict[source]
        
        for result in results_list:
            refs.append(result["gt"])
            hyps.append(result["response"].lstrip())
        
        overall_refs.extend(refs)
        overall_hyps.extend(hyps)
        
        score = accuracy_score(refs, hyps)
        avg_neurons = sum(r['num_neurons'] for r in results_list) / len(results_list)
        print(f"{source} ACC_score: {score:.4f} (n={len(hyps)}), avg_neurons: {avg_neurons:.2f}")
    
    if overall_refs:
        overall_score = accuracy_score(overall_refs, overall_hyps)
        overall_avg_neurons = sum(neurons_used) / len(neurons_used)
        print(f"\nOverall ACC_score: {overall_score:.4f} (n={len(overall_refs)}), avg_neurons: {overall_avg_neurons:.2f}")