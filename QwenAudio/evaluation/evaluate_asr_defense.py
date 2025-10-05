import argparse
import json
import os
import random
import time
from functools import partial
import re
from evaluate_tokenizer import EvaluationTokenizer
import editdistance as ed
import torch
from transformers.pipelines.audio_utils import ffmpeg_read
import requests
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
from cn_tn import TextNorm
import noisereduce as nr

from noise_defender import run_with_intervention
from noise_analyzer import detect_noise_sensitive_neurons, get_activations
# from df import enhance, init_df

from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

PUNCS = '!,.?;:'

ds_collections = {
    'librispeech': {'path': 'librispeech_eval.jsonl','language': 'en'}
}

# Initialize normalizers
english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode='',
)
basic_normalizer = BasicTextNormalizer()

# Initialize denoiser
# denoiser, df_state, _ = init_df()  # Load default model

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, ds, sample_ratio=1.0, seed=None):
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
    """Fixed collate function that properly handles text and audio processing"""
    input_texts = [_['prompt'] for _ in inputs]
    source = [_['source'] for _ in inputs]
    gt = [_['gt'] for _ in inputs]
    audio_path = [_['audio'] for _ in inputs]
    
    # Read and process original audio
    input_audios = [ffmpeg_read(read_audio(_['audio']), 
                   sampling_rate=processor.feature_extractor.sampling_rate) 
                   for _ in inputs]
    
    # Create denoised version for clean inputs
    denoised_audios = [nr.reduce_noise(y=audio, sr=processor.feature_extractor.sampling_rate) for audio in input_audios]

    # Process both original and denoised inputs
    inputs_noisy = processor(text=input_texts, 
                           audios=input_audios,
                           sampling_rate=processor.feature_extractor.sampling_rate, 
                           return_tensors="pt", 
                           padding=True)
    
    inputs_clean = processor(text=input_texts, 
                           audios=denoised_audios,
                           sampling_rate=processor.feature_extractor.sampling_rate, 
                           return_tensors="pt", 
                           padding=True)
    
    return inputs_noisy, inputs_clean, audio_path, source, gt

def run_defense(model, noisy_inputs, clean_inputs, processor, defense_type='prune', top_k=0.5, activation_choice='lm'):
    """Apply defense mechanism and generate output"""
    noise_neurons, clean_activations, noisy_activations = detect_noise_sensitive_neurons(
        model, 
        clean_inputs,
        noisy_inputs,
        top_k_percent=top_k,
        activations=activation_choice
    )
        
    output = run_with_intervention(
        model=model,
        input_data=noisy_inputs,
        neurons=noise_neurons,
        intervention_type=defense_type,
        clean_activations=clean_activations,
        activations_choice=activation_choice,
        mode='all',
        bias=0.5 if defense_type == 'bias' else 1.0
    )
    
    return output, len(noise_neurons)

def remove_sp(text, language):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    if language == "zh":
        gt = re.sub(rf"\s+", r"", gt)
    return gt

def compute_wer(refs, hyps, language):
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )
    
    for i in range(len(refs)):
        ref = refs[i]
        pred = hyps[i]
        
        if language in ["en"]:
            ref = english_normalizer(ref)
            pred = english_normalizer(pred)
        elif language in ["zh"]:
            ref = chinese_normalizer(ref)
            pred = chinese_normalizer(pred)
        else:
            ref = basic_normalizer(ref)
            pred = basic_normalizer(pred)
            
        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()
        
        if language in ["zh", "yue"]:
            ref_items = [x for x in "".join(ref_items)]
            pred_items = [x for x in "".join(pred_items)]
            
        if i == 0:
            print(f"ref: {ref}")
            print(f"pred: {pred}")
            print(f"ref_items:\n{ref_items}\n{len(ref_items)}")
            print(f"pred_items:\n{pred_items}\n{len(pred_items)}")
            
        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
        
    return distance/ref_length

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--defense-type', type=str, default='prune', choices=['prune', 'bias', 'patch'])
    parser.add_argument('--top-k', type=float, default=0.5)
    parser.add_argument('--activation-choice', type=str, default='lm', choices=['lm', 'audio'])
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(111)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint, device_map='cuda', torch_dtype='auto', trust_remote_code=True)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.tokenizer.padding_side = 'left'

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Initialize dataset and dataloader
    dataset = AudioDataset(ds=ds_collections[args.dataset], sample_ratio=1)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
        shuffle=False
    )

    print(f"Starting evaluation with {args.defense_type} defense...")
    print(f"Top-k: {args.top_k}, Activation choice: {args.activation_choice}")
    
    count = 0
    results = []
    try:
        for inputs_noisy, inputs_clean, audio_paths, sources, gts in tqdm(data_loader):
            count += 1
            # Process each item in batch individually
            for i in range(len(sources)):
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
                output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                # Store results
                results.append({
                    'gt': gts[i],
                    'response': output_text,
                    'source': sources[i],
                    'audio_path': audio_paths[i],
                    'num_neurons': num_neurons,
                    'defense_type': args.defense_type,
                    'top_k': args.top_k,
                    'activation_choice': args.activation_choice
                })
                
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        # Save partial results if available
        if results:
            print("Saving partial results...")
    finally:
        # Save results and compute metrics
        if results:
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{args.dataset}_{args.defense_type}_{args.top_k}_{args.activation_choice}_sampled.json'
            json.dump(results, open(results_file, 'w'))

            # Compute WER for each source
            results_dict = {}
            for item in results:
                source = item["source"]
                results_dict.setdefault(source, []).append(item)

            lan = ds_collections[args.dataset]['language']
            print("\nWER Results:")
            for source in results_dict:
                refs, hyps = [], []
                results_list = results_dict[source]
                for result in results_list:
                    gt = result["gt"]
                    response = result["response"]
                    gt = remove_sp(gt, lan)
                    response = remove_sp(response, lan)
                    refs.append(gt)
                    hyps.append(response)
                wer = compute_wer(refs, hyps, lan)
                print(f"source: {source}  cnt: {len(refs)} wer: {wer:.4f}")