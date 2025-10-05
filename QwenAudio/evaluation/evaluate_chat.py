import argparse
import json
import os
import random
import time
from functools import partial
import torch
import requests
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read


ds_collections = {
    'airbench_level3': {'path': 'airbench_level_3_eval.jsonl'}
}


def add_noise(audio, eps, distribution='uniform'):
    """
    Add noise to audio with specified epsilon and distribution
    Works on CPU tensors to avoid CUDA issues with multiprocessing

    Args:
        audio: Audio numpy array
        eps: Noise magnitude (epsilon)
        distribution: 'uniform' or 'normal'

    Returns:
        Noisy audio numpy array, clamped between min=0 and max=1
    """
    if eps == 0:
        return audio
    
    # Create noise directly using numpy to avoid CUDA
    if distribution == 'uniform':
        noise = np.random.uniform(-eps, eps, audio.shape)
    else:  # normal
        noise = np.random.normal(0, eps, audio.shape)
    
    # Apply noise and clamp
    noisy_audio = np.clip(audio + noise, 0, 1)
    return noisy_audio


class AudioChatDataset(torch.utils.data.Dataset):

    def __init__(self, ds):
        path = ds['path']
        self.datas = open(path).readlines()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio = data['audio']
        data_idx = data['id']
        query = data['query']

        return {
            'audio': audio,
            'data_idx': data_idx,
            'query': query,
        }


def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        # We need to actually check for a real protocol, otherwise it's impossible to use a local file
        # like http_huggingface_co.png
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs


def collate_fn(inputs, processor, noise_eps, noise_distribution):
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

    audio_path = [_['audio'] for _ in inputs]
    data_idxs = [_['data_idx'] for _ in inputs]

    # Read audio files
    input_audios = [
        ffmpeg_read(
            read_audio(_['audio']), 
            sampling_rate=processor.feature_extractor.sampling_rate
        ) for _ in inputs
    ]

    # Apply noise directly to numpy arrays (CPU operation)
    noisy_audios = [
        add_noise(audio, noise_eps, noise_distribution) 
        for audio in input_audios
    ]

    # Process inputs with noisy audio
    processor_inputs = processor(
        text=text_list, 
        audios=noisy_audios, 
        sampling_rate=processor.feature_extractor.sampling_rate, 
        return_tensors="pt", 
        padding=True
    )

    # We'll move to device later in the main loop, not here in the worker process
    return processor_inputs, audio_path, data_idxs


if __name__ == '__main__':
    # No need to set multiprocessing start method anymore as we've moved 
    # CUDA operations out of worker processes

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='airbench_level3')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise-eps', type=float, default=0.0, 
                        help='Epsilon value for noise magnitude (0.0 for no noise)')
    parser.add_argument('--noise-distribution', type=str, default='uniform', 
                        choices=['uniform', 'normal'], 
                        help='Noise distribution type: uniform or normal')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on (cuda or cpu)')
    args = parser.parse_args()

    device = args.device
    
    # Set the device for cuda
    if device == 'cuda':
        torch.cuda.set_device(0)  # Use the first GPU by default

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint, device_map=device, torch_dtype='auto', trust_remote_code=True).eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.tokenizer.padding_side = 'left'

    random.seed(args.seed)
    
    dataset = AudioChatDataset(
        ds=ds_collections[args.dataset],
    )
    
    # Pass noise parameters to collate_fn
    collate_fn_with_noise = partial(
        collate_fn, 
        processor=processor,
        noise_eps=args.noise_eps,
        noise_distribution=args.noise_distribution
    )
    
    # Create regular DataLoader with specified num_workers
    # Now safe to use with multiple workers
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_with_noise,
        shuffle=False,  # Keep in order for reproducibility
    )

    idxs = []
    rets = []
    audio_paths = []
    
    print(f"Evaluating {args.dataset} with noise epsilon={args.noise_eps}, distribution={args.noise_distribution}")
    
    # Run inference
    for _, (inputs, audio_path, data_idxs) in tqdm(enumerate(data_loader), total = len(data_loader)):
        # Move inputs to device here, after they've been collected from workers
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        output_ids = model.generate(**inputs, max_new_tokens=256, min_new_tokens=1, do_sample=False)
        output_ids = output_ids[:, inputs['input_ids'].size(1):]
        output = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rets.extend(output)
        audio_paths.extend(audio_path)
        idxs.extend(data_idxs)

    # Prepare results
    results = []
    for idx, response, audio_path in zip(idxs, rets, audio_paths):
        results.append({
            'idx': idx,
            'response': response,
            'audio_path': audio_path,
        })
    
    # Include noise parameters in filename
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = f'{args.dataset}_eps{args.noise_eps}_{args.noise_distribution}_{time_prefix}.json'
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f)
    
    print(f"Results saved to {results_file}")