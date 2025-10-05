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


ds_collections = {
    'meld': {'path': 'meld_eval.jsonl'}
}


def add_noise(audio, eps, distribution='uniform'):
    """
    Add noise to audio with specified epsilon and distribution
    """
    # Convert to tensor if needed
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio)
    
    if distribution == 'uniform':
        noise = torch.empty_like(audio).uniform_(-eps, eps)
    else:  # normal
        noise = torch.empty_like(audio).normal_(0, eps)
    
    if eps == 0:
        return audio.numpy() if not isinstance(audio, torch.Tensor) else audio
    else:
        # Note: Changed clipping range to match audio range (typically -1 to 1 for audio)
        noisy_audio = torch.clamp(audio + noise, min=-1, max=1)
        return noisy_audio.numpy() if audio.requires_grad == False else noisy_audio


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
        # We need to actually check for a real protocol, otherwise it's impossible to use a local file
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs


def collate_fn(inputs, processor, noise_eps=0.0, noise_distribution='uniform'):
    input_texts = [_['prompt'] for _ in inputs]
    source = [_['source'] for _ in inputs]
    gt = [_['gt'] for _ in inputs]
    audio_path = [_['audio'] for _ in inputs]
    
    # Read and process audio
    input_audios = []
    for audio_file in audio_path:
        # Read audio
        audio_data = ffmpeg_read(read_audio(audio_file), sampling_rate=processor.feature_extractor.sampling_rate)
        
        # Add noise to corrupt the audio
        if noise_eps > 0:
            audio_data = add_noise(audio_data, noise_eps, noise_distribution)
        
        input_audios.append(audio_data)
    
    # Process with the model's processor
    inputs = processor(text=input_texts, 
                      audios=input_audios, 
                      sampling_rate=processor.feature_extractor.sampling_rate, 
                      return_tensors="pt", 
                      padding=True)
    
    return inputs, audio_path, source, gt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B')
    parser.add_argument('--dataset', type=str, default='meld')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise-eps', type=float, default=0.1, help='Epsilon for noise addition')
    parser.add_argument('--noise-distribution', type=str, default='uniform', 
                       choices=['uniform', 'normal'], help='Noise distribution type')
    args = parser.parse_args()

    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint, device_map=device, torch_dtype='auto', trust_remote_code=True).eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.tokenizer.padding_side = 'left'

    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Initialize dataset and dataloader
    dataset = AudioDataset(ds=ds_collections[args.dataset])
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor, 
                          noise_eps=args.noise_eps, 
                          noise_distribution=args.noise_distribution),
        shuffle=False  # Keep order for evaluation
    )

    # Lists to store results
    gts = []
    sources = []
    responses = []
    audio_paths = []

    # Process each batch
    print(f"Processing {args.dataset} with noise_eps={args.noise_eps} ({args.noise_distribution} distribution)...")
    for _, (inputs, audio_path, source, gt) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Move input tensors to the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate predictions
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=256, 
            min_new_tokens=1, 
            do_sample=False
        )
        
        # Extract the generated text, skipping the input tokens
        output_ids = output_ids[:, inputs['input_ids'].size(1):]
        output = processor.batch_decode(
            output_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        # Extend the result lists
        gts.extend(gt)
        responses.extend(output)
        sources.extend(source)
        audio_paths.extend(audio_path)

    print(f"Evaluating {args.dataset} ...")

    # Process results
    results = []
    for gt, response, source, audio_path in zip(gts, responses, sources, audio_paths):
        results.append({
            'gt': gt,
            'response': response,
            'source': source,
            'audio_path': audio_path,
        })
    
    # Save results to file
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = f'{args.dataset}_noisy_eps{args.noise_eps}_{time_prefix}.json'
    json.dump(results, open(results_file, 'w'), indent=2)
    print(f"Results saved to: {results_file}")
    
    # Group results by source and calculate accuracy
    results_dict = {}
    for item in tqdm(results):
        source = item["source"]
        results_dict.setdefault(source, []).append(item)
    
    # Calculate accuracy for each source
    print(f"\nAccuracy Results (with noise_eps={args.noise_eps}):")
    overall_refs, overall_hyps = [], []
    
    for source in results_dict:
        refs, hyps = [], []
        results_list = results_dict[source]
        
        for result in results_list:
            gt = result["gt"]
            response = result["response"].lstrip()
            refs.append(gt)
            hyps.append(response)
            overall_refs.append(gt)
            overall_hyps.append(response)
        
        # Calculate accuracy for this source
        score = accuracy_score(refs, hyps)
        print(f"{source} ACC_score: {score:.4f} (n={len(hyps)})")
    
    # Calculate overall accuracy
    if overall_refs:
        overall_score = accuracy_score(overall_refs, overall_hyps)
        print(f"\nOverall ACC_score: {overall_score:.4f} (n={len(overall_refs)})")