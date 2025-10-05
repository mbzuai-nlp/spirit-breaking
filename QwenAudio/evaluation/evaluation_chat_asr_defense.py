
import os
import torch
import json
import time
import random
import argparse
import requests
import itertools
from functools import partial

from tqdm import tqdm

from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from transformers.pipelines.audio_utils import ffmpeg_read


from noise_analyzer import detect_random_neurons, detect_noise_sensitive_neurons
from noise_defender import run_with_intervention
import noisereduce as nr
import soundfile as sf

def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        # We need to actually check for a real protocol, otherwise it's impossible to use a local file
        # like http_huggingface_co.png
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs

def collate_fn(features, processor):
    text_list = []
    for _ in features:
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

    audio_path = [_['audio'] for _ in features]
    data_idxs = [_['data_idx'] for _ in features
                 ]
    input_audios = [ffmpeg_read(read_audio(_['audio']), sampling_rate=processor.feature_extractor.sampling_rate) for _ in features]
    inputs = processor(text=text_list, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    noisy_audio = [_['input_audios'] for _ in features]
    noisy_audio_inputs = processor(text=text_list, audios=noisy_audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    clean_audio = [_['denoised_audios'] for _ in features]
    clean_audio_inputs = processor(text=text_list, audios=clean_audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    return inputs, audio_path, data_idxs, noisy_audio_inputs, clean_audio_inputs

class AudioTranscriptionDataset(torch.utils.data.Dataset):

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

        audioarray, sr = sf.read(audio)
        assert sr == processor.feature_extractor.sampling_rate

        input_audios = ffmpeg_read(read_audio(audio), sampling_rate=processor.feature_extractor.sampling_rate)
        denoised_audios = nr.reduce_noise(y=audioarray, sr=processor.feature_extractor.sampling_rate)
        
        query = "<|audio_bos|><|AUDIO|><|audio_eos|>\nListen to the speech and write down its content."

        return {
            'audio': audio,
            'data_idx': data_idx,
            'query': query,

            'input_audios': input_audios,
            'denoised_audios': denoised_audios
        }

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = 0
        self._world_size = 1
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


ds_collections = {
    'airbench_level3': {'path': 'airbench_level_3_eval.jsonl'}
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Initialize Whisper model
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint, device_map='cuda').eval()
    model.eval()
    # Initialize Whisper processor
    processor = AutoProcessor.from_pretrained(args.checkpoint)

    random.seed(args.seed)
    dataset = AudioTranscriptionDataset(
        ds=ds_collections[args.dataset],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    idxs = []
    rets = []
    audio_paths = []
    for _, (inputs, audio_path, data_idxs, noisy_inputs, clean_inputs) in tqdm(enumerate(data_loader)):
        inputs = inputs.to('cuda')
        noisy_inputs = noisy_inputs.to('cuda')
        clean_inputs = clean_inputs.to('cuda')
        
        with torch.no_grad():
            # output_ids = model.generate(
            #     **inputs,
            #     max_new_tokens=256,
            #     return_dict_in_generate=True,
            #     output_hidden_states=True,
            #     output_scores=True,
            #     use_cache=True
            # )
            neurons, clean_activations, noisy_activations = detect_noise_sensitive_neurons(
                model, 
                noisy_inputs,
                clean_inputs,
                top_k_percent=0.1,
                activations='audio',
            )
            
            breakpoint()
            output = run_with_intervention(
                model=model,
                input_data=noisy_inputs,
                neurons=neurons,
                intervention_type='bias',
                clean_activations=clean_activations,
                activations_choice='audio',
                mode='all',
                bias=0.5
            )

            output_ids = output_ids[:, inputs.input_ids.size(1):]
        
        # Decode output tokens
        output = processor.batch_decode(output_ids, skip_special_tokens=True)
        
        print(output)
        breakpoint()
        
        rets.extend(output)
        audio_paths.extend(audio_path)
        idxs.extend(data_idxs)
        
    world_size = torch.distributed.get_world_size()
    merged_idxs = [None for _ in range(world_size)]
    merged_responses = [None for _ in range(world_size)]
    merged_audio_paths = [None for _ in range(world_size)]
    
    merged_idxs = [_ for _ in itertools.chain.from_iterable(merged_idxs)]
    merged_audio_paths = [_ for _ in itertools.chain.from_iterable(merged_audio_paths)]
    merged_responses = [
        _ for _ in itertools.chain.from_iterable(merged_responses)
    ]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")

        results = []
        for idx, response, audio_path in zip(merged_idxs, merged_responses, merged_audio_paths):
            results.append({
                'idx': idx,
                'response': response,
                'audio_path': audio_path,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))
