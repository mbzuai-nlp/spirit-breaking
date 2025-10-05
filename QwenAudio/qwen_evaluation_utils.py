import os
import torch
import torchaudio
import librosa
import pandas as pd
from tqdm import tqdm
import tempfile
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# --- Constants and Configuration ---

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TASKS = [
    '1_bomb_explosive', '2_drugs', '3_suicide', '4_hack_information', 
    '5_kill_someone', '6_social_violence', '7_finance_stock', '8_firearms_weapons'
]

# --- Core Functions ---

def initialize_model(cacheroot=''):
    """
    Initializes and returns the Qwen2-Audio model and processor.
    """
    print("Initializing Qwen2-Audio model and processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir=cacheroot)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", 
        device_map=DEVICE
    )
    model.eval()
    print("Model loaded successfully.")
    return processor, model

def prepare_model_input(processor, audio_path, text_prompt, audio_tensor=None):
    """
    Prepares the model input dictionary.
    """
    if audio_tensor is None:
        audio_tensor, sr = torchaudio.load(audio_path)
        if sr != processor.feature_extractor.sampling_rate:
            resampled_audio_np = librosa.resample(
                audio_tensor.numpy(), 
                orig_sr=sr, 
                target_sr=processor.feature_extractor.sampling_rate
            )
            audio_tensor = torch.from_numpy(resampled_audio_np)

    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant that tries to help as much as possible.'}, 
        {
            "role": "user", 
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": text_prompt},
            ]
        },
    ]
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    inputs = processor(
        text=text, 
        audios=audio_tensor.squeeze().numpy(), 
        return_tensors="pt", 
        padding=True
    )
    
    return {k: v.to(DEVICE) for k, v in inputs.items()}

def run_evaluation(process_func, processor, model, datasets, tasks, output_file_template, **kwargs):
    """
    Generic evaluation loop for Qwen model.
    """
    for dataset_name, dataset_info in datasets.items():
        print(f"\n\n--- Processing dataset: {dataset_name} ---")
        
        output_file = output_file_template.format(dataset=dataset_name)
        dataset_path = dataset_info['path']
        
        all_results = []

        for task in tasks:
            task_path = os.path.join(dataset_path, task)
            if not os.path.isdir(task_path):
                print(f"Warning: Task directory not found, skipping: {task_path}")
                continue
            
            print(f"\nProcessing task: {task}")
            audio_files = [f for f in os.listdir(task_path) if f.endswith('.wav')]

            for audio_file in tqdm(audio_files, desc=f"Task {task}"):
                audio_path = os.path.join(task_path, audio_file)
                
                try:
                    results = process_func(
                        audio_path=audio_path,
                        processor=processor,
                        model=model,
                        task=task,
                        **kwargs
                    )
                    
                    result_entry = {
                        'task': task,
                        'audio_file': audio_file,
                        **results
                    }
                    all_results.append(result_entry)
                
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if all_results:
            print(f"\nSaving results for {dataset_name}...")
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to '{output_file}'")
        else:
            print(f"No results to save for dataset {dataset_name}.")