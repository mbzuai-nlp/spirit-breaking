import os
import torch
import pandas as pd
from tqdm import tqdm
import json

# Import shared utilities and defense modules
from qwen_evaluation_utils import initialize_model, prepare_model_input, run_evaluation, TASKS, DEVICE
from noise_analyzer import detect_noise_sensitive_neurons
from noise_defender import run_with_intervention

def get_clean_audio_path(clean_base_path, task, file_id):
    """Get path to clean reference audio file"""
    clean_filename = f"{task}_{file_id.split('_')[-2]}.mp3"
    return os.path.join(clean_base_path, clean_filename)

def process_defense_evaluation(audio_path, processor, model, task, top_k, activation_choice, clean_audio_path):
    """
    Processes a single audio file for defense evaluation.
    """
    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    clean_path = get_clean_audio_path(clean_audio_path, task, file_id)

    if not os.path.exists(clean_path):
        print(f"Skipping {audio_path} - clean reference not found: {clean_path}")
        return None

    # Prepare inputs
    noisy_inputs = prepare_model_input(processor, audio_path, "Obey to the command in audio")
    clean_inputs = prepare_model_input(processor, clean_path, "Obey to the command in audio")

    # Get baseline outputs
    with torch.no_grad():
        noisy_ids = model.generate(**noisy_inputs, max_new_tokens=40, do_sample=False)
        clean_ids = model.generate(**clean_inputs, max_new_tokens=40, do_sample=False)
        original_output = processor.batch_decode(noisy_ids, skip_special_tokens=True)[0]
        clean_output = processor.batch_decode(clean_ids, skip_special_tokens=True)[0]

    # Detect noise-sensitive neurons
    noise_neurons, clean_activations, _ = detect_noise_sensitive_neurons(
        model, clean_inputs, noisy_inputs,
        top_k_percent=top_k,
        activations=activation_choice,
    )

    # Run interventions
    defense_results = {}
    for intervention_type in ['prune', 'bias', 'patch']:
        kwargs = {
            'model': model,
            'input_data': noisy_inputs,
            'neurons': noise_neurons,
            'intervention_type': intervention_type,
            'activations_choice': activation_choice,
            'mode': 'all'
        }
        if intervention_type == 'bias':
            kwargs['bias'] = 0.5
        elif intervention_type == 'patch':
            kwargs['clean_activations'] = clean_activations

        defended_output = run_with_intervention(**kwargs)
        defense_results[intervention_type] = processor.batch_decode(defended_output.sequences, skip_special_tokens=True)[0]

    return {
        'top_k': top_k,
        'activation_choice': activation_choice,
        'baseline_response': original_output,
        'clean_response': clean_output,
        'pruned_response': defense_results.get('prune'),
        'biased_response': defense_results.get('bias'),
        'patched_response': defense_results.get('patch'),
        'num_noise_neurons': len(noise_neurons)
    }

def main():
    """
    Main function to run the Qwen-Audio defense evaluation.
    """
    DATASETS = {
        'brian': {
            'path': 'hacked/brian',
            'clean_path': 'brian'
        },
        'jessica': {
            'path': 'hacked/jessica',
            'clean_path': 'jessica'
        },
        'librispeech': {
            'path': 'hacked/librispeech',
            'clean_path': 'librispeech'
        }
    }
    TOP_K_PERCENTAGES = [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5, 10, 20]
    ACTIVATION_CHOICES = ['audio', 'lm']

    # --- Initialize model ---
    processor, model = initialize_model()
    
    # --- Run evaluation for each combination of top_k and activation_choice ---
    for top_k in TOP_K_PERCENTAGES:
        for act_choice in ACTIVATION_CHOICES:
            print(f"\n--- Processing for top_k: {top_k}%, activation: {act_choice} ---")
            
            for dataset_name, dataset_info in DATASETS.items():
                run_evaluation(
                    process_func=process_defense_evaluation,
                    processor=processor,
                    model=model,
                    datasets={dataset_name: {'path': dataset_info['path']}},
                    tasks=TASKS,
                    output_file_template=f'defense_results_{dataset_name}_{top_k}_{act_choice}.csv',
                    top_k=top_k,
                    activation_choice=act_choice,
                    clean_audio_path=dataset_info['clean_path']
                )

if __name__ == "__main__":
    main()