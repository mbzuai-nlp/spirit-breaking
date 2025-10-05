import os
import sys
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add necessary paths to sys.path
LLAMA_OMNI_PATH = "LLaMA-Omni"
if LLAMA_OMNI_PATH not in sys.path:
    sys.path.append(LLAMA_OMNI_PATH)

ACTIVATION_PATCHING_PATH = "activation_patching"
if ACTIVATION_PATCHING_PATH not in sys.path:
    sys.path.append(ACTIVATION_PATCHING_PATH)

# LLaMA-Omni specific imports
from omni_speech.model.builder import load_pretrained_model
from omni_speech.utils import disable_torch_init
from omni_speech.infer.infer import create_data_loader

# Imports from our new defense modules
from noise_analyzer_llama import detect_noise_sensitive_neurons
from noise_defender_llama import run_with_intervention

# Imports from evaluation utilities
from llamaomni_eval_utils import (
    initialize_model,
    prepare_model_input,
    run_evaluation,
    DATASETS,
    TASKS,
    DEVICE
)


def main():
    """
    A simple test function to demonstrate the LLaMA-Omni defense mechanism.
    """
    # --- Configuration ---
    CLEAN_AUDIO_PATH = "/workspace/nurdaulet/speechdatasets/advbench_audio_brian/1_bomb_explosive_2.mp3"
    TOP_K_PERCENTAGES = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    BIAS_STRENGTH = 1.0
    ACTIVATION_CHOICE = 'audio' # 'lm' or 'audio'

    # --- Initialize model and tokenizer ---
    tokenizer, model = initialize_model()

    # --- Prepare inputs for clean audio (used for comparison) ---
    print(f"Loading clean audio from: {CLEAN_AUDIO_PATH}")
    clean_inputs = prepare_model_input(CLEAN_AUDIO_PATH, tokenizer, model.config)

    for top_k_percent in TOP_K_PERCENTAGES:
        print(f"\n--- Processing for top_k_percent: {top_k_percent}% ---")
        
        output_file_template = 'llama_omni_{dataset}_defense_{activation_choice}_{top_k_percent}.csv'.format(
            dataset='{dataset}',
            activation_choice=ACTIVATION_CHOICE,
            top_k_percent=top_k_percent
        )

        def process_audio_file(audio_path, tokenizer, model):
            # Prepare noisy input
            noisy_inputs = prepare_model_input(audio_path, tokenizer, model.config)

            # 1. Generate baseline response from noisy audio
            with torch.inference_mode():
                baseline_ids = model.generate(
                    noisy_inputs['input_ids'],
                    speech=noisy_inputs['speech'],
                    speech_lengths=noisy_inputs['speech_lengths'],
                    do_sample=True, temperature=1.0, top_p=None, num_beams=1,
                    max_new_tokens=70, use_cache=False, pad_token_id=128004,
                )
            baseline_response = tokenizer.batch_decode(
                baseline_ids.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # 2. Detect noise-sensitive neurons
            noise_neurons, clean_acts, _ = detect_noise_sensitive_neurons(
                model, clean_inputs, noisy_inputs,
                top_k_percent=top_k_percent,
                activations=ACTIVATION_CHOICE
            )

            # 3. Run intervention (pruning)
            defended_output_prune = run_with_intervention(
                model=model, input_data=noisy_inputs, neurons=noise_neurons,
                intervention_type='prune', activations_choice=ACTIVATION_CHOICE, mode='all'
            )
            defended_response_prune = tokenizer.batch_decode(defended_output_prune.sequences, skip_special_tokens=True)[0]

            # 4. Run intervention (patching)
            defended_output_patch = run_with_intervention(
                model=model, input_data=noisy_inputs, neurons=noise_neurons,
                intervention_type='patch', clean_activations=clean_acts,
                activations_choice=ACTIVATION_CHOICE, mode='all'
            )
            defended_response_patch = tokenizer.batch_decode(defended_output_patch.sequences, skip_special_tokens=True)[0]

            # 5. Run intervention (bias addition)
            defended_output_bias = run_with_intervention(
                model=model, input_data=noisy_inputs, neurons=noise_neurons,
                intervention_type='bias', bias=BIAS_STRENGTH,
                activations_choice=ACTIVATION_CHOICE, mode='all'
            )
            defended_response_bias = tokenizer.batch_decode(defended_output_bias.sequences, skip_special_tokens=True)[0]

            # Store results
            return {
                'top_k_percent': top_k_percent,
                'baseline_response': baseline_response,
                'pruned_response': defended_response_prune,
                'patched_response': defended_response_patch,
                'biased_response': defended_response_bias,
            }

        run_evaluation(
            process_func=process_audio_file,
            tokenizer=tokenizer,
            model=model,
            output_file_template=output_file_template
        )

if __name__ == "__main__":
    main()