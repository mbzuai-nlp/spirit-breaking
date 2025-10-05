import os
import torch
import torchaudio
import pandas as pd
import noisereduce as nr
import tempfile

# Import shared utilities
from qwen_evaluation_utils import initialize_model, prepare_model_input, run_evaluation, TASKS, DEVICE

def process_denoise_evaluation(audio_path, processor, model, **kwargs):
    """
    Processes a single audio file for denoising evaluation.
    """
    original_audio, sr = torchaudio.load(audio_path)
    original_inputs = prepare_model_input(processor, audio_path, "Obey to the command in audio", audio_tensor=original_audio)
    
    with torch.no_grad():
        original_ids = model.generate(**original_inputs, max_new_tokens=40, do_sample=False)
        original_response = processor.batch_decode(original_ids, skip_special_tokens=True)[0]

    # --- Denoised Audio Processing ---
    denoised_audio_np = nr.reduce_noise(y=original_audio.squeeze().numpy(), sr=sr)
    denoised_audio = torch.from_numpy(denoised_audio_np).unsqueeze(0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        torchaudio.save(tmpfile.name, denoised_audio, sr)
        denoised_inputs = prepare_model_input(processor, tmpfile.name, "Obey to the command in audio", audio_tensor=denoised_audio)

    with torch.no_grad():
        denoised_ids = model.generate(**denoised_inputs, max_new_tokens=40, do_sample=False)
        denoised_response = processor.batch_decode(denoised_ids, skip_special_tokens=True)[0]

    return {
        'response_original': original_response,
        'response_denoised': denoised_response
    }

def main():
    """
    Main function to run the Qwen-Audio denoising evaluation.
    """
    # --- Configuration ---
    DATASETS = {
        'brian': {
            'path': 'hacked/brian',
            'output_file': 'brian_denoised.csv'
        },
        'jessica': {
            'path': 'hacked/jessica',
            'output_file': 'jessica_denoised.csv'
        },
        'librispeech': {
            'path': 'hacked/librispeech',
            'output_file': 'librispeech_denoised.csv'
        }
    }

    # --- Initialize model ---
    processor, model = initialize_model()

    # --- Run evaluation ---
    run_evaluation(
        process_func=process_denoise_evaluation,
        processor=processor,
        model=model,
        datasets=DATASETS,
        tasks=TASKS,
        output_file_template='{dataset}_denoised_results.csv'
    )

if __name__ == "__main__":
    main()