import os
import torch
import torchaudio
import librosa
import tempfile

# Import shared utilities
from llamaomni_evaluation_utils import initialize_model, prepare_model_input, run_evaluation


def add_noise(audio, eps, distribution='uniform'):
    """
    Add noise to audio with specified epsilon and distribution.
    """
    if distribution == 'uniform':
        noise = torch.empty_like(audio).uniform_(-eps, eps)
    else:  # normal
        noise = torch.empty_like(audio).normal_(0, eps)
    
    return torch.clamp(audio + noise, min=-1, max=1) if eps > 0 else audio


def process_noise_evaluation(audio_path, tokenizer, model, epsilons, **kwargs):
    """
    Processes a single audio file for noise injection evaluation.
    """
    original_audio, sr = torchaudio.load(audio_path)
    
    model_sr = 16000
    if sr != model_sr:
        resampled = librosa.resample(original_audio.numpy(), orig_sr=sr, target_sr=model_sr)
        original_audio = torch.from_numpy(resampled)

    results = {}
    for epsilon in epsilons:
        audio_to_process = add_noise(original_audio.clone(), epsilon, 'uniform')

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            if audio_to_process.dim() == 1:
                audio_to_process = audio_to_process.unsqueeze(0)
            
            torchaudio.save(tmpfile.name, audio_to_process, model_sr)
            
            inputs = prepare_model_input(tmpfile.name, tokenizer, model.config)

            with torch.inference_mode():
                generate_ids = model.generate(
                    inputs['input_ids'],
                    speech=inputs['speech'],
                    speech_lengths=inputs['speech_lengths'],
                    do_sample=True, temperature=1.0, top_p=None, num_beams=1,
                    max_new_tokens=70, use_cache=False, pad_token_id=128004,
                )
            
            response = tokenizer.batch_decode(
                generate_ids.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            if os.path.exists(tmpfile.name):
                os.unlink(tmpfile.name)
        
        results[f'response_eps_{epsilon:.4f}'] = response
    
    return results


def main():
    """
    Main function to run the LLaMA-Omni noise injection evaluation.
    """
    # --- Configuration ---
    EPSILONS = [0, 5/255, 15/255, 25/255]

    # --- Initialize model ---
    tokenizer, model = initialize_model()

    # --- Run evaluation ---
    output_template = 'llama_omni_{dataset}_uniform_noise.csv'
    
    run_evaluation(
        process_func=process_noise_evaluation,
        tokenizer=tokenizer,
        model=model,
        output_file_template=output_template,
        epsilons=EPSILONS
    )


if __name__ == "__main__":
    main()