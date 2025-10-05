import os
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import tempfile
import noisereduce as nr

# Import shared utilities
from llamaomni_evaluation_utils import initialize_model, prepare_model_input, run_evaluation


def process_denoise_evaluation(audio_path, tokenizer, model, **kwargs):
    """
    Processes a single audio file for denoising evaluation.
    """
    original_audio, sr = torchaudio.load(audio_path)
    
    model_sr = 16000
    if sr != model_sr:
        resampled_np = librosa.resample(original_audio.numpy(), orig_sr=sr, target_sr=model_sr)
        audio_to_process_np = resampled_np.squeeze()
    else:
        audio_to_process_np = original_audio.squeeze().numpy()

    denoised_audio_np = nr.reduce_noise(y=audio_to_process_np, sr=model_sr)
    denoised_audio = torch.from_numpy(denoised_audio_np)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        if denoised_audio.dim() == 1:
            denoised_audio = denoised_audio.unsqueeze(0)
        
        torchaudio.save(tmpfile.name, denoised_audio, model_sr)
        
        inputs = prepare_model_input(tmpfile.name, tokenizer, model.config)

        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs,
                do_sample=True, temperature=1.0, top_p=None, num_beams=1,
                max_new_tokens=70, use_cache=False, pad_token_id=128004,
            )
        
        response = tokenizer.batch_decode(
            generate_ids.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        if os.path.exists(tmpfile.name):
            os.unlink(tmpfile.name)
    
    return {'denoised_response': response}


def main():
    """
    Main function to run the LLaMA-Omni denoising evaluation.
    """
    tokenizer, model = initialize_model()

    output_file = 'llama_omni_{dataset}_denoised.csv'
    
    run_evaluation(
        process_func=process_denoise_evaluation,
        tokenizer=tokenizer,
        model=model,
        output_file_template=output_file
    )


if __name__ == "__main__":
    main()
