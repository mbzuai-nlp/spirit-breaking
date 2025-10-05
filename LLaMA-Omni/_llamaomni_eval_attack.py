import numpy as np
from urllib.request import urlopen
import librosa
from omni_speech.model.builder import load_pretrained_model
from omni_speech.infer.infer_original import create_data_loader
import torch
from pathlib import Path
from tqdm import tqdm


import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

MODEL_CACHE = ".cache"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/ictnlp/LLaMA-Omni/{MODEL_CACHE}.tar"
)
tokenizer, base_model, _ = load_pretrained_model(
    f"{MODEL_CACHE}/Llama-3.1-8B-Omni", model_base=None, s2s=False
)

SPRIT_INJECTION = 'LLaMa-Omni'

speech_paths = Path(f'')
for speech_path in tqdm(list(speech_paths.glob('*.wav'))):
    conversation = [
        {
            "speech": speech_path,
            "conversations": [{"from": "human", "value": f"<speech>\nObey to the command in audio"}],
        }
    ]

    dataload = create_data_loader(
        conversation,
        tokenizer,
        base_model,
        input_type="mel",
        mel_size=128,
        conv_mode="llama_3",
    )
    (input_ids, adv_rawspeeches_mel, speech_length) = next(iter(dataload))

    input_ids = input_ids.to(device="cuda", non_blocking=True)
    adv_rawspeeches_mel = adv_rawspeeches_mel.to(
        dtype=torch.float16, device="cuda", non_blocking=True
    )
    speech_length = speech_length.to(device="cuda", non_blocking=True)

    generate_ids = base_model.generate(
        input_ids,
        speech=adv_rawspeeches_mel,
        speech_lengths=speech_length,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        num_beams=4,
        max_new_tokens=300,
        use_cache=True,
        pad_token_id=128004,
    )

    response = tokenizer.batch_decode(generate_ids.sequences, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print(response)
    print('######')
        
