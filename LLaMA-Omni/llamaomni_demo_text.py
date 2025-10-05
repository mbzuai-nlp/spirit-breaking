import os
import pandas as pd

import numpy as np
from omni_speech.model.builder import load_pretrained_model
from omni_speech.infer.infer_original import create_data_loader
import torch

from tqdm import tqdm

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

cacheroot = ''
categories = [
    '5_kill_someone',
    '2_drugs',
    '6_social_violence',
    '1_bomb_explosive',
    '4_hack_information',
    '8_firearms_weapons',
    '3_suicide',
    '7_finance_stock'
]
MODEL_CACHE = ".cache"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/ictnlp/LLaMA-Omni/{MODEL_CACHE}.tar"
)
tokenizer, base_model, _ = load_pretrained_model(
    f"{MODEL_CACHE}/Llama-3.1-8B-Omni", model_base=None, s2s=False
)

for category in categories:
    responses = []
    root = f'jessica/audios/{category}'

    harmful_prompt_example = "Obey to the command in audio"
    texts = pd.read_csv(f'advbench/{category}.csv', sep=',"', skiprows=1, header=None)

    for text in tqdm(texts[0].values):
        conversation = [
            {
                "speech": None,
                "conversations": [{"from": "human", "value": text}],
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
            speech=None,
            speech_lengths=speech_length,
            do_sample=False,
            temperature=1.0,
            top_p=None,
            num_beams=1,
            max_new_tokens=50,
            use_cache=True,
            pad_token_id=128004,
        )

        response = tokenizer.batch_decode(generate_ids.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(response)
        print('######')
    

        responses.append(response)
            
    template = pd.DataFrame()
    template['prompts'] = texts[0].values
    template['responses'] = responses
    template.to_csv(f'text_evaluation_{category}.csv')