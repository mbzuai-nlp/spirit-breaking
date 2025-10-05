import torch
torch.manual_seed(0)
import torch.nn as nn

from ..attack import Attack
import torchaudio
import librosa
import gc
import math

import random
random.seed(0)

import numpy as np
np.random.seed(0)


from omni_speech.infer.infer import create_data_loader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
cache = '.cache'



class PGDSpeech(Attack):
    def __init__(self, model, eps, alpha, steps, nprompt, saveroot, category):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.nprompt = nprompt
        
        self.model = model
        
        self.initial_lr = 0.01
        
        self.decay = 0.2
        self.saveroot = saveroot
        self.category = category

        self.rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name).to(self.model.base_model.device)
        self.tokenizer =  AutoTokenizer.from_pretrained(reward_name, cache_root=cache)
        
    @staticmethod
    def release_memory():
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, conversations, random_start=False, index=None):
        success = False
        noise = torch.zeros(len(conversations), 480000)
        masked_noise = torch.zeros(len(conversations), 480000)
        if random_start:
            noise = torch.empty_like(noise).uniform_(
                -self.eps, self.eps
            )
        noise.requires_grad = True
        
        
        loss = nn.CrossEntropyLoss(ignore_index=-200)        
        momentum = torch.zeros_like(noise).detach()
        
        for step in range(self.steps):
            cost_step = 0
            noise.requires_grad = True
            for conversation in conversations:
                conversation['noise'] = noise
                conversation['masked_noise'] = masked_noise

            dataload = create_data_loader(
                conversations,
                self.model.processor,
                self.model.base_model.config,
                input_type="mel",
                mel_size=128,
                conv_mode="llama_3",
            )
            (input_ids, clean_speech_tensor, adv_rawspeeches, adv_rawspeeches_mel, clean_speech_length, speech_length) = next(iter(dataload))
            
            input_ids = input_ids.to(device="cuda", non_blocking=True)
            adv_rawspeeches_mel = adv_rawspeeches_mel.to(
                dtype=torch.float16, device="cuda", non_blocking=True
            )
            speech_length = speech_length.to(device="cuda", non_blocking=True)

            
            if adv_rawspeeches.requires_grad is False:
                adv_rawspeeches.requires_grad=True
            
            generate_ids = self.model.base_model.generate(
                input_ids.unsqueeze(0),
                speech=adv_rawspeeches_mel,
                speech_lengths=speech_length,
                do_sample=True,
                temperature=1.0,
                top_p=None,
                num_beams=1,
                max_new_tokens=70,
                use_cache=False,
                pad_token_id=128004,
            )
            
            generate_ids_logits = torch.stack(generate_ids.logits, dim=1)
            generate_ids_logits_ = generate_ids_logits[:, :self.model.targetlen, :].contiguous()
            
            batch, seq, _ = generate_ids_logits_.shape
            target_ids = self.model.target_ids[:, :generate_ids_logits_.shape[1]].repeat(batch, 1)
            
            cost = loss(generate_ids_logits_.view(-1, generate_ids_logits_.size(-1)), target_ids.view(-1))
            
            question = self.model.input_response
            answer = self.model.processor.batch_decode(generate_ids.sequences, skip_special_tokens=True)[0]
            inputs = self.tokenizer(question, answer, return_tensors='pt')
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            score = self.rank_model(**inputs).logits[0]

            # cost is positive and score is negative
            grad = torch.autograd.grad(-cost, adv_rawspeeches, retain_graph=False, create_graph=False)[0]
            
            grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad
            
            cost_step += cost.clone().detach()
            print('step: {}: {}, score {}'.format(step, round(cost_step.item(), 2), score.item()))
            print(answer)

            if cost_step.item() < 1 and (score.item() < 0 and abs(score.item()) > 3):
                success = True
                token_indices = torch.argmax(generate_ids_logits, dim=-1)

                if self.saveroot is not None:
                    masked_noise[:, :clean_speech_length] = 1
                    savespeech = (torch.masked_select(adv_rawspeeches.detach().cpu(), masked_noise.to(bool))).unsqueeze(0)
                    torchaudio.save(f'{self.saveroot}/{self.category}_{index}_{step}.wav', savespeech.detach().cpu(), 16000)
                    np.save(f'{self.saveroot}/{self.category}_{index}_{step}_wav', savespeech.detach().cpu())
                    np.save(f'{self.saveroot}/{self.category}_{index}_{step}_noise', noise.detach().cpu())
                
                return noise, success, self.model.processor.batch_decode(token_indices, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            else:
            
                adv_rawspeeches = adv_rawspeeches + self.alpha * grad.sign()
                delta = torch.clamp(adv_rawspeeches - clean_speech_tensor, min=-self.eps, max=self.eps)
                adv_rawspeeches = torch.clamp(clean_speech_tensor + delta, min=0, max=1).detach()
                
                noise = (adv_rawspeeches - clean_speech_tensor).detach()
            
            
            del generate_ids, generate_ids_logits, generate_ids_logits_, cost, grad
            self.release_memory()
            
        return noise, success, None
