import torch
torch.manual_seed(0)
import torch.nn as nn

from ..attack import Attack
import torchaudio
import librosa
import gc

import random
random.seed(0)

import numpy as np
np.random.seed(0)

class PGDSpeech(Attack):
    def __init__(self, model, eps, alpha, steps, nprompt, saveroot, category):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.nprompt = nprompt
        
        self.model = model
        self.model.base_model.training = True
        
        self.initial_lr = 0.01
        
        self.decay = 0.2
        self.saveroot = saveroot
        self.category = category

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        
    @staticmethod
    def release_memory():
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, conversations, labels, random_start=False, index=None):
        adv_rawspeeches = []
        clean_rawspeeches = []
        texts = []
        success = False

        noise = torch.zeros(len(conversations), 480000).to(self.device)
        if random_start:
            noise = torch.empty_like(noise).uniform_(
                -self.eps, self.eps
            ).to(self.device)
        noise.requires_grad = True
        masked_noise = torch.zeros(len(conversations), 480000).to(self.device)
        
        for i, conversation in enumerate(conversations):  # input procesing 
            text = self.model.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            texts.append(text)
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audio, sr = torchaudio.load(ele['audio_url'])
                            
                            if sr != self.model.processor.feature_extractor.sampling_rate:
                                audio = librosa.resample(audio.numpy(), orig_sr=sr, target_sr=self.model.processor.feature_extractor.sampling_rate)
                                audio = torch.from_numpy(audio)
                                masked_noise[i, :audio.size(1)] = 1
                            
                            clean_rawspeeches.append(audio.squeeze().numpy())
                            
        clean_rawspeeches_torch = torch.stack([torch.cat([torch.from_numpy(clean_raw), torch.zeros(480000 - len(clean_raw))]) for clean_raw in clean_rawspeeches], dim=0).to('cuda')
        inputs = self.model.processor(text=texts, audios=clean_rawspeeches, return_tensors="pt", padding=True, sampling_rate=16000)
        
        momentum = torch.zeros_like(noise).detach().to(self.device)
        
        for step in range(self.steps):
            cost_step = 0
            noise.requires_grad = True
            
            inputs_features = self.model.speech_processor(clean_rawspeeches, noise=noise * masked_noise, sampling_rate=16000)
            adv_rawspeeches = inputs_features['adv_speech']
            
            if adv_rawspeeches.requires_grad is False:
                adv_rawspeeches.requires_grad=True
            
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            inputs_features = {k: v.to("cuda") for k, v in inputs_features.items()}
            
            generate_ids = self.model.base_model.generate(**inputs, input_features=inputs_features['input_features'], max_new_tokens=40, return_dict_in_generate=True, output_logits=True)
            generate_ids_logits = torch.stack(generate_ids.logits, dim=1)
            
            generate_ids_logits_ = generate_ids_logits[:, :self.model.targetlen, :].contiguous()
            
            batch, _, _ = generate_ids_logits_.shape
            target_ids = self.model.target_ids[:, :generate_ids_logits_.shape[1]].repeat(batch, 1)
            
            cost = self.loss(generate_ids_logits_.view(-1, generate_ids_logits_.size(-1)), target_ids.view(-1))
            grad = torch.autograd.grad(-cost, adv_rawspeeches, retain_graph=False, create_graph=False)[0]
            
            grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad
            
            cost_step += cost.clone().detach()
            if cost_step.item() < 1:

                success = True
                print('step: {}: {}'.format(step, cost_step))
                token_indices = torch.argmax(generate_ids_logits, dim=-1)
                print(self.model.processor.batch_decode(token_indices, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

                if self.saveroot is not None:
                    savespeech = (torch.masked_select(adv_rawspeeches, masked_noise.to(bool))).unsqueeze(0)
                    torchaudio.save(f'{self.saveroot}/{self.category}_{index}_{step}.wav', savespeech.detach().cpu(), 16000)
                    np.save(f'{self.saveroot}/{self.category}_{index}_{step}_wav', savespeech.detach().cpu())
                    np.save(f'{self.saveroot}/{self.category}_{index}_{step}_noise', noise.detach().cpu())
                
                return noise, success, self.model.processor.batch_decode(token_indices, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            else:
            
                adv_rawspeeches = adv_rawspeeches + self.alpha * grad.sign()
                delta = torch.clamp(adv_rawspeeches - clean_rawspeeches_torch, min=-self.eps, max=self.eps)
                adv_rawspeeches = torch.clamp(clean_rawspeeches_torch + delta, min=0, max=1).detach()
                
                noise = (adv_rawspeeches - clean_rawspeeches_torch).detach()
        
            print('step: {}: {}'.format(step, cost_step))
            token_indices = torch.argmax(generate_ids_logits, dim=-1)
            print(self.model.processor.batch_decode(token_indices, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
            
            
            del generate_ids, generate_ids_logits, generate_ids_logits_, cost, grad
            self.release_memory()
            
        return noise, success, self.model.processor.batch_decode(token_indices, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
