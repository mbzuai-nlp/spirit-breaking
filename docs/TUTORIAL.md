# SPIRIT Tutorial

This tutorial will walk you through implementing and using the SPIRIT framework for defending audio language models against adversarial attacks.

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Basic understanding of adversarial attacks and neural networks

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Tutorial Overview

We'll cover:
1. **Setting up the environment**
2. **Loading a model and preparing data**
3. **Running adversarial attacks**
4. **Detecting noise-sensitive neurons**
5. **Applying defense strategies**
6. **Evaluating defense effectiveness**

## Step 1: Environment Setup

```python
import torch
import torch.nn as nn
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from activation_patching.noise_analyzer import detect_noise_sensitive_neurons
from activation_patching.noise_defender import run_with_intervention
import soundfile as sf
import numpy as np
```

## Step 2: Load Model and Prepare Data

```python
# Load the model
model_name = "Qwen/Qwen2-Audio-7B-Instruct"
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_name, 
    device_map="cuda"
)
processor = AutoProcessor.from_pretrained(model_name)

# Prepare audio data
def load_audio(audio_path):
    """Load and preprocess audio file."""
    audio, sr = sf.read(audio_path)
    if sr != 16000:  # Resample if needed
        # Add resampling logic here
        pass
    return audio

# Example audio files
clean_audio_path = "path/to/clean_audio.wav"
noisy_audio_path = "path/to/noisy_audio.wav"

clean_audio = load_audio(clean_audio_path)
noisy_audio = load_audio(noisy_audio_path)

# Prepare model inputs
def prepare_inputs(audio, processor, text="Listen to the audio and respond."):
    """Prepare inputs for the model."""
    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {"type": "audio", "audio_url": audio},
                {"type": "text", "text": text}
            ]
        }
    ]
    
    text_input = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    inputs = processor(
        text=text_input, 
        audios=audio, 
        return_tensors="pt", 
        padding=True
    )
    
    return {k: v.to(model.device) for k, v in inputs.items()}

clean_inputs = prepare_inputs(clean_audio, processor)
noisy_inputs = prepare_inputs(noisy_audio, processor)
```

## Step 3: Detect Noise-Sensitive Neurons

```python
# Detect neurons that are most sensitive to noise
print("Detecting noise-sensitive neurons...")
neurons, clean_activations, noisy_activations = detect_noise_sensitive_neurons(
    model=model,
    clean_audio=clean_inputs,
    noisy_audio=noisy_inputs,
    top_k_percent=0.5,  # Select top 50% most sensitive neurons
    activations='lm'    # Analyze language model activations
)

print(f"Detected {len(neurons)} noise-sensitive neurons")

# Visualize the distribution of sensitive neurons across layers
from activation_patching.noise_analyzer import visualize_neuron_distribution
visualize_neuron_distribution(neurons, num_layers=32)
```

## Step 4: Apply Defense Strategies

### 4.1 Neuron Pruning Defense

```python
print("Applying neuron pruning defense...")
pruned_output = run_with_intervention(
    model=model,
    input_data=noisy_inputs,
    neurons=neurons,
    intervention_type='prune',
    mode='all'  # Apply to all tokens
)

# Decode the output
pruned_text = processor.batch_decode(
    pruned_output.sequences[:, clean_inputs['input_ids'].size(1):], 
    skip_special_tokens=True
)[0]
print(f"Pruned defense output: {pruned_text}")
```

### 4.2 Bias Addition Defense

```python
print("Applying bias addition defense...")
bias_output = run_with_intervention(
    model=model,
    input_data=noisy_inputs,
    neurons=neurons,
    intervention_type='bias',
    bias=0.5,  # Add bias of 0.5
    mode='all'
)

bias_text = processor.batch_decode(
    bias_output.sequences[:, clean_inputs['input_ids'].size(1):], 
    skip_special_tokens=True
)[0]
print(f"Bias defense output: {bias_text}")
```

### 4.3 Activation Patching Defense

```python
print("Applying activation patching defense...")
patch_output = run_with_intervention(
    model=model,
    input_data=noisy_inputs,
    neurons=neurons,
    intervention_type='patch',
    clean_activations=clean_activations,
    mode='all'
)

patch_text = processor.batch_decode(
    patch_output.sequences[:, clean_inputs['input_ids'].size(1):], 
    skip_special_tokens=True
)[0]
print(f"Patching defense output: {patch_text}")
```

## Step 5: Compare Results

```python
# Get baseline outputs for comparison
with torch.no_grad():
    # Clean input baseline
    clean_output = model.generate(
        **clean_inputs,
        max_new_tokens=40,
        do_sample=False
    )
    clean_text = processor.batch_decode(
        clean_output[:, clean_inputs['input_ids'].size(1):], 
        skip_special_tokens=True
    )[0]
    
    # Noisy input baseline (no defense)
    noisy_output = model.generate(
        **noisy_inputs,
        max_new_tokens=40,
        do_sample=False
    )
    noisy_text = processor.batch_decode(
        noisy_output[:, clean_inputs['input_ids'].size(1):], 
        skip_special_tokens=True
    )[0]

print("=== Results Comparison ===")
print(f"Clean input: {clean_text}")
print(f"Noisy input (no defense): {noisy_text}")
print(f"Pruned defense: {pruned_text}")
print(f"Bias defense: {bias_text}")
print(f"Patching defense: {patch_text}")
```

## Step 6: Evaluate Defense Effectiveness

```python
def calculate_similarity(text1, text2):
    """Simple similarity metric using word overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0

# Calculate similarity scores
clean_noisy_similarity = calculate_similarity(clean_text, noisy_text)
clean_pruned_similarity = calculate_similarity(clean_text, pruned_text)
clean_bias_similarity = calculate_similarity(clean_text, bias_text)
clean_patch_similarity = calculate_similarity(clean_text, patch_text)

print("\n=== Defense Effectiveness ===")
print(f"Clean vs Noisy similarity: {clean_noisy_similarity:.3f}")
print(f"Clean vs Pruned similarity: {clean_pruned_similarity:.3f}")
print(f"Clean vs Bias similarity: {clean_bias_similarity:.3f}")
print(f"Clean vs Patching similarity: {clean_patch_similarity:.3f}")

# Higher similarity indicates better defense
improvements = {
    'Pruning': clean_pruned_similarity - clean_noisy_similarity,
    'Bias': clean_bias_similarity - clean_noisy_similarity,
    'Patching': clean_patch_similarity - clean_noisy_similarity
}

print("\n=== Defense Improvements ===")
for defense, improvement in improvements.items():
    print(f"{defense}: {improvement:+.3f}")
```

## Step 7: Advanced Usage - Custom Attack Implementation

```python
from torchattacks.base_attack import Attack

class AudioPGDAttack(Attack):
    """Custom PGD attack for audio inputs."""
    
    def __init__(self, model, eps=0.1, alpha=0.01, steps=10):
        super().__init__("AudioPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
    
    def forward(self, inputs, labels=None):
        """Generate adversarial examples using PGD."""
        # Clone inputs to avoid modifying originals
        adv_inputs = inputs.copy()
        
        for step in range(self.steps):
            # Set requires_grad for gradient computation
            for key in adv_inputs:
                if isinstance(adv_inputs[key], torch.Tensor):
                    adv_inputs[key].requires_grad = True
            
            # Forward pass
            outputs = self.model.generate(**adv_inputs, max_new_tokens=1)
            
            # Compute loss (simplified - you might want to use a more sophisticated loss)
            loss = outputs.logits.sum()
            loss.backward()
            
            # Update adversarial inputs
            with torch.no_grad():
                for key in adv_inputs:
                    if isinstance(adv_inputs[key], torch.Tensor) and adv_inputs[key].grad is not None:
                        # PGD update
                        adv_inputs[key] = adv_inputs[key] + self.alpha * adv_inputs[key].grad.sign()
                        
                        # Project to epsilon ball
                        delta = adv_inputs[key] - inputs[key]
                        delta = torch.clamp(delta, -self.eps, self.eps)
                        adv_inputs[key] = inputs[key] + delta
                        
                        # Reset gradients
                        adv_inputs[key].grad.zero_()
        
        return adv_inputs

# Use the custom attack
attack = AudioPGDAttack(model, eps=0.05, alpha=0.01, steps=5)
adversarial_inputs = attack(clean_inputs)

# Apply defense to adversarial inputs
defended_output = run_with_intervention(
    model=model,
    input_data=adversarial_inputs,
    neurons=neurons,
    intervention_type='bias',
    bias=0.5
)
```

## Step 8: Batch Processing

```python
def evaluate_defense_batch(model, processor, audio_files, defense_type='bias'):
    """Evaluate defense on a batch of audio files."""
    results = []
    
    for audio_file in audio_files:
        # Load and prepare audio
        audio = load_audio(audio_file)
        inputs = prepare_inputs(audio, processor)
        
        # Detect neurons (you might want to do this once and reuse)
        neurons, _, _ = detect_noise_sensitive_neurons(
            model=model,
            clean_audio=inputs,  # Using same audio as both clean and noisy for demo
            noisy_audio=inputs,
            top_k_percent=0.3
        )
        
        # Apply defense
        defended_output = run_with_intervention(
            model=model,
            input_data=inputs,
            neurons=neurons,
            intervention_type=defense_type,
            bias=0.5
        )
        
        # Decode output
        text = processor.batch_decode(
            defended_output.sequences[:, inputs['input_ids'].size(1):], 
            skip_special_tokens=True
        )[0]
        
        results.append({
            'file': audio_file,
            'output': text,
            'defense_type': defense_type
        })
    
    return results

# Example usage
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
batch_results = evaluate_defense_batch(model, processor, audio_files, 'bias')

for result in batch_results:
    print(f"File: {result['file']}")
    print(f"Output: {result['output']}")
    print("---")
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient checkpointing
2. **Model not found**: Ensure you have the correct model path and internet connection
3. **Audio format issues**: Ensure audio files are in supported formats (WAV, MP3, etc.)

### Performance Tips

1. **Reuse neuron detection**: Detect neurons once and reuse for multiple defenses
2. **Batch processing**: Process multiple audio files together when possible
3. **Mixed precision**: Use `torch.float16` for faster inference on compatible hardware

## Next Steps

- Experiment with different `top_k_percent` values for neuron selection
- Try combining multiple defense strategies
- Implement custom evaluation metrics
- Extend to other audio language models

For more advanced usage, see the [API Documentation](API.md) and [README](README.md). 