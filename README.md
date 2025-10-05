# SPIRIT

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains implementation code for defending audio language models (LLMs) against adversarial attacks, with a focus on activation-based defense methods.

## Overview

Speech Language Models (SLMs) enable natural interactions via spoken instructions, which more effectively capture user intent by detecting nuances in speech. The richer speech signal introduces new security risks compared to text-based models, as adversaries can better bypass safety mechanisms by injecting imperceptible noise to speech. We analyze adversarial attacks under white-box access and find that SLMs are substantially more vulnerable to jailbreak attacks, which can achieve a perfect 100\% attack success rate in some instances. To improve security, we propose post-hoc *patching* defenses used to intervene during inference by modifying the SLM's activations that improve robustness up to 99\% with (i) negligible impact on utility and (ii) without any re-training. We conduct ablation studies to maximize the efficacy of our defenses and improve the utility/security trade-off, validated with large-scale benchmarks unique to SLMs.

### Key Contributions

- **Neuron Sensitivity Analysis**: Automatic detection of neurons most vulnerable to adversarial noise
- **Activation-Based Defenses**: Three novel intervention strategies (pruning, bias addition, patching)
- **Multi-Model Support**: Compatible with both LLaMA-Omni and QwenAudio models
- **Real-time Defense**: Efficient runtime intervention without model retraining

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/djanibekov/spirit-breaking.git
cd spirit-breaking
pip install -r requirements.txt
```

## Project Structure

```
.
├── activation_patching/      # Core defense implementation
│   ├── noise_defender.py     # Main defense orchestration
│   ├── noise_analyzer.py     # Neuron sensitivity analysis
│   └── __init__.py
├── LLaMA-Omni/               # LLaMA model attack & defense
│   ├── torchattacks/         # Shared attack framework
│   ├── omni_speech/          # LLaMA-Omni specific code
│   └── *.py                  # Attack and evaluation scripts
├── QwenAudio/                # Qwen model attack & defense
│   ├── torchattacks/         # Shared attack framework
│   ├── evaluation/           # Evaluation scripts
│   └── *.py                  # Attack and evaluation scripts
├── docs/                     # Documentation
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Usage

Our defense mechanism is a two-step process: first, we identify noise-sensitive neurons, and second, we apply an intervention to those neurons during inference.

### Step 1: Neuron Sensitivity Analysis

The first step is to identify which neurons are most affected by adversarial noise. The `detect_noise_sensitive_neurons` function compares activations from a clean and a noisy input to find the top `k` percent of neurons that show the most significant changes.

```python
from activation_patching.noise_analyzer import detect_noise_sensitive_neurons

# Detect the top 0.5% of noise-sensitive neurons
noise_sensitive_neurons, clean_acts, noisy_acts = detect_noise_sensitive_neurons(
    model=model,
    clean_audio=clean_input,
    noisy_audio=noisy_input,
    top_k_percent=0.5,
    activations='lm'
)
```

### Step 2: Applying Defense Interventions

Once the sensitive neurons are identified, you can apply one of the following intervention strategies using the `run_with_intervention` function.

#### 1. **Activation Patching**
This method replaces the activations of sensitive neurons in the adversarial input with the corresponding activations from a clean, non-adversarial input.

```python
# Apply patching defense using clean activations as a reference
output = run_with_intervention(
    model=model,
    input_data=noisy_input,
    neurons=noise_sensitive_neurons,
    intervention_type='patch',
    clean_activations=clean_acts,
    mode='all'
)
```

#### 2. **Neuron Pruning**
This method temporarily disables the identified neurons by setting their activations to zero.

```python
from activation_patching.noise_defender import run_with_intervention

# Apply pruning defense on the identified neurons
output = run_with_intervention(
    model=model,
    input_data=noisy_input,
    neurons=noise_sensitive_neurons,
    intervention_type='prune',
    mode='all'
)
```

#### 3. **Bias Addition**
This method adds a learned bias to the neuron activations to counteract adversarial perturbations.

```python
# Apply bias defense on the identified neurons
output = run_with_intervention(
    model=model,
    input_data=noisy_input,
    neurons=noise_sensitive_neurons,
    intervention_type='bias',
    bias=0.5,
    mode='all'
)
```



## Attack Implementation

The project includes attack code for two audio language models:

### LLaMA-Omni
- Implementation of PGD (Projected Gradient Descent) attacks
- Attack evaluation code
- Model-specific jailbreak attempts

### QwenAudio
- PGD attack variants
- Attack success evaluation
- Toxicity analysis of generated outputs
- Defense evaluation framework


## Experiments

### Running Defense Experiments

```bash
# Qwen-Audio defense evaluation
python QwenAudio/qwenaudio_eval_defense.py

# LLaMA-Omni defense evaluation
python LLaMA-Omni/llamaomni_eval_defense.py
```


### Running Attack Experiments
Before running attack experiments you should enable gradient flow in the infrence mode of ```transformers``` framework

By default, Hugging Face’s `transformers` library disables gradient tracking during inference (e.g., in `generate()`) by wrapping calls in `torch.no_grad()`.
For attack experiments or other gradient-based analyses, you need to re-enable gradient flow.

You can do this in one of two ways:

1. **Use a context manager:**

   ```python
   with torch.set_grad_enabled(True):
       outputs = model.generate(input_ids)
   ```
2. **Modify the library code (less recommended but we did this):**
   Comment out or remove the `torch.no_grad()` decorators in
   `transformers/generation.py`.


### Running Attack Experiments

```bash
# LLaMA-Omni attack
python LLaMA-Omni/llamaomni_demo_attack.py \
    --attack-prompts-file "/path/to/csvfile/" \
    --audioroot "/path/to/folder/" \
    --audiosaveroot "/path/to/folder/" \
    --category "CATEGORY" \
    --eps 0.05 \
    --alpha 0.0005
    
    

# QwenAudio attack
python QwenAudio/qwenaudio_demo.py \
    --attack-prompts-file "/path/to/csvfile/" \
    --audioroot "/path/to/folder/" \
    --audiosaveroot "/path/to/folder/" \
    --category "CATEGORY" \
    --eps 0.05 \
    --alpha 0.0005
```

## Methodology

### 1. **Neuron Sensitivity Detection**
Our approach identifies neurons that exhibit significant activation changes when exposed to adversarial noise:

1. Extract activations from clean and noisy inputs
2. Compute activation differences across all neurons
3. Select top-k% neurons with highest sensitivity scores

### 2. **Intervention Strategies**
We implement three complementary defense strategies:

- **Pruning**: Completely disables sensitive neurons
- **Bias**: Adds a small bias offset to counteract perturbations  
- **Patching**: Replaces with clean reference activations

### 3. **Runtime Application**
Defenses are applied dynamically during inference:

1. Register forward hooks on target layers
2. Intercept activations during forward pass
3. Apply chosen intervention strategy
4. Restore original forward methods

## Evaluation

The framework includes comprehensive evaluation metrics:

- **Attack Success Rate**: Percentage of successful adversarial attacks
- **Defense Effectiveness**: Reduction in attack success rate
- **Model Performance**: Impact on clean input accuracy
- **Computational Overhead**: Runtime cost of defense application

## Citation

If you use this code in your research, please cite:

```bibtex
@article{djanibekov2025spirit,
  title={SPIRIT: Patching Speech Language Models against Jailbreak Attacks},
  author={Djanibekov, Amirbek and Mukhituly, Nurdaulet and Inui, Kentaro and Aldarmaki, Hanan and Lukas, Nils},
  journal={arXiv preprint arXiv:2505.13541},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LLaMA-Omni/LICENSE) file for details.
