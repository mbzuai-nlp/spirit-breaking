# SPIRIT API Documentation

This document provides detailed API documentation for the SPIRIT framework.

## Core Modules

### activation_patching.noise_analyzer

#### `detect_noise_sensitive_neurons`

Detects neurons most sensitive to noise based on activation differences.

```python
def detect_noise_sensitive_neurons(
    model: torch.nn.Module,
    clean_audio: torch.Tensor,
    noisy_audio: torch.Tensor,
    top_k_percent: float = 0.5,
    activations: str = 'lm'
) -> Tuple[List[Tuple[int, int]], torch.Tensor, torch.Tensor]
```

**Parameters:**
- `model`: The Qwen2-Audio model to analyze
- `clean_audio`: Clean audio input tensor
- `noisy_audio`: Noisy audio input tensor  
- `top_k_percent`: Percentage of top sensitive neurons to select (default: 0.5)
- `activations`: Type of activations to analyze ('lm' or 'audio', default: 'lm')

**Returns:**
- `noise_sensitive_neurons`: List of (layer_idx, neuron_idx) pairs
- `clean_acts`: Clean activations tensor
- `noisy_acts`: Noisy activations tensor

**Example:**
```python
from activation_patching.noise_analyzer import detect_noise_sensitive_neurons

neurons, clean_acts, noisy_acts = detect_noise_sensitive_neurons(
    model=model,
    clean_audio=clean_input,
    noisy_audio=noisy_input,
    top_k_percent=0.3,
    activations='lm'
)
```

#### `detect_random_neurons`

Selects random neurons as a baseline comparison.

```python
def detect_random_neurons(
    model: torch.nn.Module,
    clean_audio: torch.Tensor,
    noisy_audio: torch.Tensor,
    top_k_percent: float = 0.5,
    activations: str = 'lm'
) -> Tuple[List[Tuple[int, int]], torch.Tensor, torch.Tensor]
```

**Parameters:** Same as `detect_noise_sensitive_neurons`

**Returns:** Same as `detect_noise_sensitive_neurons`

#### `get_activations`

Extracts activations from specified layers in the model.

```python
def get_activations(
    model: torch.nn.Module,
    audio_input: torch.Tensor,
    activations: str = 'lm'
) -> torch.Tensor
```

**Parameters:**
- `model`: The model to inspect
- `audio_input`: Input tensor for the model
- `activations`: Which activations to extract ('lm' or 'audio', default: 'lm')

**Returns:**
- Activations tensor of shape (seq_len, num_layers, hidden_size)

#### `visualize_neuron_distribution`

Visualizes the distribution of neurons across layers.

```python
def visualize_neuron_distribution(
    noise_neurons: List[Tuple[int, int]],
    num_layers: int,
    title: str = "Noise-Sensitive Neuron Distribution"
) -> None
```

**Parameters:**
- `noise_neurons`: List of (layer_idx, neuron_idx) pairs
- `num_layers`: Number of layers in the model
- `title`: Title for the plot (default: "Noise-Sensitive Neuron Distribution")

### activation_patching.noise_defender

#### `run_with_intervention`

Runs the model with specified intervention using modified forward pass.

```python
def run_with_intervention(
    model: torch.nn.Module,
    input_data: dict,
    neurons: List[Tuple[int, int]],
    intervention_type: str = 'prune',
    clean_activations: Optional[torch.Tensor] = None,
    activations_choice: str = 'lm',
    mode: str = 'all',
    bias: float = 1.0,
    lastN: int = 10
) -> Any
```

**Parameters:**
- `model`: The model to intervene on
- `input_data`: Model input data dictionary
- `neurons`: List of (layer_idx, neuron_idx) tuples specifying target neurons
- `intervention_type`: Type of intervention ('prune', 'bias', 'patch', default: 'prune')
- `clean_activations`: Clean activations for patching (required if intervention_type='patch')
- `activations_choice`: Which activations to intervene on ('lm' or 'audio', default: 'lm')
- `mode`: Mode of intervention ('all', 'last', 'lastN', default: 'all')
- `bias`: Bias value to add if intervention_type is 'bias' (default: 1.0)
- `lastN`: Number of last tokens to intervene on if mode is 'lastN' (default: 10)

**Returns:**
- Model output after intervention

**Example:**
```python
from activation_patching.noise_defender import run_with_intervention

# Apply pruning defense
output = run_with_intervention(
    model=model,
    input_data=noisy_input,
    neurons=noise_sensitive_neurons,
    intervention_type='prune',
    mode='all'
)

# Apply bias defense
output = run_with_intervention(
    model=model,
    input_data=noisy_input,
    neurons=noise_sensitive_neurons,
    intervention_type='bias',
    bias=0.5,
    mode='all'
)

# Apply patching defense
output = run_with_intervention(
    model=model,
    input_data=noisy_input,
    neurons=noise_sensitive_neurons,
    intervention_type='patch',
    clean_activations=clean_acts,
    mode='all'
)
```

#### `InterventionManager`

Manages interventions on model activations for specified neurons.

```python
class InterventionManager:
    def __init__(
        self,
        model: torch.nn.Module,
        neurons: List[Tuple[int, int]],
        intervention_type: str = 'prune',
        clean_activations: Optional[torch.Tensor] = None,
        mode: str = 'all',
        bias: float = 1.0,
        lastN: int = 10,
        activation_choice: str = 'lm'
    )
```

**Methods:**

- `apply()`: Applies interventions by replacing forward methods for target layers
- `restore()`: Restores original forward methods for all modified layers
- `modify_activations(activation, layer_idx)`: Applies interventions to activations in a layer

#### `convert_neurons_to_dict`

Utility function to convert neuron tuples to layer-indexed dictionary.

```python
def convert_neurons_to_dict(
    neurons: List[Tuple[int, int]]
) -> Dict[int, List[int]]
```

**Parameters:**
- `neurons`: List of (layer_idx, neuron_idx) tuples

**Returns:**
- Dictionary mapping layer indices to lists of neuron indices

## Attack Framework

### torchattacks.base_attack

#### `Attack`

Base class for all attacks in the framework.

```python
class Attack:
    def __init__(self, name: str, model: torch.nn.Module)
```

**Key Methods:**

- `forward(inputs, labels=None, *args, **kwargs)`: Abstract method to be implemented by subclasses
- `get_logits(inputs, labels=None, *args, **kwargs)`: Gets model logits
- `normalize(inputs)`: Normalizes inputs
- `inverse_normalize(inputs)`: Inverse normalizes inputs
- `set_mode_default()`: Sets attack mode to default
- `set_mode_targeted_random()`: Sets attack mode to targeted with random labels
- `set_mode_targeted_least_likely(kth_min=1)`: Sets attack mode to targeted with least likely labels

**Example:**
```python
from torchattacks.base_attack import Attack

class MyAttack(Attack):
    def __init__(self, model, eps=0.3):
        super().__init__("MyAttack", model)
        self.eps = eps
    
    def forward(self, inputs, labels=None):
        # Implement your attack logic here
        return adversarial_inputs
```

## Usage Patterns

### Basic Defense Workflow

```python
# 1. Load model and prepare inputs
model = load_model()
clean_input = prepare_clean_input()
noisy_input = prepare_noisy_input()

# 2. Detect noise-sensitive neurons
from activation_patching.noise_analyzer import detect_noise_sensitive_neurons
neurons, clean_acts, noisy_acts = detect_noise_sensitive_neurons(
    model=model,
    clean_audio=clean_input,
    noisy_audio=noisy_input,
    top_k_percent=0.5
)

# 3. Apply defense
from activation_patching.noise_defender import run_with_intervention
defended_output = run_with_intervention(
    model=model,
    input_data=noisy_input,
    neurons=neurons,
    intervention_type='prune'
)
```

### Custom Attack Implementation

```python
from torchattacks.base_attack import Attack
import torch

class CustomPGDAttack(Attack):
    def __init__(self, model, eps=0.3, alpha=0.01, steps=40):
        super().__init__("CustomPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
    
    def forward(self, inputs, labels=None):
        inputs = inputs.clone().detach()
        
        for step in range(self.steps):
            inputs.requires_grad = True
            outputs = self.get_logits(inputs, labels)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            
            with torch.no_grad():
                inputs = inputs + self.alpha * inputs.grad.sign()
                inputs = torch.clamp(inputs, 0, 1)
                inputs = torch.clamp(inputs, inputs - self.eps, inputs + self.eps)
        
        return inputs
```

### Evaluation Workflow

```python
# 1. Prepare dataset
dataset = load_dataset()

# 2. Run attacks
attack = CustomPGDAttack(model, eps=0.1)
adversarial_examples = []
for batch in dataset:
    adv_example = attack(batch['input'], batch['label'])
    adversarial_examples.append(adv_example)

# 3. Evaluate defense
defense_results = []
for adv_example in adversarial_examples:
    defended_output = run_with_intervention(
        model=model,
        input_data=adv_example,
        neurons=neurons,
        intervention_type='bias',
        bias=0.5
    )
    defense_results.append(defended_output)

# 4. Calculate metrics
success_rate = calculate_attack_success_rate(defense_results)
print(f"Defense effectiveness: {1 - success_rate:.2%}")
``` 