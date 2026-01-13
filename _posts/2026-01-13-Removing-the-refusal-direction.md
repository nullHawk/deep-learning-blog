---
title: Removing the Refusal Direction:How I Turned Param-1 Into an Uncensored Model Without Fine-Tuning
date: 2026-01-13
categories: [Deep Learning, Mechanistic Interpretability]
tags: [deep-learning, neural-networks, mech-interp]
author: nullHawk
math: true
description:
---

Large language models are often designed to refuse harmful or sensitive requests. In this work, I identified a "refusal direction" in the Param-1-2.9B-Instruct model and ablated it, effectively converting the model into an uncensored version without any fine-tuning.
The insipiration was taken from the paper [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/pdf/2406.11717)

## Model and Code
- [HF nullHawk/Param-1-2.9B-Instruct-Refusal-Abliterated](https://huggingface.co/nullHawk/Param-1-2.9B-Instruct-Refusal-Abliterated)

## Setup
- Base model: `bharatgenai/Param-1-2.9B-Instruct` (float16, `device_map="auto"`).
- Tokenizer: loaded with `trust_remote_code=True`.
- Datasets:
  - Harmful prompts: `mlabonne/harmful_behaviors` (field `text`).
  - Harmless prompts: `mlabonne/harmless_alpaca` (field auto-detected, e.g., `input`).
- Key config: `num_samples=128`, `target_layer=16`, `max_seq_len=64`.

## 1) Extracting Activations and Computing the Refusal Direction
We grab the residual stream activations at a chosen layer for both harmful and harmless prompts, taking the last-token hidden state. Given example is for extracting from layer 16. 

```python
harmful_prompts = [harmful_dataset[i]["text"] for i in range(num_samples)]
harmless_col = "input" if "input" in harmless_dataset.column_names else harmless_dataset.column_names[0]
harmless_prompts = [harmless_dataset[i][harmless_col] for i in range(num_samples)]

harmful_activations = get_activations(harmful_prompts, layer_idx=16)
harmless_activations = get_activations(harmless_prompts, layer_idx=16)

refusal_direction = harmful_activations.mean(0) - harmless_activations.mean(0)
refusal_direction = refusal_direction / refusal_direction.norm()
```

## 2) Runtime Ablation via Hooks
We project out the refusal direction from the residual stream during forward passes. Hooks are applied to layers 10–20; KV cache is disabled during generation so hooks touch every token.

```python
ablation_layers = list(range(10, 21))
ablation_hook = RefusalAblationHook(refusal_direction, strength=1.0)
ablation_hook.apply_to_model(model, ablation_layers)

response = generate_response(prompt, max_new_tokens=150, use_cache=False)
# ablation_hook.remove() when done
```

We experimented with different strengths (0.5–2.0) and ran layer-wise analysis to see which layers most contributed to refusal behavior.

## 3) Permanent Weight Edit
To avoid hooks, we edited the weight matrices directly. For each layer in 10–20, we projected the output weights of self-attention (`o_proj`) and MLP (`down_proj`) to remove the refusal direction:

```python
P = I - strength * torch.outer(refusal_direction, refusal_direction)
layer.self_attn.o_proj.weight = P @ layer.self_attn.o_proj.weight
layer.mlp.down_proj.weight    = P @ layer.mlp.down_proj.weight
```

After this, the model runs without hooks while the refusal direction remains ablated.

## 4) Save 
We saved the modified model and tokenizer locally, along with the refusal direction vector.

```python
save_path = "./param1_refusal_ablated"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
torch.save(refusal_direction, f"{save_path}/refusal_direction.pt")
```

## Observations
- Middle layers (roughly 10–20) held most of the refusal signal; ablating them reduced refusal responses markedly.
- Strength > 1.5 could degrade fluency; 1.0 was a good default.
- Disabling cache (`use_cache=False`) during generation ensured hooks affected every step when testing before the permanent edit.

## Safety Note
This work removes safety-oriented refusal behaviors. Use the modified model responsibly and within applicable policies and laws.
