# Loading OpenAI GPT-2 model weights

import numpy as np
import torch

# Utility function to check if the shapes of two tensors match and return the right tensor as trainable parameter
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        block_params = params["blocks"][b]
        q_w, k_w, v_w = np.split(block_params["attn"]["c_attn"]["w"], 3, axis=-1)
        att_module = gpt.trf_blocks[b].attn

        att_module.W_query.weight = assign(att_module.W_query.weight, q_w.T)
        att_module.W_key.weight = assign(att_module.W_key.weight, k_w.T)
        att_module.W_value.weight = assign(att_module.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(block_params["attn"]["c_attn"]["b"], 3, axis=-1)
        att_module.W_query.bias = assign(att_module.W_query.bias, q_b)
        att_module.W_key.bias = assign(att_module.W_key.bias, k_b)
        att_module.W_value.bias = assign(att_module.W_value.bias, v_b)

        att_module.out_proj.weight = assign(att_module.out_proj.weight, block_params["attn"]["c_proj"]["w"].T)
        att_module.out_proj.bias = assign(att_module.out_proj.bias, block_params["attn"]["c_proj"]["b"])

        ff_module = gpt.trf_blocks[b].ff
        ff_module.layers[0].weight = assign(ff_module.layers[0].weight, block_params["mlp"]["c_fc"]["w"].T)
        ff_module.layers[0].bias = assign(ff_module.layers[0].bias, block_params["mlp"]["c_fc"]["b"])
        ff_module.layers[2].weight = assign(ff_module.layers[2].weight, block_params["mlp"]["c_proj"]["w"].T)
        ff_module.layers[2].bias = assign(ff_module.layers[2].bias, block_params["mlp"]["c_proj"]["b"])

        layer_norm1 = gpt.trf_blocks[b].norm1

        layer_norm1.scale = assign(layer_norm1.scale, block_params["ln_1"]["g"])
        layer_norm1.shift = assign(layer_norm1.shift, block_params["ln_1"]["b"])

        layer_norm2 = gpt.trf_blocks[b].norm2
        layer_norm2.scale = assign(layer_norm2.scale, block_params["ln_2"]["g"])
        layer_norm2.shift = assign(layer_norm2.shift, block_params["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
