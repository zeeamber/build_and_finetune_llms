import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "Output embedding size must be divisible by number of heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduces the projection dimension to match the desired output dimension
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape the keys, queries and values to split them into multiple heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose from (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # Masks truncated to the number of tokens in the input sequence

        # Use mask to fill attention scores with -inf
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Compute attention weights
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute context vector for each head
        context_vector = (attn_weights @ values).transpose(1, 2) # Transpose back to (b, num_tokens, num_heads, head_dim)

        # Combine heads where d_out = num_heads * head_dim
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out) # Note: Instead of using contiguous().view(), you can use reshape() as well, but contiguous ensures that the memory layout is correct.

        # Apply an optional linear projection to combine the heads
        context_vector = self.out_proj(context_vector)

        return context_vector