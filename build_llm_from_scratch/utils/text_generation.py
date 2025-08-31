import torch

# A simple text generation function
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Keep only the last context_size tokens

        with torch.no_grad(): # Disable gradient calculation because that is just inefficient for this purpose
            logits = model(idx_cond)

        # Get the last token's logits
        logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size). Last row.
        probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # Get the index of the most probable token

        # Append the predicted token to the input sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# A modified text generation function with more diversity in the output
def generate_text(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Keep only the last context_size tokens

        with torch.no_grad():
            logits = model(idx_cond)
        # Get the last token's logits
        logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size). Last row.

        if top_k is not None:
            top_logits, top_pos = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits >= min_val, logits, torch.tensor(float('-inf')).to(logits.device))

        if temperature > 0.0:  # This also avoids division by zero
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break  # Stop if the end-of-sequence token is generated
        idx = torch.cat((idx, idx_next), dim=1)  # Append the predicted token to the input sequence

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # Remove batch dimension
    return tokenizer.decode(flat.tolist())