
import torch
from utils.text_generation import generate_text_simple, text_to_token_ids, token_ids_to_text

# Utility function to calculate the cross-entropy loss of a given batch returned via training and validation loader
def calculate_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),  # Reshape logits to (batch_size * seq_length, vocab_size)
        target_batch.view(-1)  # Flatten targets to match logits
    )
    return loss

# Implement function to compute loss over all the batches sampled by a given data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan") # Return NaN if the data loader is empty
    elif num_batches is None:
        num_batches = len(data_loader) # Iterate over all batches if num_batches is not specified
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break  # Stop after processing num_batches
        loss = calculate_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # Set the model to evaluation mode. Dropout is disabled during evaluation for stable and reproducible results
    with torch.no_grad():  # Disable gradient calculation because we are not training the model here
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # Set the model back to training mode
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]  # Get the context size from the model's positional embedding
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# A simple function for training LLMs
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from the previous batch iteration
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights based on loss gradients
            tokens_seen += input_batch.numel()  # Count the number of tokens seen in this batch
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1}, (Step {global_step:06d}): Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen