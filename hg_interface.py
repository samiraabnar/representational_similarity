import torch
import numpy as np

def get_word_idx(sent: str, word: str):
     return sent.split(" ").index(word)
 
 
def get_hidden_states(encoded, token_ids_word, model, layers, device):
     """Push input IDs through model. Stack and sum `layers` (last four by default).
        Select only those subword token outputs that belong to our word of interest
        and average them.""" 
     # Get all hidden states
     inputs = torch.tensor(encoded['input_ids']).to(device)
     attention_mask =  torch.tensor(encoded['attention_mask']).to(device)
     output = model(**encoded)
     states = output.hidden_states
     layers = layers or list(range(len(output.hidden_states)))
     # Stack all requested layers
     # Only select the tokens that constitute the requested word
     output = torch.stack([states[i].detach().squeeze()[token_ids_word] for i in layers])
     word_tokens_output = output.mean(axis=1).detach()
     return word_tokens_output.cpu().numpy()
 
 
def get_word_vector(sent, tokenizer, model, layers, device):
     """Get a word vector by first tokenizing the input sentence, getting all token idxs
        that make up the word of interest, and then `get_hidden_states`."""
     encoded = tokenizer.encode_plus(sent, return_tensors="pt")
     for key in encoded:
       encoded[key] = torch.tensor(encoded[key]).to(device)
     # get all token idxs that belong to the word of interest
     hidden_states = {}
     words = []
     for word_id in set(encoded.word_ids()):
      if word_id is not None:
        token_ids_word = np.where(np.array(encoded.word_ids()) == word_id)
        hidden_states.append(get_hidden_states(encoded, token_ids_word, model, layers))
        words.append(''.join(np.asarray(encoded.tokens())[token_ids_word]).replace('#', ''))
     return hidden_states
