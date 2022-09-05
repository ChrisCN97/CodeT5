# 1. insert prompt to dataset
# 2. prompt vector through LSTM
# 3. replace prompt vector to dataset input

import torch
from torch import nn

use_ptuning = True
use_T5 = True

prompt_id = 50
prompt_len = 5

prompt_str = " ".join(["<prompt"+str(i)+">" for i in range(prompt_len)])

def add_prompt_into_ids(all_source_ids):
    if not use_ptuning or use_T5:
        return all_source_ids
    num = all_source_ids.shape[0]
    prompt_ids = torch.full((num, prompt_len), prompt_id)
    return torch.cat((prompt_ids, all_source_ids), 1)

class Prompt(torch.nn.Module):
    def __init__(self, embed_size):
        super(Prompt, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = self.embed_size
        self.prompt_length = prompt_len
        # The pattern_id is supposed to indicate the number of continuous prompt tokens.

        self.prompt_embeddings = torch.nn.Embedding(self.prompt_length, self.embed_size)
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size,
                                       num_layers=2,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))

    def replace_prompt_vector_into_inputs(self, source_ids, word_embeddings):
        if not use_ptuning:
            return None
        raw_embeds = word_embeddings(source_ids)
        replace_embeds = self.prompt_embeddings(
            torch.LongTensor(list(range(self.prompt_length))).cuda())
        replace_embeds = replace_embeds.unsqueeze(0)
        replace_embeds = self.lstm_head(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
        if self.prompt_length == 1:
            replace_embeds = self.mlp_head(replace_embeds)
        else:
            replace_embeds = self.mlp_head(replace_embeds).squeeze()
        for bidx in range(raw_embeds.shape[0]):
            for i in range(self.prompt_length):
                raw_embeds[bidx, i, :] = replace_embeds[i, :]
        return raw_embeds

def add_prompt_for_t5(model, tokenizer):
    prompt_tokens = ["<prompt"+str(i)+">" for i in range(prompt_len)]
    tokenizer.add_tokens(list(prompt_tokens))
    model.resize_token_embeddings(len(tokenizer))

def add_prompt_to_str_for_t5(str):
    if not use_ptuning or not use_T5:
        return str
    return prompt_str + str