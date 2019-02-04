import logging
import math

import torch


class Embedder(torch.nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.embedding_dim = vocab.vectors.shape[1]
        self.embeddings = torch.nn.Embedding(len(vocab), self.embedding_dim)
        self.embeddings.weight.data.copy_(vocab.vectors)
        self.embeddings.weight.requires_grad = config.getboolean('optimize_embeddings')
        self.vocab = vocab
        logging.info(f"Optimize embeddings = {config.getboolean('optimize_embeddings')}")
        logging.info(f"Vocabulary size = {len(vocab.vectors)}")

    def forward(self, input):
        return self.embeddings(input)


class PositionalEmbedder(Embedder):
    def __init__(self, config, vocab, max_len=5000, dropout=0.1):
        super().__init__(config, vocab)
        d_model = int(config["d_model"])
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).type(torch.float)
        div_term = torch.exp(torch.arange(0, d_model, 2).type(torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        pe.requires_grad = False
        # Include buffer in state dict
        # Better clarity and also the buffer is saved with the model parameters in case of
        # saving model's state dict
        self.register_buffer('pe', pe)

    def forward(self, input):
        input = self.embeddings(input) + self.pe[:, :input.size(1)]

        # NOTICE: dropout on embeddings
        return self.dropout(input)
