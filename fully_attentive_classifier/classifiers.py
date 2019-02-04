import torch

from fully_attentive_classifier.encoders import SelfAttentiveEncoder


class SelfAttentiveClassifier(torch.nn.Module):
    def __init__(self, config, vocab, classes, embed_klazz, transducer):
        super(SelfAttentiveClassifier, self).__init__()
        self.embedder = embed_klazz(config, vocab)
        self.ctx_encoder = SelfAttentiveEncoder(config, transducer)
        hidden_size = int(config['FC_nhidden'])
        # classin_dim = int(config['RNN_nhidden']) * 2 * int(config['ATTENTION_hops'])
        classin_dim = int(config['d_model']) * int(config['ATTENTION_hops'])
        self.drop = torch.nn.Dropout(float(config['OUTPUT_dropout']))
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(classin_dim, hidden_size)
        self.outplinear = torch.nn.Linear(hidden_size, classes)
        # Softmax is included in CE loss function

    def embedded_dropout(self, embed, words, dropout=0.1):
        mask = embed.weight.data.new_empty((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
        return torch.nn.functional.embedding(words, masked_embed_weight)

    def forward(self, inp):
        emb = self.embedder(inp)
        outp, attention = self.ctx_encoder.forward(inp, emb, self.embedder.vocab.stoi['<pad>'])
        # Flatten all dimensions except batch
        outp = outp.view(outp.size(0), -1)  # batchsize x 2*nhid
        pred_logits = self.outplinear(self.drop(self.relu(self.fc(outp))))
        return pred_logits, attention
