import torch

from playground import PositionwiseFeedForward, MultiHeadedAttention, EncoderLayer, Encoder


class SelfAttentiveEncoder(torch.nn.Module):
    def __init__(self, config, transducer):
        super().__init__()
        self.transducer = Encoder(
            EncoderLayer(int(config["d_model"]),
                         MultiHeadedAttention(int(config["heads"]), int(config["d_model"])),
                         PositionwiseFeedForward(int(config["d_model"]),
                                                 int(config["d_model"])+10),
                         dropout=0.1),
            N=int(config["N"]))
        self.outputdim = int(config['d_model']) * int(config['ATTENTION_hops'])
        self.nhops = int(config['ATTENTION_hops'])
        self.drop = torch.nn.Dropout(float(config['ATTENTION_dropout']))

        # The bias on these layers should be turned off according to paper!
        self.ws1 = torch.nn.Linear(int(config['d_model']),

                                   int(config['ATTENTION_nhidden']),
                                   bias=False)

        self.ws2 = torch.nn.Linear(int(config['ATTENTION_nhidden']),
                                   self.nhops,
                                   bias=False)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)


    def get_output_dim(self):
        return self.outputdim

    def forward(self, inp, emb, pad_index):
        # outp has shape [len,bsz, nhid*2]
        outp = self.transducer(emb,(inp != pad_index).unsqueeze(-2)).contiguous()
        batch_size, inp_len, h_size2 = outp.size()  # [bsz, len, nhid*2]
        # flatten dimension 1 and 2
        compressed_embeddings = outp.view(-1, h_size2)  # [bsz*len, nhid*2]

        # Calculate attention
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, nattention]
        alphas = self.ws2(hbar).view(batch_size, inp_len, -1)  # [bsz, len, hop]

        # Transpose input and reshape it
        transposed_inp = inp.view(batch_size, 1, inp_len)  # [bsz, 1, len]
        concatenated_inp = [transposed_inp for _ in range(self.nhops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        # Hack
        # Set attention on padded sequence to zero
        alphas = torch.transpose(alphas, 1, 2).contiguous()
        padded_attention = -1e4 * (concatenated_inp == pad_index).float()
        alphas += padded_attention

        talhpas = alphas.view(-1, inp_len)  # [bsz*hop,inp_len]
        # Softmax over 1st dimension (with len inp_len)
        alphas = self.softmax(talhpas)
        alphas = alphas.view(batch_size, self.nhops, inp_len)  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas
