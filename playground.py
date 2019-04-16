import copy
import logging
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import wget
from torchtext import data, datasets

from util import setup_logging

seaborn.set_context(context="talk")
VERBOSE = False

"""
Implemented according to tutorial http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # TODO: inputs to Sublayer are ALWAYS normalized, initial
        # Embeddings are normalized too!
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        # TODO: Why norm at the end?
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 2 because there are 2 sublayers in each block
        # 1st sublayers does multi-head attention
        # 2nd sublayer does position-wise feed forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """

        :param x:
        :param memory: contains encoded input
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    # In[6]: subsequent_mask(5)[0]
    # Out[6]:
    # tensor([[1, 0, 0, 0, 0],
    #         [1, 1, 0, 0, 0],
    #         [1, 1, 1, 0, 0],
    #         [1, 1, 1, 1, 0],
    #         [1, 1, 1, 1, 1]], dtype=torch.uint8)
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    if VERBOSE:
        print("Computing attention:")
        print(f"Q shape:{query.shape}")
        print(f"K shape:{key.shape}")
        print(f"V shape:{value.shape}")
    # compute similarity of query-to-key vectors via dot product
    # normalize it via length of dimension
    #
    # From the paper:
    # The two most commonly used attention functions are
    #   additive attention
    #   dot-product (multiplicative) attention.
    #
    # Dot-product attention is identical to our algorithm, except for the scaling factor of 1/√d_k.
    # Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.
    # While the two are similar in theoretical complexity, dot-product attention is much faster and more
    # space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
    #
    # While for small values of d_k the two mechanisms perform similarly, additive attention outperforms
    # dot product attention without scaling for larger values of d_k. We suspect that for large values of
    # d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has
    # extremely small gradients
    #
    # To illustrate why the dot products get large, assume that the components of
    # q and k are independent random variables with mean 0 and variance 1. Then their dot product, q⋅k= i from {1 ... d_k} ∑qiki,
    # has mean 0 and variance d_k.
    # To counteract this effect, we scale the dot products by 1/√d_k.
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    # scores has shape
    # BATCH x HEADS x LEN_QUERY x LEN_KEY
    if mask is not None:
        # masked fill is broadcastable
        # dimensions 1 and 2 are broadcasted
        # mask through the dimension
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # Mask has shape BATCH x 1 x S x LEN
    # where S is:  1 for src sequence data
    #            LEN for tgt sequence data

    # NOTICE: Dropout on attention
    if dropout is not None:
        p_attn = dropout(p_attn)

    # The result is
    # KEY aware query representation
    # It will have length BATCH x HEADS x Query_LEN x d_k
    # where there is Query_LEN d_k vectors, each mixed from learned
    # weighted average of value vectors
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // heads  # dimensionality over one head
        self.h = heads
        # 4 - for query, key, value transformation + output transformation
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # Query has shape BATCH x LEN x D_MODEL
        # Mask has shape BATCH x S x LEN
        # where S is:  1 for src sequence data
        #            LEN for tgt sequence data

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # mask is now BATCH x 1 x S x LEN
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        # If query,key,value are of size bsz x length x d_model
        # this code transforms query, key and value with d_model x d_model matrices
        # and splits each into bsz x h (number of splits) x length x d_k
        # query_, key_, value_ = \
        #     [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #      for l, x in zip(self.linears, (query, key, value))]

        # Rewritten into more clear representation
        query = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linears[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linears[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # assert torch.equal(query, query_) and torch.equal(key, key_) and torch.equal(value, value_)

        # 2) Apply attention on all the projected vectors in batch.
        # x has shape bsz x length x d_model
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=mask,
                                                    dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    From the paper:
    "In addition to attention sub-layers, each of the layers in our encoder and decoder contains
    a fullyconnected feed-forward network, which is applied to each position separately and
    identically. This consists of two linear transformations with a ReLU activation in between.
    FFN(x) = max(0,xW1+b1)W2+b2
    While the linear transformations are the same across different positions, they use
    different parametersfrom layer to layer. Another way of describing this is as
    two convolutions with kernel size 1. The dimensionality of input and output is
    dmodel= 512, and the inner-layer has dimensionality dff= 2048."
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    # Embeddings are learned jointly
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """

    Hypothesis behind encodings

    From the paper:
    "Wechose this function because we hypothesized it would allow the model to easily learn to
    attend by relative positions, since for any fixed offset k,PE_{pos+k} can be represented
    as a linear function of PE_{pos}."

    =========================================================

    Whether to choose fixed or learned positional encodings

    "We also experimented with using learned positional embeddings nstead, and found that the two
    versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version
    because it may allow the model to extrapolate to sequence lengths longer than the ones encountered
    during training."

    In my words:
    Since sinusoidal positional encodings are well-defined, they can easily represent positions longer than
    any of those in training data. When learning the positional encodings, the encodings for these positions
    are just a random mess.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        """
        Tensor pe has shape batch_size x max_len x d_model
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]

        # NOTICE: dropout on embeddings
        return self.dropout(x)


def subsequent_mask_demo():
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0])


def PEncodings_demo():
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))
    plt.plot(np.arange(100), y[0, :, 0:8].data.numpy())
    plt.legend(["dim %d" % p for p in range(0, 8)])


def create_transformer_model(src_vocab_size, tgt_vocab_size, N=6,
                             d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    # d_ff is dimensionality of the inner layer
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        # Sequential secures embedding + positional embedding
        # TODO: but why copy of position is needed?
        nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
        Generator(d_model, tgt_vocab_size))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def model_demo():
    # Small example model.
    tmp_model = create_transformer_model(10, 10, 2)


# L = 3
# D_model = 512
# BSZ = 2
# Q = K = V = torch.randn(BSZ, L, D_model)
# h = 8
#
# modl = MultiHeadedAttention(h = h, d_model=D_model )
# f = modl.forward(Q,K,V)
# fv = modl.forward(Q,K,V)

class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # print("-" * 10 + "SRC" + "-" * 10)
        # print(totext(batch.src, vocab[0]))
        # print("-" * 10 + "TGT" + "-" * 10)
        # print(totext(batch.trg, vocab[1]))
        # print("-" * 30)

        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens.type(torch.float))
        total_loss += loss.item()
        total_tokens += batch.ntokens.item()
        tokens += batch.ntokens.item()
        if i % 50 == 1:
            elapsed = (time.time() - start) + 1e-20
            print(f"Epoch Step: {i} Loss: {loss / batch.ntokens.type(torch.float)} "
                  f"Tokens per Sec: {tokens / elapsed}")
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


# Sentence pairs were batched together by approximate sequence length.
# Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def hyperparam_demo():
    # Three settings of the lrate hyperparameters.
    opts = [NoamOpt(512, 1, 4000, None),
            NoamOpt(512, 1, 8000, None),
            NoamOpt(256, 1, 4000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.show()


class LabelSmoothing(nn.Module):
    """
    Implement label smoothing according to https://arxiv.org/pdf/1512.00567.pdf
    """

    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        """
        :param vocab_size:
        :param padding_idx: index of padding token in vocabulary
        :param smoothing: Amount of probability to be smoothed around vocabulary
        (the bigger this is, the less confident model is and more aggresive smoothing is applied)
        """
        super(LabelSmoothing, self).__init__()
        # if size average is False on loss, losses are summed over minibatch and dimensions
        # otherwise they are averaged
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = vocab_size
        self.true_dist = None

    def forward(self, x, target):
        """
        Label smoothing implemented using the KL div loss. Instead of using a one-hot target distribution,
        we create a distribution that has confidence of the correct word
        and the rest of the smoothing mass distributed throughout the vocabulary.
        :param x: (batch * outlen) x VOCAB_LEN
        :param target:  (batch * outlen)
        :return:
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # smooth rest of the mass (1-confidence) equally through vocabulary
        true_dist.fill_(self.smoothing / (self.size - 2))
        # set confidence mass for correct words
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # mask confidence for padding
        true_dist[:, self.padding_idx] = 0

        # check whether padding idx is in-between targets
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.nelement() > 0:
            # if so, set all those 'probabilities' to 0.0
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        self.true_dist.requires_grad = False
        # return KL div between x and this distribution
        return self.criterion(x, true_dist)


def labelsmoothing_demo1():
    """
    Here we can see an example of how the mass is distributed to the words based on confidence.
    """
    # Example of label smoothing.
    crit = LabelSmoothing(vocab_size=5, padding_idx=0, smoothing=0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = crit(predict.log(),
             torch.LongTensor([2, 1, 0]))

    # Show the target distributions expected by the system.
    plt.imshow(crit.true_dist)
    plt.show()


def labelsmoothing_demo2():
    """
    Label smoothing actually starts to penalize the model
    if it gets very confident about a given choice.
    """
    crit = LabelSmoothing(5, 0, 0.1)

    def loss(x):
        d = x + 3 * 1
        predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                     ])
        # print(predict)
        return crit(predict.log(),
                    torch.LongTensor([1])).item()

    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.show()


class SingleGPULossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        # x has shape batch x outlen x d
        # y has shape batch x outlen

        x = self.generator(x)
        # x has shape batch x outlen x vocab

        # x changed to shape (batch * outlen) x VOCAB_LEN
        # y changed to shape (batch * outlen)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm


def data_gen(V, batch, nbatches, device):
    """
    Synthethic data generator
    "Generate random data for a src-tgt copy task."
    """
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10))).to(device)
        data.requires_grad = False
        data[:, 0] = 1
        src = data
        tgt = torch.cat((data, data), 1)
        yield Batch(src, tgt, 0)


def train_on_synthethic():
    vocab_size = 11  # Vocabulary size
    batch_size = 256

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = LabelSmoothing(vocab_size=vocab_size, padding_idx=0, smoothing=0.0)
    model = create_transformer_model(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, N=3).to(device)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(2):
        model.train()
        run_epoch(data_gen(vocab_size, batch_size, nbatches=20, device=device), model,
                  SingleGPULossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(vocab_size, batch_size, nbatches=5, device=device), model,
                        SingleGPULossCompute(model.generator, criterion, None)))

    model.eval()
    print("Demo evaluation:")
    input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    src = torch.LongTensor([input]).to(device)
    print(f"Input:  {input}")
    src_mask = torch.ones(1, 1, 10).to(device)
    decoded = greedy_decode(model, src, src_mask, max_len=20, start_symbol=1)
    print(f"Output: {list(decoded.cpu().numpy()[0])}")


class DataIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                # Iterate over 100*batch chunks
                # Do the local shuffling over 100 batches
                for p in data.batch(d, self.batch_size * 100):
                    # Sorted by maximum length of src/target sentence
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


def pretrained_IWSLT_demo():
    """
    Demo on  IWSLT German-English Translation task
    """

    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)
    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                              len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
    BATCH_SIZE = 100
    valid_iter = DataIterator(val, batch_size=BATCH_SIZE, device=device,
                              repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn, train=False)

    # using the pre_trained model from https://s3.amazonaws.com/opennmt-models/iwslt.pt
    if not os.path.exists("iwslt.pt"):
        wget.download("https://s3.amazonaws.com/opennmt-models/iwslt.pt")

    model = torch.load("iwslt.pt")

    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        print("Source:", end="\t")
        for i in range(0, src.size(1)):
            sym = SRC.vocab.itos[src[0, i]]
            print(sym, end=" ")
        print()

        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()


def totext(batch, vocab, batch_first=True, remove_specials=False, check_for_zero_vectors=True, pad_token='<blank>',
           eos_token=None, sep=" "):
    textlist = []
    if not batch_first:
        batch = batch.transpose(0, 1)
    for ex in batch:
        if remove_specials:
            textlist.append(
                sep.join(
                    [vocab.itos[ix.item() if hasattr(ix, "item") else ix] for ix in ex
                     if ix != vocab.stoi[pad_token] and eos_token is not None and ix != vocab.stoi["<eos>"]]))
        else:
            if check_for_zero_vectors:
                text = []
                for ix in ex:
                    text.append(vocab.itos[ix.item() if hasattr(ix, "item") else ix])
                textlist.append(sep.join(text))
            else:
                textlist.append(sep.join([vocab.itos[ix.item() if hasattr(ix, "item") else ix] for ix in ex]))
    return textlist


def train_IWSLT():
    """
    Train on  IWSLT German-English Translation task
    """
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)
    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                              len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    pad_idx = TGT.vocab.stoi["<blank>"]
    model = create_transformer_model(len(SRC.vocab), len(TGT.vocab), N=6).to(device)
    criterion = LabelSmoothing(vocab_size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    BATCH_SIZE = 1024
    # These examples are shuffled
    train_iter = DataIterator(train, batch_size=BATCH_SIZE, device=device,
                              repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn, train=True)
    # These examples are not shuffled
    valid_iter = DataIterator(val, batch_size=BATCH_SIZE, device=device,
                              repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn, train=False)

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model,
                  SingleGPULossCompute(model.generator, criterion, model_opt))
        model.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model,
                         SingleGPULossCompute(model.generator, criterion, opt=None))
        print(loss)
        logging.info(f"Validation Loss: {loss}")


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(memory.shape[0], 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           ys,
                           subsequent_mask(ys.size(1))
                           .type_as(src.data))
        log_prob = model.generator(out[:, -1])
        _, next_word = torch.max(log_prob, dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
    return ys[:, 1:]  # do not return start token


class Beam():
    ''' Beam search '''

    def __init__(self, size, pad, bos, eos, device=False):

        self.size = size
        self._done = False
        self.PAD = pad
        self.BOS = bos
        self.EOS = eos
        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        # Initialize to [BOS, PAD, PAD ..., PAD]
        self.next_ys = [torch.full((size,), self.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = self.BOS

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_logprob):
        "Update beam status and check if finished or not."
        num_words = word_logprob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_logprob + self.scores.unsqueeze(1).expand_as(word_logprob)
        else:
            # in initial case,
            beam_lk = word_logprob[0]

        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == self.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


def beam_search(model, src, src_mask, max_len, pad, bos, eos, beam_size, device):
    ''' Translation work in one batch '''

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        ''' Indicate the position of an instance in a tensor. '''
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        # active instances (elements of batch) * beam search size x seq_len x h_dimension
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        # select only parts of tensor which are still active
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(
            src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, beam_size)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        active_src_mask = collect_active_part(src_mask, active_inst_idx, n_prev_active_inst, beam_size)

        return active_src_enc, active_src_mask, active_inst_idx_to_position_map

    def beam_decode_step(
            inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm):
        ''' Decode and update beam status, and then return active beam idx '''

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            # Batch size x Beam size x Dec Seq Len
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            # Batch size*Beam size x Dec Seq Len
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def predict_word(dec_seq, enc_output, n_active_inst, n_bm):
            # print("Encoder output")
            # print(enc_output.shape)
            # print("Src mask")
            # print(src_mask.shape)
            # print("Decoder output")
            # print(dec_seq.shape)
            assert enc_output.shape[0] == dec_seq.shape[0] == src_mask.shape[0]
            out = model.decode(enc_output, src_mask,
                               dec_seq,
                               subsequent_mask(dec_seq.size(1))
                               .type_as(src.data))
            word_logprob = model.generator(out[:, -1])
            # dec_output, *_ = self.model.decoder(dec_seq, dec_pos, src_seq, enc_output)
            # dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
            # word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1)
            word_logprob = word_logprob.view(n_active_inst, n_bm, -1)

            return word_logprob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(
                    word_prob[inst_position])  # Fill Beam object with assigned probabilities
                if not is_inst_complete:  # if top beam ended with eos, we do not add it
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        # get decoding sequence for each beam
        # size: Batch size*Beam size x Dec Seq Len
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)

        # get word probabilities for each beam
        # size: Batch size x Beam size x Vocabulary
        word_logprob = predict_word(dec_seq, enc_output, n_active_inst, n_bm)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(
            inst_dec_beams, word_logprob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    with torch.no_grad():
        # -- Encode
        src_enc = model.encode(src, src_mask)

        #  Repeat data for beam search
        NBEST = beam_size
        batch_size, sent_len, h_dim = src_enc.size()
        src_enc = src_enc.repeat(1, beam_size, 1).view(batch_size * beam_size, sent_len, h_dim)
        src_mask = src_mask.repeat(1, beam_size, 1).view(batch_size * beam_size, 1, src_mask.shape[-1])

        # -- Prepare beams
        inst_dec_beams = [Beam(beam_size, pad, bos, eos, device) for _ in range(batch_size)]

        # -- Bookkeeping for active or not
        active_inst_idx_list = list(range(batch_size))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # -- Decode
        for len_dec_seq in range(1, max_len + 1):

            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_enc, inst_idx_to_position_map, beam_size)

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>
            # filter out inactive tensor parts (for already decoded sequences)
            src_enc, src_mask, inst_idx_to_position_map = collate_active_info(
                src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list)

    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, NBEST)

    return batch_hyp, batch_scores


#####################################
# About IWSLT dataset:

# These are the data sets for the MT tasks of the evaluation campaigns of IWSLT.
# They are parallel data sets used for building and testing MT systems. They are publicly available
# through the WIT3 website wit3.fbk.eu, see release: 2016-01.
# IWSLT 2016: from/to English to/from Arabic, Czech, French, German
# Data are crawled from the TED website and carry the respective licensing conditions (for training, tuning and testing MT systems).

# Approximately, for each language pair, training sets include 2,000 talks, 200K sentences and 4M tokens per side,
# while each dev and test sets 10-15 talks, 1.0K-1.5K sentences and 20K-30K tokens per side. In each edition,
# the training sets of previous editions are re-used and updated with new talks added to the TED repository in the meanwhile.

### Example of data format (tokens are joined via space)
# Source:
# ['Bakterien haben also nur sehr wenige Gene und genetische Informationen um sämtliche Merkmale , die sie ausführen , zu <unk> .',
#  'Die Idee von Krankenhäusern und Kliniken stammt aus den 1780ern . Es wird Zeit , dass wir unser Denken aktualisieren .',
#  'Ein Tier benötigt nur zwei Hundertstel einer Sekunde , um den Geruch zu unterscheiden , es geht also sehr schnell .',
#  'Es stellte sich heraus , dass die Ölkatastrophe eine weißes Thema war , dass <unk> eine vorherrschend schwarzes Thema war .',
#  'Wie ich in meinem Buch schreibe , bin ich genau so jüdisch , wie " Olive Garden " italienisch ist .',
#  'Es gibt einen belüfteten Ziegel den ich letztes Jahr in <unk> machte , als Konzept für New <unk> in Architektur .',
#  'Aber um die Zukunft des Wachstums zu verstehen , müssen wir Vorhersagen über die zugrunde liegenden <unk> des Wachstums machen .',
#  'Ich hatte einen Plan , und ich hätte nie gedacht , wem dabei eine Schlüsselrolle zukommen würde : dem Banjo .',
#  'Im Jahr 2000 hat er entdeckt , dass Ruß wahrscheinlich die zweitgrößte Ursache der globalen Erwärmung ist , nach CO2 .']
#
# Target:
# ['<s> They have very few genes , and genetic information to encode all of the traits that they carry out . </s>',
#  '<s> Humans invented the idea of hospitals and clinics in the 1780s . It is time to update our thinking . </s>',
#  '<s> An animal only needs two hundredths of a second to discriminate the scent , so it goes extremely fast . </s>',
#  '<s> It turns out that oil spill is a mostly white conversation , that cookout is a mostly black conversation . </s>',
#  "<s> As I say in my book , I 'm Jewish in the same way the Olive Garden is Italian . </s>",
#  "<s> There 's an aerated brick I did in <unk> last year , in Concepts for New Ceramics in Architecture . </s>",
#  '<s> but to understand the future of growth , we need to make predictions about the underlying drivers of growth . </s>',
#  '<s> I had a plan , and I never ever thought it would have anything to do with the banjo . </s>',
#  '<s> In 2000 , he discovered that soot was probably the second leading cause of global warming , after CO2 . </s>']


if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath="logs/",
                  config_path="configurations/logging.yml")
    # train_on_synthethic()
    # labelsmoothing_demo1()
    # labelsmoothing_demo2()
    # hyperparam_demo()
    # pretrained_IWSLT_demo()
    train_IWSLT()
