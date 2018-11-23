import copy
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x, mask):
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
        self.heads = heads
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
        query_, key_, value_ = \
            [l(x).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # Rewritten into more clear representation
        query = self.linears[0](query).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
        key = self.linears[1](key).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
        value = self.linears[2](value).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)

        assert torch.equal(query, query_) and torch.equal(key, key_) and torch.equal(value, value_)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=mask,
                                                    dropout=self.dropout)
        # x has shape bsz x length x d_model
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.heads * self.d_k)
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
        self.embedder = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedder(x) * math.sqrt(self.d_model)


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


def make_model(src_vocab_size, tgt_vocab_size, N=6,
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
            nn.init.xavier_uniform(p)
    return model


def model_demo():
    # Small example model.
    tmp_model = make_model(10, 10, 2)


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


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def hyperparam_demo():
    # Three settings of the lrate hyperparameters.
    opts = [NoamOpt(512, 1, 4000, None),
            NoamOpt(512, 1, 8000, None),
            NoamOpt(256, 1, 4000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        """
        We implement label smoothing using the KL div loss.
        Instead of using a one-hot target distribution,
        we create a distribution that has confidence of the correct word and
        the rest of the smoothing mass distributed throughout the vocabulary.
        :param vocab_size:
        :param padding_idx:
        :param smoothing:
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
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.nelement() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        self.true_dist.requires_grad = False
        return self.criterion(x, true_dist)


def labelsmoothing_demo1():
    """
    Here we can see an example of how the mass is distributed to the words based on confidence.
    """
    # Example of label smoothing.
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = crit(predict.log(),
             torch.LongTensor([2, 1, 0]))

    # Show the target distributions expected by the system.
    plt.imshow(crit.true_dist)


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
                    torch.LongTensor([1])).data[0]

    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        # x has shape batch x outlen x d
        # y has shape bath x outlen

        x = self.generator(x)
        # x has shape batch x outlen x vocab

        # x has shape x VOCAB_LEN
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           ys,
                           subsequent_mask(ys.size(1))
                           .type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


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
    model = make_model(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, N=3).to(device)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(vocab_size, batch_size, nbatches=20, device=device), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(vocab_size, batch_size, nbatches=5, device=device), model,
                        SimpleLossCompute(model.generator, criterion, None)))

    model.eval()
    print("Demo evaluation:")
    input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    src = torch.LongTensor([input]).to(device)
    print(f"Input:  {input}")
    src_mask = torch.ones(1, 1, 10).to(device)
    decoded = greedy_decode(model, src, src_mask, max_len=20, start_symbol=1)
    print(f"Output: {list(decoded.cpu().numpy()[0])}")


from torchtext import data


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
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


def train_on_real():
    from torchtext import data, datasets

    if True:
        import spacy
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

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
        valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=False)

        # use pre_trained model
        # wget https://s3.amazonaws.com/opennmt-models/iwslt.pt
        model = torch.load("iwslt.pt")

        for i, batch in enumerate(valid_iter):
            src = batch.src.transpose(0, 1)[:1]
            src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask,
                                max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
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
            break


if __name__ == "__main__":
    train_on_synthethic()
    
    # train on real is broken for now :(
    # train_on_real()
