import math

import torch
import torch.nn.functional as F

from playground import clones


class RNN(torch.nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.hidden_states = self.rnn = None

    def init_hidden(self, batch_size, directions, initfun=torch.randn, batch_wise_params=False, requires_grad=True):
        """
        Init RNN hidden state
        :param batch_wise_params: If true, hidden state will be different for each batch element, this introduces non-determinism
        :param batch_size: size of batch, note that for each batch sample, the hidden state may be and usually is
                            initialized differently, in case of example being present in another batch dimension
                            the model can be "Non-deterministic"
        :param directions: number of directions
        :param initfun: function to initialize hidden state from,
                default: torch.randn, which provides samples from normal gaussian distribution (0 mean, 1 variance)
        :param requires_grad: if the hidden states should be learnable, default = True

        Initializes variable self.hidden
        """

        if batch_wise_params:
            self.hidden_params = torch.nn.Parameter(
                initfun(self.layers * directions, batch_size, self.hidden_size,
                        requires_grad=requires_grad))

            self.cell_params = torch.nn.Parameter(
                initfun(self.layers * directions, batch_size, self.hidden_size, requires_grad=requires_grad))
        else:
            self.hidden_params = torch.nn.Parameter(
                initfun(self.layers * directions, 1, self.hidden_size, requires_grad=requires_grad)
                # .expand(-1, batch_size, -1)
            )
            self.cell_params = torch.nn.Parameter(
                initfun(self.layers * directions, 1, self.hidden_size, requires_grad=requires_grad))
            # .expand(-1, batch_size, -1))

            self.hidden_states = (self.hidden_params, self.cell_params)

    def forward(self, inp):
        """
        :param inp: Shape BATCH_SIZE x LEN x H_DIM
        """
        assert self.rnn
        bsz = inp.shape[0]
        # This needs to be done in order to resize hidden vector
        # for the last batch
        hidden_params = self.hidden_params.repeat(1, bsz, 1)
        cell_params = self.cell_params.repeat(1, bsz, 1)
        # hidden = (hidden[0][:, :bsz, :].contiguous(), hidden[1][:, :bsz, :].contiguous())
        outp = self.rnn(inp, (hidden_params, cell_params))[0]
        return outp


class BiLSTM(RNN):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = int(config['RNN_nhidden'])
        self.layers = int(
            config['RNN_layers'])
        self.rnn = torch.nn.LSTM(
            int(config["embedding_dim"]),
            self.hidden_size, self.layers,
            dropout=float(config['RNN_dropout']),
            batch_first=True,
            bidirectional=True)
        self.init_hidden(int(config["batch_size"]), directions=2)


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, config, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        d_model = int(config['d_model'])
        heads = int(config['heads'])
        assert d_model % heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // heads  # dimensionality over one head
        self.heads = heads
        # 4 - for query, key, value transformation + output transformation
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, input, mask=None):
        return self.attention(input, input, input, mask)

    def attention(self, query, key, value, mask=None):
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

        # Rewritten into more clear representation
        query = self.linears[0](query).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
        key = self.linears[1](key).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
        value = self.linears[2](value).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.scaled_dot_product_attention(query, key, value, mask=mask,
                                                         dropout=self.dropout)
        # x has shape bsz x length x d_model
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.heads * self.d_k)
        return self.linears[-1](x)

    def scaled_dot_product_attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
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
