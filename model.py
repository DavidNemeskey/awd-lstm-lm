import torch
import torch.nn as nn
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
from pytorch_lm.rnn.lstm import PytorchLstmLayer

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0,
                 tie_weights=False, alpha=0, beta=0):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        # assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        # if rnn_type == 'LSTM':
        self.rnns = [PytorchLstmLayer(
            ninp if l == 0 else nhid,
            nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
            1, dropout=0
        ) for l in range(nlayers)]
        if wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

        self.alpha = alpha
        self.beta = beta
        self.loss_reg = 0

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        # Not needed for ASM
        result = self.decoder(result)

        self.loss_reg = 0
        raw_output = raw_outputs[-1]

        alpha_stuff = beta_stuff = 0
        if self.beta:
            beta_stuff = self.beta * (
                raw_output[1:] - raw_output[:-1]).pow(2).mean()
        if self.alpha:
            alpha_stuff = self.alpha * output.pow(2).mean()
        self.loss_reg += alpha_stuff + beta_stuff

        return result, hidden

    def loss_regularizer(self):
        return self.loss_reg

    def init_hidden(self, batch_size):
        return [rnn.init_hidden(batch_size) for rnn in self.rnns]
