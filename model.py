import torch
import torch.nn as nn
from torch.autograd import Variable

from weight_drop import WeightDrop
from pytorch_lm.utils.config import create_object
from pytorch_lm.dropout import create_dropout

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0,
                 tie_weights=False, alpha=0, beta=0):
        super(RNNModel, self).__init__()
        self.in_do = create_dropout('{}s'.format(dropouti))
        self.lay_do = nn.ModuleList(
            [create_dropout('{}s'.format(dropouth)) for _ in range(nlayers - 1)])
        self.out_do = create_dropout('{}s'.format(dropout))

        self.encoder = nn.Embedding(ntoken, ninp)
        rnn = {'class': 'PytorchLstmLayer'}
        self.rnns = nn.ModuleList()
        for l in range(nlayers):
            in_size = ninp if not l else nhid
            out_size = nhid if l + 1 != nlayers or not tie_weights else ninp
            self.rnns.append(
                create_object(rnn, base_module='pytorch_lm.rnn',
                              args=[in_size, out_size])
            )
        # Should be this instead, but it fucks up the sequence of the random
        # number generator, so we wouldn't get the same numbers... -- but maybe
        # better? TODO
        # self.decoder = nn.Linear(ninp if tie_weights else nhid, ntoken)
        self.decoder = nn.Linear(nhid, ntoken)

        if wdrop:
            self.rnns = nn.ModuleList(
                [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop)
                 for rnn in self.rnns]
            )

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

    def embedded_dropout(self, words):
        """
        Another (less memory-consuming) way to compute embedding dropout.
        Note that just like the original AWD-LSTM (Merity et al., 2018), but as
        opposed to the RHN implementation in Zilly et al. (2017), the same
        words are masked in all sequences of the batch.
        """
        emb = self.encoder(words)
        if self.training and self.dropoute:
            weight = self.encoder.weight
            dropout = self.dropoute
            mask_ = weight.data.new_ones((weight.size(0), 1))
            mask = nn.functional.dropout(mask_, dropout, self.training)
            m = torch.gather(mask.expand(weight.size(0), words.size(1)), 0, words)
            emb = (emb * m.unsqueeze(2).expand_as(emb)).squeeze()
        return emb

    def forward(self, input, hidden):
        emb = self.embedded_dropout(input)
        emb = self.in_do(emb)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lay_do[l](raw_output)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.out_do(raw_output)
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
