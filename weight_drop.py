import torch
from torch.nn import Parameter
from functools import wraps

class BackHook(torch.nn.Module):
    """Fix for https://github.com/salesforce/awd-lstm-lm/issues/86"""
    def __init__(self, hook):
        super(BackHook, self).__init__()
        self._hook = hook
        self.register_backward_hook(self._backward)

    def forward(self, *inp):
        return inp

    @staticmethod
    def _backward(self, grad_in, grad_out):
        self._hook()
        return None

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()
        # Fix for https://github.com/salesforce/awd-lstm-lm/issues/86
        self.hooker = BackHook(self._backward)

    def _setup(self):
        # Fix for https://github.com/salesforce/awd-lstm-lm/issues/86
        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            self.register_parameter(name_w + '_raw', Parameter(w.data))
            # setattr(self, name_w + '_raw', w.data)

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            if self.training:
                mask = raw_w.new_ones((raw_w.size(0), 1))
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
                setattr(self, name_w + "_mask", mask)
            else:
                w = raw_w
            rnn_w = getattr(self.module, name_w)
            rnn_w.data.copy_(w)

    def _backward(self):
        # transfer gradients from embeddedRNN to raw params
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            rnn_w = getattr(self.module, name_w)
            raw_w.grad = rnn_w.grad * getattr(self, name_w + "_mask")

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

if __name__ == '__main__':
    import torch
    from weight_drop import WeightDrop

    # Input is (seq, batch, input)
    x = torch.autograd.Variable(torch.randn(2, 1, 10)).cuda()
    h0 = None

    ###

    print('Testing WeightDrop')
    print('=-=-=-=-=-=-=-=-=-=')

    ###

    print('Testing WeightDrop with Linear')

    lin = WeightDrop(torch.nn.Linear(10, 10), ['weight'], dropout=0.9)
    lin.cuda()
    run1 = [x.sum() for x in lin(x).data]
    run2 = [x.sum() for x in lin(x).data]

    print('All items should be different')
    print('Run 1:', run1)
    print('Run 2:', run2)

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]

    print('---')

    ###

    print('Testing WeightDrop with LSTM')

    wdrnn = WeightDrop(torch.nn.LSTM(10, 10), ['weight_hh_l0'], dropout=0.9)
    wdrnn.cuda()

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]

    print('First timesteps should be equal, all others should differ')
    print('Run 1:', run1)
    print('Run 2:', run2)

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1]

    print('---')
