import argparse
import logging
import math
import random
import time

import numpy as np
import torch

from common import model_save, model_load
from common import create_criterion, load_embedding, ensure_corpus
import model as mod
from utils import batchify, get_batch, repackage_hidden

from adamw_paper import AdamW
from adam_paper import Adam
from lr_schedule import ParamScheduler, ConstantPhase, LinearPhase, CosinePhase


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer '
                             '(0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN '
                             'hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation '
                             '(alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN '
                             'activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume')
    parser.add_argument('--optimizer', type=str,  default='adam',
                        choices=['adam', 'adamw'],
                        help='optimizer to use (adam, adamw)')
    # parser.add_argument('--when', nargs="+", type=int, default=[-1],
    #                     help='When (which epochs) to divide the learning rate '
    #                          'by 10 - accepts multiple')
    parser.add_argument('--from-embedding',
                        help='initialize the embedding (and softmax) weights '
                             'from those of an already existing model')
    parser.add_argument('--log-level', '-L', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    args = parser.parse_args()
    args.tied = True
    return args

###############################################################################
# Training functions
###############################################################################

def evaluate(model, data_source, args, criterion, batch_size=10):
    """
    Evaluates on the specified data (typically the eval / test sets).

    :param model: the RNN model.
    :param data_source: data_source the batch. Output of :func:`utils.batchify`.
    :param args: args the command-line arguments. Ugly, but oh well.
    :param criterion: the criterion to evaluate the data with.
    :param batch_size: the batch size.
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(output, targets).data
        # ASM total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train(model, data_source, args, criterion, optimizer, params, epoch, ps):
    """
    Runs a training epoch.

    :param model: the RNN model to train.
    :param data_source: data_source the batch. Output of :func:`utils.batchify`.
    :param args: args the command-line arguments. Ugly, but oh well.
    :param criterion: the criterion to train against.
    :param optimizer: the optimizer.
    :param params: the model parameters. Needed for gradient clipping.
    :param epoch: the current epoch. For printing the results.
    :param ps: the parameter scheduler the governs the learning rate
    """
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < data_source.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        epoch_decimal = batch / (len(data_source) // args.bptt)
        ps.update(epoch_decimal)
        logging.debug('epoch_decimal: {}'.format(epoch_decimal))
        logging.debug('LR: {}, betas: {}'.format(optimizer.param_groups[0]['lr'],
                                                 optimizer.param_groups[0]['betas']))

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(data_source, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(output, targets)
        # ASM raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha:
            loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean()
                              for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta:
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                              for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            # try-except for debugging purposes only
            try:
                logging.info(
                    '| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
                    'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                        epoch, batch, len(data_source) // args.bptt,
                        optimizer.param_groups[0]['lr'],
                        elapsed * 1000 / args.log_interval, cur_loss,
                        math.exp(cur_loss), cur_loss / math.log(2))
                )
            except OverflowError:
                logging.exception('cur_loss = {} @ batch {}'.format(cur_loss, i))
                raise
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s'
    )

    model, criterion, optimizer = None, None, None

    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            logging.warning('WARNING: You have a CUDA device, so you should '
                            'probably run with --cuda')
        else:
            torch.cuda.manual_seed(args.seed)

    # Load corpus + create batches
    corpus = ensure_corpus(args.data)
    eval_batch_size = 10
    test_batch_size = 1
    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    ###########################################################################
    # Build the model
    ###########################################################################

    ntokens = len(corpus.dictionary)
    model = mod.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                         args.nlayers, args.dropout, args.dropouth,
                         args.dropouti, args.dropoute, args.wdrop, args.tied)
    ###
    if args.resume:
        logging.info('Resuming model ...')
        model, criterion, optimizer = model_load(args.resume)
        optimizer.param_groups[0]['lr'] = args.lr
        model.dropouti, model.dropouth, model.dropout, args.dropoute = (
            args.dropouti, args.dropouth, args.dropout, args.dropoute)
        if args.wdrop:
            from weight_drop import WeightDrop
            for rnn in model.rnns:
                if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
                elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
    ###
    if not criterion:
        criterion = create_criterion()

    ### Load the embedding, if required
    if args.from_embedding:
        logging.info('Loading embedding from {}'.format(args.from_embedding))
        load_embedding(args.from_embedding, model)

    orig_emb = model.encoder.weight.detach()
    assert ((orig_emb - model.encoder.weight).norm(2) == 0).all()
    assert ((orig_emb - model.decoder.weight).norm(2) == 0).all()
    del orig_emb  # TODO: test

    ###
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    ###
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    logging.debug('Args: {}'.format(args))
    logging.debug('Model total parameters: {}'.format(total_params))

    ###########################################################################
    # Training code
    ###########################################################################

    # Loop over epochs.
    best_val_loss = []
    stored_loss = 100000000

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        opt_class = Adam if args.optimizer == 'adam' else AdamW
        optimizer = opt_class(params, lr=args.lr, weight_decay=args.wdecay,
                              betas=(0.9, 0.99))
        min_mom, max_mom = 0.7, 0.8
        ps = ParamScheduler(
            optimizer,
            {
                'lr': [
                    LinearPhase(7.5, args.lr / 10, args.lr),
                    ConstantPhase(37.5, args.lr),
                    LinearPhase(37.5, args.lr, args.lr / 10),
                    CosinePhase(7.5, args.lr / 10, args.lr / 1000)
                ],
                ('betas', 0): [
                    LinearPhase(7.5, max_mom, min_mom),
                    ConstantPhase(37.5, min_mom),
                    LinearPhase(37.5, min_mom, max_mom),
                    ConstantPhase(7.5, max_mom)
                ]
            }
        )
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            ps.new_epoch()  # AdamW
            train(train_data, args, criterion, optimizer, params, epoch, ps)

            val_loss = evaluate(val_data, eval_batch_size)
            logging.info('-' * 89)
            logging.info(
                '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss,
                    math.exp(val_loss), val_loss / math.log(2))
            )
            logging.info('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save, model, criterion, optimizer)
                logging.info('Saving model (new best validation)')
                stored_loss = val_loss

            # if epoch in args.when:
            #     logging.info('Saving model before learning rate decreased')
            #     model_save('{}.e{}'.format(args.save, epoch))
            #     logging.info('Dividing learning rate by 10')
            #     optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early')

    # Load the best saved model.
    model, criterion, optimizer = model_load(args.save)

    # Run on test data.
    test_loss = evaluate(test_data, test_batch_size)
    logging.info('=' * 89)
    logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f} | '
                 'test bpc {:8.3f}'.format(test_loss, math.exp(test_loss),
                                           test_loss / math.log(2)))
    logging.info('=' * 89)


if __name__ == '__main__':
    main()
