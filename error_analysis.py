#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Loads a model and computes various statistics: the PPL for each individual word,
as well as the top N candidates, the rank of the real word, etc.
"""

import argparse
import logging
from math import exp

import torch
import torch.nn.functional as F

from common import ensure_corpus, model_load
from utils import batchify, get_batch, repackage_hidden


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', '-d', type=str, default='data/penn',
                        help='location of the data corpus (files called '
                             'train|valid|test.txt).')
    parser.add_argument('--file', '-f', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='which file to load. Default: test.')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='the model file (--save in main.py).')
    parser.add_argument('--no-cuda', '-c', dest='cuda', action='store_false',
                        help='do not use CUDA')
    parser.add_argument('--batch', '-b', type=int, dest='batch_size', default=1,
                        help='the batch size. Default is 1.')
    parser.add_argument('--steps', '-s', type=int, dest='bptt', default=70,
                        help='the number of timesteps. Default is 70.')
    parser.add_argument('--log-level', '-L', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    return parser.parse_args()


def evaluate(model, data_source, corpus, criterion, args, batch_size=1):
    """
    Evaluates on the specified data (typically the eval / test sets) and
    collects the statistics.

    :param model: the RNN model.
    :param data_source: data_source the batch. Output of :func:`utils.batchify`.
    :param args: args the command-line arguments. Ugly, but oh well.
    :param criterion: the criterion to evaluate the data with.
    :param batch_size: the batch size.
    """
    assert batch_size == 1, 'batch sizes other than 1 are not supported yet'
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if hasattr(model, 'reset'): model.reset()  # QRNN
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    context = [[] for _ in range(batch_size)]

    print('target_word', 'context', 'loss', 'perplexity', 'entropy',
          'target_index', 'target_p', 'predicted_p', 'most_probable',
          sep='\t')
    for i in range(0, data_source.size(0) - 1, args.bptt):
        # data.size() = bptt x bs
        # targets.size() = bptt
        data, targets = get_batch(data_source, i, args, evaluation=True)
        bptt = targets.size()[0]
        # output.size() = bptt x |V| -- this is the default
        output, hidden = model(data, hidden)
        # TODO: mondatkezdo
        sorted_logits, most_probable = torch.sort(output, dim=1, descending=True)
        eq_target = (most_probable == targets.unsqueeze(1))
        # idx = list(most_probable.data[1]).index(targets.data[1])
        # logging.info(f'mp: {idx} => {most_probable.data[1, idx]}')
        # logging.info(f'eq_target: {eq_target[1, idx-5:idx+5]} out of {sum(list(eq_target[1].data))}')

        # indices.size() = bptt ; nonzero()'s first column is the index (0 .. bptt - 1)
        indices = eq_target.nonzero()[:, 1]
        # for i in range(args.bptt):
        #     logging.info(f'i: {i}')
        #     word_id = targets.data[i]
        #     idx = list(most_probable.data[i]).index(word_id)
        #     assert eq_target[i, idx].data[0] == 1
        #     assert i == nnz[i].data[0]
        #     assert idx == nnz[i].data[1]
        # assert eq_target.data.sum() == args.bptt
        # indices = targets.index_put((nnz[:, 0], nnz[:, 1]), nnz[:, 2])
        # losses.size() = bptt
        losses = criterion(output, targets)
        # losses = criterion(output, targets).data.view(batch_size, args.bptt)
        # loss = criterion(output, targets).data
        # probabilities.size() = bptt x |V|, each row sums to 1
        probabilities = F.softmax(sorted_logits, dim=1)
        entropy = (-probabilities * F.log_softmax(sorted_logits, dim=1)).sum(dim=1)
        # coordx, coordy = torch.meshgrid([torch.arange(batch_size), torch.arange(args.bptt)])
        # target_probs = probabilities[coordx, coordy, indices]
        # predicted_probs = probabilities[:, :, 0]
        total_loss += torch.sum(losses)[0]
        hidden = repackage_hidden(hidden)

        # TODO: not necessarily cuda()
        # target_probs.size() = bptt
        target_probs = probabilities[torch.arange(bptt).cuda().long(), indices]

        # predicted_probs.size() = bptt
        predicted_probs = probabilities[torch.arange(bptt).cuda().long(),
                                       torch.zeros(bptt).cuda().long()]

        for i in range(data.size(0)):
            context[0] = (context[0] + [corpus.dictionary.idx2word[data[i, 0].data[0]]])[-10:]
            # word, context, loss, perplexity, entropy of the distribution,
            # index of the target word, probability of target word,
            # probability of predicted word, most probable words
            print(corpus.dictionary.idx2word[targets[i].data[0]],  # word
                  ' '.join(context[0]),  # context
                  losses[i].data[0],  # [0]?  # loss
                  exp(losses[i]),  # perplexity
                  entropy[i].data[0],  # entropy of the distribution
                  indices[i].data[0],  # index of the target word
                  target_probs[i].data[0],  # probability of target word
                  predicted_probs[i].data[0],  # probability of predicted word
                  ' '.join(corpus.dictionary.idx2word[w.data[0]]
                           for w in most_probable[i, :5]),
                  sep='\t')
    return total_loss / batch_size / len(data_source)


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s'
    )
    logging.info(f'Arguments: {args}')

    corpus = ensure_corpus(args.data)

    eval_batch_size = 1

    test_data = batchify(corpus.test, eval_batch_size, args)
    model, _, _ = model_load(args.model)
    criterion = torch.nn.CrossEntropyLoss(reduce=False)
    logging.info('Running evaluation...')
    evaluate(model, test_data, corpus, criterion, args, eval_batch_size)
    logging.info('Done.')


if __name__ == '__main__':
    main()
