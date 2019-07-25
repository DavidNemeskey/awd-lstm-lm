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


def evaluate(model, data_source, corpus, criterion, args, batch_size=10):
    """
    Evaluates on the specified data (typically the eval / test sets) and
    collects the statistics.

    :param model: the RNN model.
    :param data_source: data_source the batch. Output of :func:`utils.batchify`.
    :param args: args the command-line arguments. Ugly, but oh well.
    :param criterion: the criterion to evaluate the data with.
    :param batch_size: the batch size.
    """
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
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        # TODO: mondatkezdo
        # Let's hope it's dim=2
        sorted_logits, most_probable = torch.sort(output, dim=2, descending=True)
        eq_target = (most_probable == targets.unsqueeze(2))
        nnz = eq_target.nonzero()
        indices = targets.index_put((nnz[:, 0], nnz[:, 1]), nnz[:, 2])
        losses = criterion(output, targets).data.view(batch_size, args.bptt)
        # loss = criterion(output, targets).data
        probabilities = F.softmax(sorted_logits, dim=2)
        coordx, coordy = torch.meshgrid([torch.arange(batch_size), torch.arange(args.bptt)])
        target_probs = probabilities[coordx, coordy, indices]
        predicted_probs = probabilities[:, :, 0]
        entropy = (-probabilities * F.log_softmax(sorted_logits, dim=2)).sum(2)
        total_loss += torch.sum(losses)[0]
        hidden = repackage_hidden(hidden)
        for i in range(data.size(0)):
            context[i] = (context[i] + [corpus.dictionary.idx2word[data[i, 0]]])[-10:]
            # word, context, loss, perplexity, entropy of the distribution,
            # index of the target word, probability of target word,
            # probability of predicted word, most probable words
            print(corpus.dictionary.idx2word[targets[i, 0]],
                  ' '.join(context[i]),
                  losses[i, 0].item(),
                  exp(losses[i, 0]),
                  entropy[i, 0].item(),
                  indices[i, 0].item(),
                  target_probs[i, 0].item(),
                  predicted_probs[i, 0].item(),
                  ' '.join(corpus.dictionary.idx2word[w] for w in most_probable[i, 0, :5]),
                  sep='\t')
    return total_loss / batch_size / len(data_source)


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s'
    )

    corpus = ensure_corpus(args.data)

    eval_batch_size = 1

    test_data = batchify(corpus.test, eval_batch_size, args)
    model, criterion, _ = model_load(args.model)
    evaluate(model, test_data, corpus, criterion, args, eval_batch_size)


if __name__ == '__main__':
    main()
