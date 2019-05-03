#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Loads a model and computes various statistics: the PPL for each individual word,
as well as the top N candidates, the rank of the real word, etc.
"""

import argparse
import hashlib
import logging
import os

import torch
import torch.nn as nn

import data

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


def model_load(fn):
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)
    return model, criterion


def evaluate(model, data_source, criterion, args, batch_size=10):
    """Runs the evaluation and collects the statistics."""
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if hasattr(model, 'reset'): model.reset()  # QRNN
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    context = [[] for _ in range(batch_size)]
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        loss = criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        total_loss += len(data) * loss
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s'
    )

    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        logging.info('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        logging.info('Caching dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)

    eval_batch_size = 1

    test_data = batchify(corpus.test, eval_batch_size, args)
    model, criterion = model_load(args.model)
    evaluate(model, test_data, criterion, args, eval_batch_size)


if __name__ == '__main__':
    main()
