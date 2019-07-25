#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Functions common to all main*py scripts."""

import hashlib
import logging
import os

import torch

import data
# ASM from splitcross import SplitCrossEntropyLoss


def create_criterion(emsize=None):
    """Returns the criterion object. Refactored to a separate function."""
    # ASM
    # splits = []
    # if ntokens > 500000:
    #     # One Billion
    #     # This produces fairly even matrix mults for the buckets:
    #     # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
    #     splits = [4200, 35000, 180000]
    # elif ntokens > 75000:
    #     # WikiText-103
    #     splits = [2800, 20000, 76000]
    # logging.info('Using splits {}'.format(splits))
    # criterion = SplitCrossEntropyLoss(emsize, splits=splits, verbose=False)
    criterion = torch.nn.CrossEntropyLoss()
    return criterion


###############################################################################
# Load data
###############################################################################

def model_save(fn, model, criterion, optimizer):
    """Saves the model, criterion and optimizer into *fn*."""
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    """Loads the model, criterion and optimizer from *fn*."""
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)
    return model, criterion, optimizer


def load_embedding(fn, model, requires_grad=False):
    """Loads an embedding from *fn* and replaces the model embedding with it."""
    with open(fn, 'rb') as f:
        embedding_model, _, _ = torch.load(f)
        model.encoder.weight = embedding_model.encoder.weight
        model.decoder.weight = embedding_model.decoder.weight
        model.encoder.weight.requires_grad = requires_grad
        model.decoder.weight.requires_grad = requires_grad


def ensure_corpus(data_file):
    """Ensures that the cached corpus file exists."""
    fn = 'corpus.{}.data'.format(hashlib.md5(data_file.encode()).hexdigest())
    if os.path.exists(fn):
        logging.info('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        logging.info('Producing dataset...')
        corpus = data.Corpus(data_file)
        torch.save(corpus, fn)
    return corpus
