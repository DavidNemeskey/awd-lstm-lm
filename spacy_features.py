#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Loads the feature file written by error_analysis.py, runs spaCy on it, and
extracts a number of linguistic features.

Note that this assumes that the tokens in the .tsv file are in sorted order. In
other words, use a batch size of 1 when running error_analysis.py.

Make sure the packages in requirements_analysis.txt are installed before running
this script.
TODO: what features?
"""

import argparse
import logging
import re

import pandas as pd
import spacy


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='the tsv file created by error_analysis.py.')
    parser.add_argument('--output-file', '-o', type=str, default=None,
                        help='the tsv file, augmented with linguistic '
                             'features. If not specified, the input file is '
                             'overwritten by this script.')
    parser.add_argument('--model', '-m', type=str, default='en',
                        help='the spaCy model to use. It must have already '
                             'been downloaded prior to running this script. '
                             'The default is "en", which is not necessarily '
                             'be available.')
    parser.add_argument('--log-level', '-L', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = args.input_file
    return args


class WhitespaceTokenizer:
    """
    Whitespace tokenizer for spaCy. Since we start with a pre-tokenized
    corpus, we need to keep stay in synch with it.
    """
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        return spacy.tokens.doc.Doc(self.vocab, text.split())


def parse_input(data: pd.DataFrame, model: str):
    text = re.sub(r'\s*<eos>\s*', '\n', ' '.join(data['target_word']))
    nlp = spacy.load(model)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    doc = nlp(text)
    columns = ['POS', 'dep_link', 'dep_head', 'shape', 'alpha', 'stop', 'depth']
    new_data = []
    for token in doc:
        new_data.append([token.text, token.tag_, token.dep_,
                         token.head.i - token.i, token.shape, token.is_alpha,
                         token.is_stop, len(token.ancestors)])
    for col, series in zip(columns, zip(*new_data)):
        data[col] = series
    return data


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s'
    )
    logging.info(f'Arguments: {args}')

    data = pd.read_csv(args.input_file, delimiter=r'\t')
    new_data = parse_input(data, args.model)
    new_data.to_csv(args.output_file, sep='\t', index=False, mode='wt')


if __name__ == '__main__':
    main()
