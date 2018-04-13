#!/usr/bin/env python

import argparse
from spacyClass import spacyClass
from magicClass import magicClass
from magicSlangClass import magicSlangClass
import time

def main():
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('--engine', type=str, default="spacy")
    argparser.add_argument('--out', type=str, default='models/patterns_spacy.pkl')
    args = argparser.parse_args()

    classes = {
        'spacy': spacyClass,
        'magic': magicClass,
        'magicSlang': magicSlangClass
    }

    if args.engine not in classes:
        raise ValueError('Sorry, this engine is not supported.')

    training_start_time = time.time()
    instance = classes[args.engine]()
    instance.train(train_file=args.filename, outfile = args.out)
    print ('Training Complete in', time.time()-training_start_time, 'seconds.')
if __name__ == '__main__':
    main()
