#!/usr/bin/env python

import argparse
from spacyClass import spacyClass
from magicClass import magicClass
from spacySlangClass import spacySlangClass
from magicSlangClass import magicSlangClass

import time

def main():
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('sentence', type=str)
    argparser.add_argument('--engine', type=str, default="spacy")
    argparser.add_argument('--model', type=str, default='model.pkl')
    args = argparser.parse_args()

    classes = {
        'spacy': spacyClass,
        'magic': magicClass,
        'spacySlang': spacySlangClass,
        'magicSlang': magicSlangClass
    }
    if args.engine not in classes:
        raise ValueError('Sorry, this engine is not supported.')

    training_start_time = time.time()
    instance = classes[args.engine]()
    instance.load_model(args.model)
    print (instance.parse(sentence = args.sentence))
    print ('Inference Complete in', time.time()-training_start_time, 'seconds.')

if __name__ == '__main__':
    main()
