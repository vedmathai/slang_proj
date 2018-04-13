from engine_base_class import EngineBaseClass
import json
import pickle
from utils import Vectors
import numpy as np
import re
from scipy.spatial.distance import cosine

class magicClass(EngineBaseClass):
    def __init__(self):
        self.vec = Vectors()
        self.model = None

    def __save_file(self, module, outdir):
        outfile = outdir
        print ('Saving file trained pickle file to ', outfile)
        picklefile = open(outfile, 'wb')
        pickle.dump(module, picklefile)

    def __check_train_file_correctness(self, model):
        keylist = set()
        for dic in model:
            keylist |= set(list(dic.keys()))
        if len(keylist - set(['sentence', 'annotations', 'intent']))>0:
            return False
        else:
            return True

    def __train_logic(self, train_module):
        bagOfWords = {}
        entities = {'None': []}
        annotatedItems = set()
        for items in train_module:
            for annotation in items['annotations']:
                if annotation[0] not in entities:
                    entities[annotation[0]] = []
                for split_word in annotation[1].lower().split():
                    entities[annotation[0]]+=[split_word]
                    annotatedItems.add(split_word)
            for word in set(items['sentence'].lower().split()) - annotatedItems:
                entities['None']+=[word]
            if items['intent'] not in bagOfWords:
                bagOfWords[items['intent']]=set()
            bagOfWords[items['intent']] |= set(items['sentence'].lower().split())
        return entities, bagOfWords

    def load_model(self, model_file='models/patterns_magic.pkl'):
        model = pickle.load(open(model_file, 'rb'))
        self.model = model

    def train(self, train_file=None, outfile = 'models/patterns_magic.pkl'):
        if train_file == None:
            raise ValueError('The train file cannot be undefined')
        trainingFile = open(train_file, 'rt')
        json_ent = json.load(trainingFile)
        if self.__check_train_file_correctness(json_ent) == False:
            raise ValueError('There is some problem with the train file')
        model = self.__train_logic(json_ent)
        self.__save_file(model, outfile)

    def processWord(self, word):
        return re.sub('[^A-Za-z0-9]+', '', word)

    def __create_json(self, sentence, finalWordLists, finalIntentScores):
        ret = {}
        ret['sentence'] = sentence
        ret['entitylist'] = []
        for word in finalWordLists:
            ret['entitylist']+=[[word[0], word[1][0][0]]]
        ret['intent'] = finalIntentScores[0][0]
        return ret

    def parse(self, sentence):
        vec = self.vec
        if self.model == None:
            raise ValueError('The model is null, please load a model first.')
        entities, bagOfWords = self.model
        original_sentence = sentence
        sentence = sentence.split()
        finalWordLists = []
        intentsScores = {}
        for word in sentence:
            #named entity recognition
            entityScores = {}
            word = self.processWord(word)
            for entity in entities:
                if entity not in entityScores:
                    entityScores[entity]=[]
                for item in entities[entity]:
                    if vec.wordExists(item.lower())==True:
                        entityScores[entity]+= [1-cosine(vec.getVec(item.lower()), vec.getVec(word))]
                cats_list = []
            for entity in entityScores:
                cats_list += [(entity, np.mean(entityScores[entity]))]
            sorted_cats_list = sorted(cats_list, key = lambda x: x[1], reverse = True)
            finalWordLists += [(word, sorted_cats_list)]

            #start of intent recognition
            for intent in bagOfWords:
                if intent not in intentsScores:
                    intentsScores[intent] = []
                for intentWord in bagOfWords[intent]:
                    if vec.wordExists(intentWord.lower())==True:
                        intentsScores[intent]+=[1-cosine(vec.getVec(intentWord.lower()), vec.getVec(word))]
        finalIntentScores = []
        for intent in intentsScores:
            finalIntentScores += [(intent, np.mean(intentsScores[intent]))]
        finalIntentScores = sorted(finalIntentScores, key = lambda x: x[1], reverse = True)

        #cleaning up
        ret = self.__create_json(original_sentence, finalWordLists, finalIntentScores)
        return ret

if __name__=='__main__':
    print ('Unit testing this class...')
    magicinstance = magicClass()
    magicinstance.train('trainfiles/all.json')
    magicinstance.load_model()
    print (magicinstance.parse('What is the weather in Chennai today?'))
