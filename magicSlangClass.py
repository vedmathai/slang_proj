from engine_base_class import EngineBaseClass
import json
import pickle
from utils import Vectors
import numpy as np
import re
from scipy.spatial.distance import cosine

class magicSlangClass(EngineBaseClass):
    def __init__(self):
        self.vec = Vectors()
        self.model = None

    def __save_file(self, module, outdir):
        outfile = outdir
        print ('Saving file trained pickle file to ', outfile)
        picklefile = open(outfile, 'wb')
        pickle.dump(module, picklefile)

    def __check_train_file_correctness(self, model):
            return True

    def __train_logic(self, train_module):
        entities = {'None':set()}
        json_ent = train_module
        for key in json_ent['types']:
            if key['name'] not in entities:
                entities[key['name']] = set()
            for act in ['samples', 'values', 'entries']:

                if act in key:
                    if type(key[act][0]) == type({}):
                        entities[key['name']] |= set([dic['identity'] for dic in key[act]])
                    else:
                        entities[key['name']] |= set(key[act])
        intents = {}
        for key in json_ent['intents']:

            for entity in key['entities']:
                entities[entity['name']] = entities[entity['type']]

            if key['name'] not in intents:
                intents[key['name']] = set()
            for example in key['examples']:
                sentence = ' '.join(part['text'] for part in example if part['entity']==None)
                for part in example:
                    if part['entity'] == None:
                        entities['None']|=set(part['text'].lower().split())
                intents[key['name']]|=set(sentence.lower().split())
        return (intents, entities)

    def load_model(self, model_file='models/patterns_slang_magic.pkl'):
        model = pickle.load(open(model_file, 'rb'))
        self.model = model

    def train(self, train_file=None, outfile = 'models/patterns_slang_magic.pkl'):
        if train_file == None:
            raise ValueError('The train file cannot be undefined')
        trainingFile = open(train_file, 'rt')
        json_ent = json.load(trainingFile)
        if self.__check_train_file_correctness(json_ent) == False:
            raise ValueError('There is some problem with the train file')
        entities = self.__train_logic(json_ent)
        self.__save_file(entities, outfile)

    def processWord(self, word):
        return re.sub('[^A-Za-z0-9]+', '', word)

    def __create_json(self, sentence, intent, finalWordLists):
        ret = {}
        ret['sentence'] = sentence
        ret['intent'] = intent[0]
        ret['entitylist'] = []
        for word in finalWordLists:
            ret['entitylist']+=[[word[0], word[1][0]]]
        return ret

    def parse(self, sentence):
        vec = self.vec
        if self.model == None:
            raise ValueError('The model is null, please load a model first.')
        entities = self.model[1]
        intents = self.model[0]
        original_sentence = sentence
        sentence = sentence.lower().split()
        entityScores = {}
        finalWordLists = []
        for word in sentence:
            word = self.processWord(word)
            for entity in entities:
                if entity not in entityScores:
                    entityScores[entity]=[]
                for item in entities[entity]:
                    if vec.wordExists(item.lower())==True and vec.wordExists(word)==True:
                        entityScores[entity]+= [1-cosine(vec.getVec(item.lower()), vec.getVec(word))]
                cats_list = []
            for entity in entityScores:
                if len(entityScores[entity])>0:
                    cats_list += [(entity, np.mean(entityScores[entity]))]
            cats_list = sorted(cats_list, key = lambda x: x[1], reverse = True)
            finalWordLists+=[(word, cats_list[0])]
        final_intents_list = []
        for intent in intents:
            score = float(len(intents[intent] & set(sentence)))
            final_intents_list += [(intent,score)]
        final_intents_list = sorted(final_intents_list, key = lambda x: x[1], reverse=True)
        intent = final_intents_list[0]
        ret = self.__create_json(original_sentence, intent, finalWordLists)
        return ret



if __name__=='__main__':
    print ('Unit testing this class...')
    magicslanginstance = magicSlangClass()
    magicslanginstance.train('trainfiles/TravelMate.json')
    magicslanginstance.load_model()
    print (magicslanginstance.parse('Can I see Bombay'))
