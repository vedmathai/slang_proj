from engine_base_class import EngineBaseClass
import json
import pickle
import spacy
import numpy as np

class spacyClass(EngineBaseClass):
    def __init__(self):
        self.model = {}
        self.nlp = None
        self.model = None

    def __getAnnotation(self, annotations, start):
        for annotation in annotations:
            if int(annotation[2])<=start<=int(annotation[3]):
                return annotation[0]
        return ''

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

    def __train_logic(self, train_module ):
        patternsDict = {}
        nlp = spacy.load('en_core_web_sm')
        for dic in train_module:
            sentence = dic['sentence']
            annotations = dic['annotations']
            intent = dic['intent']
            doc = nlp(sentence)
            for word in doc:
                wordAnnotation = self.__getAnnotation(annotations, word.idx)
                if word.ent_type_:
                    ent_type = word.ent_type_
                else:
                    ent_type = 'None'
                if word.dep_ not in patternsDict:
                    patternsDict[word.dep_]={}
                if word.pos_ not in patternsDict[word.dep_]:
                    patternsDict[word.dep_][word.pos_]={}
                if ent_type not in patternsDict[word.dep_][word.pos_]:
                    patternsDict[word.dep_][word.pos_][ent_type]={}
                    patternsDict[word.dep_][word.pos_][ent_type]['annotations']={}
                    patternsDict[word.dep_][word.pos_][ent_type]['intent']={}
                if wordAnnotation not in patternsDict[word.dep_][word.pos_][ent_type]['annotations']:
                    patternsDict[word.dep_][word.pos_][ent_type]['annotations'][wordAnnotation]=0
                patternsDict[word.dep_][word.pos_][ent_type]['annotations'][wordAnnotation]+=1
                if intent not in patternsDict[word.dep_][word.pos_][ent_type]['intent']:
                    patternsDict[word.dep_][word.pos_][ent_type]['intent'][intent]=0
                patternsDict[word.dep_][word.pos_][ent_type]['intent'][intent]+=1
        return patternsDict

    def train(self, train_file=None, outfile = 'models/patterns_spacy.pkl'):
        if train_file == None:
            raise ValueError('The train file cannot be undefined')
        train_file_open = open(train_file)
        train_docs = json.load(train_file_open)
        if not self.__check_train_file_correctness(train_docs):
            raise ValueError('There is some problem with the train file')
        model = self.__train_logic(train_docs)
        self.__save_file(model, outfile)

    def load_model(self, model_file='models/patterns_spacy.pkl'):
        model = pickle.load(open(model_file, 'rb'))
        self.model = model

    def __create_json(self, sentence, finalWordLists, finalIntentsList):
        ret = {}
        ret['sentence'] = sentence
        ret['entitylist'] = []
        ret['intent'] = finalIntentsList
        for word in finalWordLists:
            ret['entitylist']+=[[word[0], word[1]]]
        return ret

    def parse(self, sentence):
        if self.model == None:
            raise ValueError('The model null, please load a model first.')
        patternsDict = self.model
        if self.nlp ==None:
            nlp = spacy.load('en_core_web_sm')
            self.nlp = nlp
        else:
            nlp = self.nlp
        original_sentence = sentence
        sentence = sentence.strip()
        doc = nlp(sentence)
        finalWordListsAnnotations = []
        finalIntentsList = {}
        for word in doc:
            if word.ent_type_:
                ent_type = word.ent_type_
            else:
                ent_type = 'None'
            if word.dep_ in patternsDict:
                if word.pos_ in patternsDict[word.dep_]:
                    if ent_type in patternsDict[word.dep_][word.pos_]:
                        dic = patternsDict[word.dep_][word.pos_][ent_type]['annotations']
                        tuples = []
                        for key in dic:
                            tuples += [(key, dic[key])]
                        tuples = sorted(tuples, key = lambda x: x[1])
                        finalWordListsAnnotations += [(word, tuples[-1][0])]
                        dic = patternsDict[word.dep_][word.pos_][ent_type]['intent']
                        tuples = []
                        for key in dic:
                            tuples += [(key, dic[key])]
                        tuples = sorted(tuples, key = lambda x: x[1])
                        firsttuple = tuples[-1]
                        if firsttuple[0] not in finalIntentsList:
                            finalIntentsList[firsttuple[0]] = []
                        finalIntentsList[firsttuple[0]]+=[firsttuple[1]]

                        continue
        finalIntentsList = [(key, np.mean(finalIntentsList[key])) for key in  finalIntentsList]
        finalIntentsList = sorted(finalIntentsList, key=lambda x: x[1], reverse = True)
        return self.__create_json(sentence, finalWordListsAnnotations, finalIntentsList)

if __name__=='__main__':
    print ('Unit testing this class...')
    spacyinstance = spacyClass()
    spacyinstance.train('trainfiles/all.json')
    spacyinstance.load_model()
    print (spacyinstance.parse('Get me table for 4 in Besant Nagar'))
