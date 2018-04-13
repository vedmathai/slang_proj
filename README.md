# Intent and Named Entity Recognition
The main two files are train.py and infer.py. Train takes input of an engine name, output directory and a training file. For example:
```python
python3 train.py --engine engine_name --out model_location train_file
```
it saves a model to the given output file.
while infer takes an input of engine name, trained model and test sentence.
In the given format:
```
python3 infer.py --engine engine_name --model model_location 'sample sentence'
```
it outputs a parse and classified intent of the sentence.

Following are the possible inputs for the arguments:
```--engine```:
* spacy
* magic
* magicSlang
```
### Requirements:
These run on python3 and require:
* spacy
* numpy
* A downloaded and unzipped version of version of the [fasttext](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip) embeddings in the same folder, for magic and magicSlang classes.

### Quick Run:
Run sh example.sh to have a quick go at training and to see output from the models.

### Usage, Expected outputs and Internals:
Below we will describe the usage, expected outputs and internals of the three classes:

#### magicSlang:
Takes an input only in the format same as that of ```TravelMate.json```.

It reads all the entities and saves the given examples in the model, and a bag of words
for the intents.
During inference it ranks the annotations based on the mean similarities to the entities collected during training. The similarity is based on **cosine similarity** of the fasttrack embeddings.The intents are found using **bag of Words** similarity between the intent and test sentence.
The output is of the form:
```
{
'sentence': sentence,
'intent': intent,
'entitylist': ['word1', 'annotation1',
                'word2', 'annotaion2'...
                'wordn', 'annotationn']
}
```



### spacy and magic:
```spacy``` and ```magic``` engine take in a json custom annotated on the snips data, namely ```all.json``` available in the trainfiles
folder. They both will output a json of the format:

```{
'sentence': sentence,
'intent': intent,
'entitylist': ['word1', 'annotation1',
                'word2', 'annotaion2'...
                'wordn', 'annotationn']
}
```
The possible list of annotations are :

* **main_verb**: This the best tell for the action of the sentence. What has to be done.
* **subject**: This the object that the verb is directly affecting.
* **location**: This may be a geographical position
* **time**: This is either a day or date.
* **event_descriptor**: This is something that modifies the subject. Like a specific name or description. Like 'Action' movie or 'Jurassic Park'
* **question**: This is a question word. Existance of this defines what has to be returned to the query.
* **aux_obj**: This another object different from the main object.
* **None**: This is a default placeholder for those words that don't fit into one of the above types.

#### Intent Classification:
Following are the list of intents that this dataset had:
* AddToPlaylist
* BookRestaurant
* GetWeather
* RateBook
* SearchCreativeWork
* SearchScreeningEvent.

#### How it works internally:
The ```spacy``` engine parses the following information from the sentence:
**Parts of speech**, **dependency parse** and the output from a **Named Enitity recognizer**.
It counts the occurence for each combination of output of the three signals and learns
a pattern mapping the output from the above three to the individual word annotations
and the overall sentence intent.
The training file saves the learned statistics and the inference file then finds
the most probable output for the given sentence.

The ```magic``` engine saves a list of all the annotated words given in the training file.
At inference time it reads the sentence and ranks the annotation for each word given
its similarity to the other words in the individual lists, as per **fasttext** embeddings.
Larger the cosine similarity of the embeddings the larger the similarity.
We use the same logic for intents classification at the sentence level.

#### How this dataset was created:
The data was annotated using a custom built annotation tool on **~40 sentences** for each intent.

### Key Takeaways:
When it comes to deep learning, or machine learning for that matter, to have a proper feedback system we'd need a **lot more data available** or else we'll only be able to build a pure feedforward system with some heuristics.

In the both the training datasets the sheer number of sentences that were available to do any form of more detailed machine learning was not enough, all the more less in ```TravelMate.json```. This leaves us to depend only on **heuristics** or on **pretrained models**. Which is what I have done. IMHO the named enitity recognizer showed promise in both the spacy case and the pre trained word embeddings.

However, the intent recognizer left a **lot to be desired**. However, this is not the end of the line since by being clever about **data collection** and reuse of data between tasks a lot of improvement can be made over time.

It may be argued that collecting data from customers has to be as **minimum** as possible however if we look the other way round and instead of asking them to name as many entities and patterns as possible we can easily build a suggestions tool that take their intitial set of sentences, **annotate them** and **label intents** with all our previous learned knowledge and ask them to **verify** some finite number of them and give them an incentive that the more they verify the more accurate the system would be. 
