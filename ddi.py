import os, json
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

java_path = 'C:/Program Files/Java/jre1.8.0_271/bin'
os.environ['JAVAHOME'] = java_path

def ner(text):
    ner = StanfordNERTagger(
        './ai-models/ner-model.ser.gz', 
        './java/stanford-ner.jar', 
        encoding = 'utf-8'
    )
    drugs = { 'drug': [], 'drug-n': [], 'group': [], 'brand': [] }
    tags = ner.tag(word_tokenize(text))
    for token in tags:
        if token[1] is not 'O':
            drugs[token[1]].append({ 'name': token[0] })
    return drugs

def re(text):
    
    return