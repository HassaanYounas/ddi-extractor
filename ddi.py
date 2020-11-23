import os
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

java_path = 'C:/Program Files/Java/jre1.8.0_271/bin'
os.environ['JAVAHOME'] = java_path

ner = StanfordNERTagger(
    './ai-models/ner-model.ser.gz', 
    './java/stanford-ner.jar', 
    encoding = 'utf-8'
)

print(ner.tag(word_tokenize('The fluoroquinolones for urinary tract infections: a review.')))