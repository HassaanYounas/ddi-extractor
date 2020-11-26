import numpy as np
from bs4 import BeautifulSoup
import os, json, re, random, glob
from gensim.models import Word2Vec
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from keras.models import model_from_json

java_path = 'C:/Program Files/Java/jre1.8.0_271/bin'
os.environ['JAVAHOME'] = java_path

def get_ner_model():
    return StanfordNERTagger(
        './ai-models/ner-model.ser.gz', 
        './java/stanford-ner.jar', 
        encoding = 'utf-8'
    )

def get_re_model():
    json_file = open('./ai-models/lstm.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('./ai-models/lstm.h5') 
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    return model

def named_entity_recognition(text):
    ner = get_ner_model()
    drugs = { 'drug': [], 'drug-n': [], 'group': [], 'brand': [] }
    tags = ner.tag(word_tokenize(text))
    to_skip , entities = 0, []
    for i in range(len(tags)):
        if tags[i][1] is not 'O':
            if to_skip == 0:
                if i != len(tags) - 1 and tags[i + 1][1] is not 'O':
                    drug_name = tags[i][0] + ' '
                    for j in range(i + 1, len(tags)):
                        if tags[j][1] is 'O':
                            break
                        else:
                            to_skip += 1
                            drug_name += tags[j][0] + ' '
                    entities.append((drug_name.strip(), tags[i][1]))
                else:
                    entities.append((tags[i][0], tags[i][1]))
            else:
                to_skip -= 1
    for drug in entities:
        if drug[1] is not 'O':
            drugs[drug[1]].append({ 'name': drug[0] })
    return drugs

def relation_extraction(text):
    relations = []
    drug_count = 0
    entities, pairs, sentence_pairs, vec_pairs, x_predict = [], [], [], [], []
    ner = get_ner_model()
    tags = ner.tag(word_tokenize(text))
    to_skip , count = 0, 0
    for i in range(len(tags)):
        if tags[i][1] is not 'O':
            if to_skip == 0:
                if i != len(tags) - 1 and tags[i + 1][1] is not 'O':
                    drug_name = tags[i][0] + ' '
                    for j in range(i + 1, len(tags)):
                        if tags[j][1] is 'O':
                            break
                        else:
                            to_skip += 1
                            drug_name += tags[j][0] + ' '
                    entities.append((drug_name.strip(), 'drug' + str(count)))
                else:
                    entities.append((tags[i][0], 'drug' + str(count)))
                count += 1
            else:
                to_skip -= 1
    for i in range(len(entities) - 1):
        for j in range(i + 1, len(entities)):
            pairs.append((entities[i][1], entities[j][1]))
    if len(pairs) is not 0:
        for entity in entities:
            text = text.replace(entity[0], 'Drug' + str(drug_count), 1)
            drug_count += 1
        tokens = word_tokenize(re.sub('\W+', ' ', text))
        clean_tokens = []
        for token in tokens:
            if re.search('^[a-zA-Z]+$', token) or re.search('^Drug[0-9]*$', token):
                clean_tokens.append(token.lower())
            elif re.search('^Drug[0-9]*[a-zA-Z]*$', token):
                clean_tokens.append(token[:-1].lower())
        for i in range(len(pairs)):
            first, second = pairs[i][0], pairs[i][1]
            current_tokens = ['ABCXYZ' if token == first else token for token in clean_tokens]
            current_tokens = ['ABCXYZ' if token == second else token for token in current_tokens]
            sentence_pairs.append(current_tokens)
        word2vec = Word2Vec.load('./ai-models/word2vec.model')
        for sentence in sentence_pairs:
            vec_pair = []
            for token in sentence:
                if token == 'ABCXYZ':
                    vec_pair.append(1)
                else:
                    try:
                        vec_pair.append(round(10 * sum(word2vec[token]), 4))
                    except:
                        vec_pair.append(round(10 * random.random(), 4))
            vec_pairs.append(vec_pair)
        for vec_pair in vec_pairs:
            index_first = [i for i, n in enumerate(vec_pair) if n == 1][0]
            index_second = [i for i, n in enumerate(vec_pair) if n == 1][1]
            features = []
            for i in range(len(vec_pair)):
                features.append([vec_pair[i], i - index_first, i - index_second])
            for i in range(128 - len(vec_pair)):
                features.append([0, 0, 0])
            x_predict.append(features)
        x_predict = np.array(x_predict)
        x_predict = x_predict.reshape(len(x_predict), 128, 3)
        model = get_re_model()
        pred_output = model.predict(x_predict, batch_size = 200)
        for i in range(len(pred_output)):
            first, second = '', ''
            for drug in entities:
                if drug[1] == pairs[i][0]:
                    first = drug[0]
                if drug[1] == pairs[i][1]:
                    second = drug[0]
            if pred_output[i][0] < 0.5:
                relations.append({ 'first': first, 'second': second, 'ddi': False })
            else:
                relations.append({ 'first': first, 'second': second, 'ddi': True })
    return relations