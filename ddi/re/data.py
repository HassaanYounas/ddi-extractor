import re, glob, os, nltk, numpy
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordNERTagger

def train_word2vec_model(path):
    data = []
    for filename in glob.glob(os.path.join(path, '*.xml')):
        file = open(filename, 'r', encoding = 'utf-8')
        soup = BeautifulSoup(file, 'html.parser')
        for sentence in soup.find_all('sentence'):
            drug_count = 0
            sentence_text = sentence.get('text')
            if sentence_text == '':
                continue
            entities, pairs = [], []
            for child in sentence.find_all('entity'):
                entities.append(child.get('text'))
            for child in sentence.find_all('pair'):
                if child.get('ddi') == 'true':
                    pairs.append((child.get('e1'), child.get('e2'), '1'))
                else:
                    pairs.append((child.get('e1'), child.get('e2'), '0'))
            if pairs == []:
                continue
            for entity in entities:
                sentence_text = sentence_text.replace(entity, 'Drug' + str(drug_count), 1)
                drug_count += 1
            tokens =  nltk.word_tokenize(re.sub('\W+',' ',sentence_text))       
            clean_tokens = []
            for token in tokens:
                if re.search('^[a-zA-Z]+$', token) or re.search('^Drug[0-9]*$', token):
                    clean_tokens.append(token.lower())
                elif re.search('^Drug[0-9]*[a-zA-Z]*$', token):
                    clean_tokens.append(token[:-1].lower())
            found = False
            for i in range(len(entities)):
                found = False
                drug = 'drug' + str(i)
                for token in clean_tokens:
                    if drug == token:
                        found = True
                        break
                if found == False:
                    break
            if found == False:
                continue
            sentence = []
            for token in clean_tokens:
                sentence.append(token)
            data.append(sentence)
    model = Word2Vec(data, min_count = 1, size = 50, window = 3, sg = 1)
    model.save('../../ai-models/word2vec.model')
    return

def generate(path, save_path, model):
    sentence_file = open(save_path + '/sentences.txt', 'w+', newline = '\n')
    relations_file = open(save_path + '/relations.txt', 'w+', newline = '\n')
    wordvec_file = open(save_path + '/word_embeddings.txt', 'w+', newline = '\n')
    tokens_done = []
    for filename in glob.glob(os.path.join(path, '*.xml')):
        file = open(filename, 'r', encoding = 'utf-8')
        soup = BeautifulSoup(file, 'html.parser')
        for sentence in soup.find_all('sentence'):
            drug_count = 0
            sentence_text = sentence.get('text')
            if sentence_text == '':
                continue
            entities, pairs, relations = [], [], []
            for child in sentence.find_all('entity'):
                entities.append(child.get('text'))
            for child in sentence.find_all('pair'):
                if child.get('ddi') == 'true':
                    pairs.append((child.get('e1'), child.get('e2'), '1'))
                else:
                    pairs.append((child.get('e1'), child.get('e2'), '0'))
            if pairs == []:
                continue
            for entity in entities:
                sentence_text = sentence_text.replace(entity, 'Drug' + str(drug_count), 1)
                drug_count += 1
            tokens =  nltk.word_tokenize(re.sub('\W+',' ',sentence_text))       
            clean_tokens = []
            for token in tokens:
                if re.search('^[a-zA-Z]+$', token) or re.search('^Drug[0-9]*$', token):
                    clean_tokens.append(token.lower())
                elif re.search('^Drug[0-9]*[a-zA-Z]*$', token):
                    clean_tokens.append(token[:-1].lower())
            found = False
            for i in range(len(entities)):
                found = False
                drug = 'drug' + str(i)
                for token in clean_tokens:
                    if drug == token:
                        found = True
                        break
                if found == False:
                    break
            if found == False:
                continue
            for token in clean_tokens:
                if token not in tokens_done:
                    embeddings = model[token]
                    embeddings_text = ''
                    for embedding in embeddings:
                        embeddings_text += str(embedding) + ' '
                    wordvec_file.write(token + ' ' + embeddings_text + '\n')
                    tokens_done.append(token)
            sentence_text = ''
            for token in clean_tokens:
                sentence_text += token 
                sentence_text += ' '
            sentence_text = sentence_text.strip().rsplit('\n')[0]
            for pair in pairs:
                first = 'drug' + str(pair[0][-1:]) 
                second = 'drug' + str(pair[1][-1:])
                relations.append((first, second, pair[2]))
            sentence_file.write(sentence_text + '\n')
            for pair in relations:
                relations_file.write(pair[0] + ' ' + pair[1] + ' ' + pair[2] + '\n')
            relations_file.write('\n')
        file.close()
    wordvec_file.close()
    sentence_file.close()
    relations_file.close()
    return

def format(path):
    wordvec_file = open(path + '/word_embeddings.txt', 'r', encoding = 'utf8')
    sentence_file = open(path + '/sentences.txt', 'r', encoding = 'utf8')
    relations_file = open(path + '/relations.txt', 'r', encoding = 'utf8')
    distances_file = open(path + '/distances.txt', 'w+', newline = '\n')
    vectors_file = open(path + '/vectors.txt', 'w+', newline = '\n')
    output_file = open(path + '/output.txt', 'w+', newline = '\n')
    sentences, ddis, row, word_embeddings = [], [], [], {}
    for line in wordvec_file:
        split_line = line.split()
        word = split_line[0]
        embedding = numpy.array([float(val) for val in split_line[1:]])
        word_embeddings[word] = embedding
    for line in sentence_file:
        text = line.rsplit('\n')[0]
        sentences.append(text)
    for line in relations_file:
        text = line.rsplit('\n')[0]
        if text != '':
            row.append(text)
        else:
            ddis.append(row)
            row = []
    sentence_file.close()
    relations_file.close()
    wordvec_file.close()
    vec_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        training_sentences = []
        for token in tokens:
            training_sentences.append(sum(word_embeddings[token]))
        vec_sentences.append([round(10 * x, 4) for x in training_sentences])
    check = 0
    for i in range(len(vec_sentences)):
        relations = ddis[i]
        sentence = word_tokenize(sentences[i])
        for relation in relations:
            pair = relation.split(' ')
            first = sentence.index(pair[0])
            second = sentence.index(pair[1])
            interaction = pair[2]
            if interaction == '0':
                if check == 5:
                    check = 0
                else:
                    check += 1
                    continue
            padding = [0] * (128 - len(vec_sentences[i]))
            vec_sentence = vec_sentences[i].copy() + padding
            vec_sentence_str = ''
            for i in range(len(vec_sentence)):
                if str(vec_sentence[i]) == '0':
                    distances_file.write(str(0) + ' ' + str(0) + '\n')
                else:
                    distances_file.write(str(i - first) + ' ' + str(i - second) + '\n')
                vec_sentence_str += str(vec_sentence[i])
                vec_sentence_str += ' '
            vectors_file.write(vec_sentence_str + '\n')
            distances_file.write('\n')
            if interaction == '0':
                output_file.write(interaction + ' ' + '1' '\n')
            else:
                output_file.write(interaction + ' ' + '0' '\n')
    distances_file.close()
    vectors_file.close()
    output_file.close()
    return

train_word2vec_model('../../corpora/')

generate('../../corpora/', 'training', Word2Vec.load('../../ai-models/word2vec.model'))
format('training')

generate('../../corpora/testing/', 'testing', Word2Vec.load('../../ai-models/word2vec.model'))
format('testing')

os.remove('./training/word_embeddings.txt')
os.remove('./testing/word_embeddings.txt')

os.remove('./training/sentences.txt')
os.remove('./testing/sentences.txt')

os.remove('./training/relations.txt')
os.remove('./testing/relations.txt')