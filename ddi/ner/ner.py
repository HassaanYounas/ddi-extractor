import glob, os, nltk, csv
from bs4 import BeautifulSoup

path = '../../corpora'
tsv_writer = csv.writer(open('../../java/train.tsv', 'w+', newline = ''), delimiter = '\t')
nltk.download('punkt')

for filename in glob.glob(os.path.join(path, '*.xml')):
    file = open(filename, 'r', encoding = 'utf-8')
    soup = BeautifulSoup(file, 'html.parser')
    for sentence in soup.find_all('sentence'):
        try:
            sentence_text = sentence.get('text')
            entities, entities_with_type = [], []
            for child in sentence.find_all('entity'):
                entities.append(child.get('text'))
                entities_with_type.append((child.get('text'), child.get('type')))
            for entity in entities:
                sentence_text = sentence_text.replace(entity, 'ABCXYZ', 1)
            sentence_text = sentence_text.replace('-', ' ')
            tokens = nltk.word_tokenize(sentence_text)
            for entity in entities:
                tokens[tokens.index('ABCXYZ')] = entity
            for token in tokens:
                if token in entities:
                    if ' ' in token:
                        token_split = token.split(' ')
                        type = entities_with_type[entities.index(token)][1]
                        for word in token_split:
                            tsv_writer.writerow((word, type))
                    else:
                        tsv_writer.writerow((token, entities_with_type[entities.index(token)][1]))
                else:
                    tsv_writer.writerow((token, 'O'))
            tsv_writer.writerow('')
        except:
            continue