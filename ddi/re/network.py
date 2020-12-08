import os
import numpy as np
from numpy.random import seed
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

vectors_file = open('training/vectors.txt', 'r', encoding = 'utf8')
distances_file = open('training/distances.txt', 'r', encoding = 'utf8')
output_file = open('training/output.txt', 'r', encoding = 'utf8')

test_vectors_file = open('testing/vectors.txt', 'r', encoding = 'utf8')
test_distances_file = open('testing/distances.txt', 'r', encoding = 'utf8')
test_output_file = open('testing/pred_output.txt', 'w+', newline = '\n')

sentences, distances, interactions = [], [], []
testing_sentences, testing_distances = [], []
x_train, y_train, x_test = [], [], []

for line in vectors_file:
    line = line.split(' ')[:-1]
    sentences.append([float(val) for val in line])

for line in distances_file:
    line = line.rstrip().split(' ')
    distances.append(line)

for line in output_file:
    interactions.append([int(line[0]), int(line[2])])

for line in test_vectors_file:
    line = line.split(' ')[:-1]
    testing_sentences.append([float(val) for val in line])

for line in test_distances_file:
    line = line.rstrip().split(' ')
    testing_distances.append(line)

j = 0
for i in range(len(sentences)):
    sentence = sentences[i]
    container = []
    k = 0
    while True:
        token = [float(sentence[k]), float(distances[j][0]), float(distances[j][1])]
        container.append(token)
        k = k + 1
        j = j + 1
        if len(distances[j]) is 1:
            j = j + 1
            break
    x_train.append(container)

j = 0
for i in range(len(testing_sentences)):
    sentence = testing_sentences[i]
    container = []
    k = 0
    while True:
        token = [float(sentence[k]), float(testing_distances[j][0]), float(testing_distances[j][1])]
        container.append(token)
        k = k + 1
        j = j + 1
        if len(testing_distances[j]) is 1:
            j = j + 1
            break
    x_test.append(container)

x_train = np.array(x_train)
x_train = x_train.reshape(len(x_train), 128, 3)
y_train = np.array(interactions)

x_test = np.array(x_test)
x_test = x_test.reshape(len(x_test), 128, 3)

model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape = (128, 3))))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
model.fit(x_train, y_train, batch_size = 200, epochs = 100, shuffle = True)

model_json = model.to_json()
with open('../../ai-models/lstm.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('../../ai-models/lstm.h5')

pred_output = model.predict(x_test, batch_size = 200)
for pred in pred_output:
    if pred[0] < 0.5:
        test_output_file.write('0' + ' ' + '1' + '\n')
    else:
        test_output_file.write('1' + ' ' + '0' + '\n')

vectors_file.close()
distances_file.close()
output_file.close()

test_vectors_file.close()
test_distances_file.close()
test_output_file.close()