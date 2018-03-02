import pandas as pd
import numpy as np

import os

from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences

import numpy as np

from tcc.core import Model

default_weights = 'ckpts/1/weigths10-0.3886.hdf5'


def save(ids, predictions, file,):
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    pd.DataFrame(predictions, index=ids, columns=labels).to_csv(file)


class KModel(Model):
    def __init__(self, in_shape, out_shape):
        self.model = Sequential()
        self.model.add(Dense(10, input_shape=(in_shape,)))
        self.model.add(Dense(out_shape, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        events= TensorBoard('events/1')
        ckpts = ModelCheckpoint('ckpts/1/weigths{epoch:02d}-{loss:.4f}.hdf5')
        self.model.fit(X_train, y_train, batch_size=10, epochs=10, callbacks=[events, ckpts])

    def load_default(self):
        self.model.load_weights(default_weights)

    def predict(self, comment):
        return self.model.predict(comment)


class CSVData(object):
    def __init__(self, csvfile):
        self.data = pd.read_csv(csvfile)

    def __getitem__(self, item):
        if item == 'comments':
            return self.data['comment_text']
        elif item == 'labels':
            return [self.data.values[i][2:] for i in range(len(self.data))]
        elif item == 'ids':
            return self.data['id']
        else:
            raise KeyError(item)



class Processor(object):
    def __init__(self):
        self.text = None

    def __call__(self, data):
        self.text = data
        return self

    @property
    def words(self):
        self.text = [[word for word in sent.split()] for sent in self.text]
        return self

    @property
    def vocabulary(self):
        vocab = dict()
        i = 0
        for sent in self.text:
            for word in sent:
                if word not in vocab:
                    vocab[word] = i
                    i+=1
        return vocab

    @property
    def max_len(self):
        length = 0
        for sent in self.text['comments']:
            if length  < len(sent):
                lenght = len(sent)
        return lenght

    @property
    def norm(self):
        for i, comment in enumerate(self.text):
            for j, word in enumerate(comment):
                self.text[i][j] = self.vocabulary[word]/len(self.vocabulary)  # normalisation step
        return pad_sequences(self.text, self.max_len)

