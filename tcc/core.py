import pandas as pd
import numpy as np

import os

from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import adam, Adam

from sklearn.feature_extraction.text import HashingVectorizer

class ToxicComments:
    """Reads a toxic comments csv file

    To get the next batch of comments, ids or labels, you'd
    have to call explicitly next(self) (or through a for loop),
    othewise it is the current batch of size self.batch_size
    that will be available"""

    def __init__(self, csvfile, batch_size=None):
        self.data = pd.read_csv(csvfile)
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = len(self.data)

        self.index = 0

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        return self

    """Return the next tuple (ids, comments, labels) of size batch_size"""
    def __next__(self):
        if self.index > len(self.data):
            raise StopIteration
        elif self.index + self.batch_size > len(self.data):
            self.batch_size = len(self.data) - self.index
            result = self.data[self.index:self.index+self.batch_size]
            self.index = self.index + self.batch_size + 1
        else:
            result = self.data[self.index:self.index+self.batch_size]
            self.index += self.batch_size

        ids = list(result['id'])
        comments = list(result['comment_text'])
        labels = [list(result.values[i][2:]) for i in range(self.batch_size)]
        return ids, comments, labels

"""Return an array of floats from document"""
def prepare(document, n_features):
    return HashingVectorizer(n_features=n_features)\
            .transform(document)\
            .toarray()


class ToxicModel:
    def __init__(self, input_shape, output_shape):
        self.model = Sequential()
        self.model.add(Dense(10, input_shape=(input_shape,)))
        self.model.add(Dense(output_shape, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y, callback_dirs=[None, None]):
        self.model.fit(X, y, batch_size=10, epochs=10,\
                       callbacks=[TensorBoard(callback_dirs[0]),\
                                  ModelCheckpoint(os.path.join(callback_dirs[1], \
                                        'weigths{epoch:02d}-{loss:.4f}.hdf5'))])

    def predict(self, X):
        return np.round(self.model.predict(X), 1)
