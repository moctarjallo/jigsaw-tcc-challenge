import pandas as pd
import numpy as np

import os

from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import adam, Adam
from keras.preprocessing.sequence import pad_sequences

# load data
data = pd.read_csv('data/sample_train.csv')
# print(data)

# preprocessing comments
comments = data['comment_text']
# print(comments)

# split into words
def word_split(comments):
    return [[word for word in comment.split()] for comment in comments]

comments = word_split(comments)
# print(comments)

# comments vocabulary
vocab = set()
for comment in comments:
    for word in comment:
        vocab.add(word)

# vocabulary size
vocab_size = len(vocab)
# print(vocab_size)

# word to int
word_to_int = dict()
for i, w in enumerate(vocab):
    word_to_int[w] = i
# print(word_to_int)

# numeric comments + normalization
def normalize(comments):
    for i, comment in enumerate(comments):
        for j, word in enumerate(comment):
            comments[i][j] = word_to_int[word]/vocab_size  # normalisation step
    return comments
comments = normalize(comments)
# print(comments)
# for c in comments:
#     print(len(c))  # different lengths

# pad comments to a fixed size
    # find the longest comment length
max_len = 0
for comment in comments:
    if max_len  < len(comment):
        max_len = len(comment)
# print(max_len)
    # pad
comments = pad_sequences(comments, max_len)

# labels
labels = [data.values[i][2:] for i in range(len(data))]
# print(labels)

# Training data
X = np.array(comments)
y = np.array(labels)
# print(X.shape)
# print(y.shape)

# model
model = Sequential()
model.add(Dense(10, input_shape=(X.shape[1],)))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # training
#     # callbacks
# events = TensorBoard('events/1')
# ckpts = ModelCheckpoint('ckpts/1/weigths{epoch:02d}-{loss:.4f}.hdf5')
# model.fit(X, y, batch_size=10, epochs=10, callbacks=[events, ckpts])

# # evaluation
# model.evaluate(X, y, batch_size=10)
#
# # prediction
# y_pred = model.predict(X)
# print(y)

# test
test_data = pd.read_csv('data/sample_train.csv')
test_comments = test_data['comment_text']
test_ids = test_data['id']
test_comments = word_split(test_comments)
test_comments = normalize(test_comments)
test_comments = np.array(pad_sequences(test_comments, max_len))
model.load_weights('ckpts/1/weigths10-0.3923.hdf5')
pred_labels = np.round(model.predict(test_comments), 1)

def save(predictions, file):
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    pd.DataFrame(predictions, index=test_ids, columns=labels).to_csv(file)

save(pred_labels, 'data/sample_test1.csv')

# Problems:
    # cannot test on new data: Key Error raised because of non existing word in word_to_int dictionary
    # is the choosen vocabulary complete enough?
    # what alternatives?

