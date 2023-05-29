# -*- coding: utf-8 -*-
"""Tutorial3_Summarizer.ipynb
Original file is located at
    https://colab.research.google.com/drive/1PYDFA-HTo5yTzieRJygcaweO_nt78rYP
"""

from random import randint
from keras.utils import pad_sequences
from pickle import dump
import re
import string
import csv

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

# load doc into memory. given a filename, it returns a sequence of text
def load_doc(filename):
    # open the file as read only
    #file = open(filename, encoding="utf8", errors='ignore')
    file = open(filename, errors='ignore')
    # read all text
    text = file.read()
    #text = ''
    #text = text.join([next(file) for _ in range(100000)])
    # close the file
    file.close()

    return text

# load document
#in_filename = 'republic.txt'
in_filename = 'cnn_dailymail\\train_summaries_truncated2.txt'
doc = load_doc(in_filename)

# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile(' [%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lowercase
    tokens = [word.lower() for word in tokens]
    return tokens

# clean document
tokens = clean_doc(doc)
print(tokens[:200])  # print first 200 tokens
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

# organize into sequences of tokens. returns a long list of lines
length = 50 + 1  # 50 previous words, and a target output word
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length: i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)

print('Total Sequences: %d' % len(sequences))

# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# save sequences to file
out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)

# Load training data using load_doc function and split it into
# seperate training sequences
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# vocabulary size for Embedding layer later
vocab_size = len(tokenizer.word_index) + 1  # add 1 since arrays start at 0

# separate into input and output
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# define the model
def define_model(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    # compile network
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# define model
model = define_model(vocab_size, seq_length)
# fit model
model.fit(X, y, batch_size=128, epochs=20)  # epoch originally 100

# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = np.argmax(model.predict(encoded, verbose=0), axis=-1)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

# select a seed text
seed_text = lines[randint(0, len(lines))]
print(seed_text + '\n')
# generate new text
# NOTE: may not work well since epoch is 5 lol
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(generated)
