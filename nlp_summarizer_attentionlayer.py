# -*- coding: utf-8 -*-
"""NLP_Summarizer_AttentionLayer.ipynb

Original file is located at
    https://colab.research.google.com/drive/1t9QuqEhovfOkFuVNo-mKQ6nUnavfCLmV
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.layers import Layer
from keras.layers import Concatenate
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, TimeDistributed, LSTM, Embedding, Input
from keras import Model

# attention.py
from attention import AttentionLayer

nltk.download('stopwords')

# dataset
data = pd.read_csv('cnn_dailymail\\train.csv', encoding='latin-1')

# CONTRACTION MAPPING FOR PREPROCESSING
mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
           "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
           "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
           "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
           "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have",
           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
           "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
           "you're": "you are", "you've": "you have"}

StopWords = set(stopwords.words('english'))


def preprocess(text):
    text = text.lower()  # converting input to lowercase
    # Removing punctuations and special characters.
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub('"', '', text)  # Removing double quotes.
    # Replacing contractions.
    text = ' '.join(
        [mapping[t] if t in mapping else t for t in text.split(" ")])
    text = re.sub(r"'s\b", "", text)  # Eliminating apostrophe.
    # Removing non-alphabetical characters
    text = re.sub("[^a-zA-Z]", " ", text)
    # Removing stopwords.
    text = ' '.join([word for word in text.split() if word not in StopWords])
    # Removing very short words
    text = ' '.join([word for word in text.split() if len(word) >= 3])
    return text


# Apply preprocessing to both text and summary
cleaned_text = []
cleaned_summary = []
for text in data['article']:
    cleaned_text.append(preprocess(text))
for summary in data['highlights']:
    cleaned_summary.append(preprocess(summary))
cleaned_data = pd.DataFrame()
cleaned_data['article'] = cleaned_text
cleaned_data['highlight'] = cleaned_summary

# Replacing empty string summaries with nan values and then dropping those datapoints.
cleaned_data['highlight'].replace('', np.nan, inplace=True)
cleaned_data.dropna(axis=0, inplace=True)

# Adding START and END tokens for indication
cleaned_data['highlight'] = cleaned_data['highlight'].apply(
    lambda x: '<START>' + ' ' + x + ' ' + '<END>')
for i in range(10):
    print('Article: ', cleaned_data['article'][i])
    print('Highlight:', cleaned_data['highlight'][i])
    print('\n')

# Get max length of texts and summaries.
news_length = max([len(text.split()) for text in cleaned_data['article']])
headline_length = max([len(text.split())
                      for text in cleaned_data['highlight']])
print(news_length, headline_length)

text_word_count = []
headline_word_count = []

for i in cleaned_data['article']:
    text_word_count.append(len(i.split()))

for i in cleaned_data['highlight']:
    headline_word_count.append(len(i.split()))

length_df = pd.DataFrame(
    {'Body': text_word_count, 'Highlights': headline_word_count})
length_df.hist(bins=20)
plt.show()

# splitting data into 80-20 ratio as train and test data
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_data['article'], cleaned_data['highlight'], test_size=0.2, random_state=0)

# Keras tokenizer for news text.
tokenizer_news = Tokenizer()
tokenizer_news.fit_on_texts(list(X_train))
x_train_seq = tokenizer_news.texts_to_sequences(X_train)
x_test_seq = tokenizer_news.texts_to_sequences(X_test)
# padding short texts with 0s.
x_train_pad = pad_sequences(x_train_seq, maxlen=news_length, padding='post')
x_test_pad = pad_sequences(x_test_seq, maxlen=news_length, padding='post')
# Vocab size of texts.
news_vocab = len(tokenizer_news.word_index) + 1

# Keras Tokenizer for summaries.
tokenizer_headline = Tokenizer()
tokenizer_headline.fit_on_texts(list(y_train))
y_train_seq = tokenizer_headline.texts_to_sequences(y_train)
y_test_seq = tokenizer_headline.texts_to_sequences(y_test)
y_train_pad = pad_sequences(
    y_train_seq, maxlen=headline_length, padding='post')
y_test_pad = pad_sequences(y_test_seq, maxlen=headline_length, padding='post')
# Vocab size of summaries.
headline_vocab = len(tokenizer_headline.word_index) + 1

K.clear_session()

embedding_dim = 300  # Size of word embeddings.
latent_dim = 500  # Number of neurons in LSTM layer.

# Embedding Layer
e_input = Input(shape=(news_length, ))
e_emb = Embedding(news_vocab, embedding_dim, trainable=True)(e_input)

# Three LSTM layers ----> encoder.
e_lstm1 = LSTM(latent_dim, return_sequences=True,
               return_state=True, dropout=0.3, recurrent_dropout=0.2)
y_1, a_1, c_1 = e_lstm1(e_emb)
e_lstm2 = LSTM(latent_dim, return_sequences=True,
               return_state=True, dropout=0.3, recurrent_dropout=0.2)
y_2, a_2, c_2 = e_lstm2(y_1)
e_lstm3 = LSTM(latent_dim, return_sequences=True,
               return_state=True, dropout=0.3, recurrent_dropout=0.2)
encoder_output, a_enc, c_enc = e_lstm3(y_2)

# Single LSTM layer ----> decoder
d_input = Input(shape=(None,))
d_emb = Embedding(headline_vocab, embedding_dim, trainable=True)(d_input)
d_lstm = LSTM(latent_dim, return_sequences=True,
              return_state=True, dropout=0.3, recurrent_dropout=0.2)
# Final output states of encoder last layer are fed into decoder.
decoder_output, decoder_fwd, decoder_back = d_lstm(
    d_emb, initial_state=[a_enc, c_enc])

# Attention Layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_output, decoder_output])

# concatenating decoder input to attention layer output
decoder_concat_input = Concatenate(
    axis=-1, name='concat_layer')([decoder_output, attn_out])
# dense time distributed layer with softw=max fucntion for predicting the next word
decoder_dense = TimeDistributed(Dense(headline_vocab, activation='softmax'))
decoder_output = decoder_dense(decoder_concat_input)

# creating model
model = Model([e_input, d_input], decoder_output)
model.summary()

# Training the model with Early Stopping callback on val_loss.
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
callback = keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=2)
history = model.fit([x_train_pad, y_train_pad[:, :-1]], y_train_pad.reshape(y_train_pad.shape[0], y_train_pad.shape[1], 1)[:, 1:], epochs=3, callbacks=[
                    callback], batch_size=512, validation_data=([x_test_pad, y_test_pad[:, :-1]], y_test_pad.reshape(y_test_pad.shape[0], y_test_pad.shape[1], 1)[:, 1:]))

# Encoder inference model with trained inputs and outputs.
encoder_model = Model(inputs=e_input, outputs=[encoder_output, a_enc, c_enc])

# Initialising state vectors for decoder.
decoder_initial_state_a = Input(shape=(latent_dim,))
decoder_initial_state_c = Input(shape=(latent_dim,))
decoder_hidden_state = Input(shape=(news_length, latent_dim))

# Decoder inference model
decoder_out, decoder_a, decoder_c = d_lstm(
    d_emb, initial_state=[decoder_initial_state_a, decoder_initial_state_c])
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state, decoder_out])
decoder_inf_concat = Concatenate(
    axis=-1, name='concat')([decoder_out, attn_out_inf])

decoder_final = decoder_dense(decoder_inf_concat)
decoder_model = Model([d_input]+[decoder_hidden_state, decoder_initial_state_a,
                      decoder_initial_state_c], [decoder_final]+[decoder_a, decoder_c])

# Function to generate output summaries.


def decoded_sequence(input_seq):
    # Collecting output from encoder inference model.
    encoder_out, encoder_a, encoder_c = encoder_model.predict(input_seq)
    # Initialise input to decoder neuron with START token. Thereafter output token predicted by each neuron will be used as input for the subsequent.
    # Single elt matrix used for maintaining dimensions.
    next_input = np.zeros((1, 1))
    next_input[0, 0] = tokenizer_headline.word_index['start']
    output_seq = ''
    # Stopping condition to terminate loop when one summary is generated.
    stop = False
    while not stop:
        # Output from decoder inference model, with output states of encoder used for initialisation.
        decoded_out, trans_state_a, trans_state_c = decoder_model.predict(
            [next_input] + [encoder_out, encoder_a, encoder_c])
        # Get index of output token from y(t) of decoder.
        output_idx = np.argmax(decoded_out[0, -1, :])
        # If output index corresponds to END token, summary is terminated without of course adding the END token itself.

        if output_idx == tokenizer_headline.word_index['end']:
            stop = True
        elif output_idx > 0 and output_idx != tokenizer_headline.word_index['start']:
            # Generate the token from index.
            output_token = tokenizer_headline.index_word[output_idx]
            output_seq = output_seq + ' ' + output_token  # Append to summary

        # Pass the current output index as input to next neuron.
        next_input[0, 0] = output_idx
        # Continously update the transient state vectors in decoder.
        encoder_a, encoder_c = trans_state_a, trans_state_c

    return output_seq

# Print predicted summmaries and actual summaries for 60 texts.


predicted = []
for i in range(20):
    print('Information:', X_test.iloc[i])
    print('Actual Highlight:', y_test.iloc[i])
    print('Predicted Highlight:', decoded_sequence(
        x_test_pad[i].reshape(1, news_length)))
    predicted.append(decoded_sequence(
        x_test_pad[i].reshape(1, news_length)).split())
