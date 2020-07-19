import numpy as np
import pandas as pd
import time, re

from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv1D
from keras.layers import Dropout, Activation, MaxPooling1D
from keras.layers.embeddings import Embedding

from keras.callbacks import EarlyStopping
from keras import regularizers, optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


## SOURCE
print('Loading data...')

cols = ['sentiment','text']
df = pd.read_csv('bin_twitter.csv',encoding='latin-1',header=None, \
                 names=cols, usecols=[0,5])

label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

X = df['text'].fillna('').tolist()
X = [str(i) for i in X]
X = [re.sub('@[^\s]+','',i) for i in X]
y = df['sentiment'].fillna('').tolist()

# balance classes 
class_weights = compute_class_weight('balanced', np.unique(y), y)
class_weights = dict(enumerate(class_weights))

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# clean text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# convert text to integer sequences
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = to_categorical(y_train, 2, dtype='int64')
y_test = to_categorical(y_test, 2, dtype='int64')

total_words = len(tokenizer.word_index) + 1  
print('Found %s unique tokens.' % total_words)
maxlen = 100

# sequences that are shorter than the max length are padded with value
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


## TARGET
cols_target = ['target','text','label']
target_df = pd.read_csv('bin_manifesto.csv', encoding='latin-1', header=None, \
                     names=cols_target, usecols=[0,1,2])
target_df = target_df[['target','text','label']]
X_target = target_df['text'].fillna('').tolist()
X_target = [decontracted(str(i)) for i in X_target]
targ_targ = target_df['target'].fillna('').tolist()
y_target = target_df['label'].fillna('').tolist()

tokenizer.fit_on_texts(X_target)
X_target = tokenizer.texts_to_sequences(X_target)
X_target = pad_sequences(X_target, padding='post', maxlen=maxlen)

X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_target, y_target, test_size=0.33, random_state=33)

y_train_B = to_categorical(y_train_B, 2, dtype='int64')
y_test_B = to_categorical(y_test_B, 2, dtype='int64')


# GloVe embeddings
embeddings_index = dict()
with open(glove_path,
          encoding="utf8") as glove:
  for line in glove:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
  glove.close()
  
embedding_matrix = np.zeros((total_words, 200))
for word, index in tokenizer.word_index.items():
    if index > total_words - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


## BASE MODEL
# parameters
maxlen = maxlen
lstm_output_size = 128
weight_decay = 1e-4
embedding_dim = 200
batch_size = 256
kernel_size = 4
epochs = 20

print('Build LSTM model...')
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, 
                    weights=[embedding_matrix],
                    input_length=maxlen,
                    trainable=False))
model.add(Bidirectional(LSTM(lstm_output_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(LSTM(lstm_output_size, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='sigmoid'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

lstm_history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              class_weight=class_weights)

model.save('base_model.h5')


## FINE-TUNED MODEL
lstm_model_A = load_model('my_lstm.h5')

X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_oos, y_oos, test_size=0.33, random_state=33)

y_train_B = to_categorical(y_train_B, 2, dtype='int64')
y_test_B = to_categorical(y_test_B, 2, dtype='int64')

lstm_model_B_on_A = Sequential(lstm_model_A.layers[:-1])
lstm_model_B_on_A.add(Dense(2, activation="sigmoid"))

for layer in lstm_model_B_on_A.layers[:-1]:
    trainable = True

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
lstm_model_B_on_A.compile(loss="binary_crossentropy", optimizer=adam,
                     metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=0)
callbacks = [early_stopping]

lstm_B_history = lstm_model_B_on_A.fit(X_train_B, y_train_B, epochs=10,
                           validation_data=(X_test_B, y_test_B),
                           callbacks=callbacks)

lstm_model_B_on_A.save('finetuned_model.h5')


## RESULTS
def argmax_keepdims(x, axis):
    output_shape = list(x.shape)
    output_shape[axis] = 1
    return np.argmax(x, axis=axis).reshape(output_shape)

y_test_fit = argmax_keepdims(y_test_B , axis=1)

cnn_model_B = load_model('finetuned_model.h5')
y_pred = cnn_model_B.predict_classes(X_test_B)

target_names = ['negative', 'non-negative']
print('----------------------EVALUATION----------------------\n')
print(classification_report(y_test_fit, y_pred, target_names=target_names))
