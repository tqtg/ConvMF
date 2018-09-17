'''
Created on Dec 8, 2015
@author: donghyun
'''
import numpy as np

np.random.seed(1337)

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Reshape, Flatten, Dropout, Dense, Input, Embedding, Concatenate
from keras.models import Model, Sequential
from keras.preprocessing import sequence


class CNN_module():
  '''
  classdocs
  '''
  batch_size = 128
  # More than this epoch cause easily over-fitting on our data sets
  nb_epoch = 5

  def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W=None):

    self.max_len = max_len
    max_features = vocab_size
    vanila_dimension = 200

    filter_lengths = [3, 4, 5]

    '''Embedding Layer'''
    model_input = Input(name='input', shape=(max_len,), dtype='int32')

    if init_W is None:
      seq_emb = Embedding(input_dim=max_features, output_dim=emb_dim,
                            input_length=max_len, name='sentence_embeddings')(model_input)
    else:
      seq_emb = Embedding(input_dim=max_features, output_dim=emb_dim, input_length=max_len,
                            weights=[init_W / 20], name='sentence_embeddings')(model_input)

    '''Reshape Layer'''
    reshape = Reshape(target_shape=(max_len, emb_dim, 1), name='reshape')(seq_emb)  # chanels last

    '''Convolution Layer & Max Pooling Layer'''
    self.convs = []
    for i in filter_lengths:
      model_internal = Sequential()
      model_internal.add(Convolution2D(filters=nb_filters, kernel_size=(i, emb_dim), activation="relu"))
      model_internal.add(MaxPooling2D(pool_size=(self.max_len - i + 1, 1)))
      model_internal.add(Flatten())
      model_internal = model_internal(reshape)
      self.convs.append(model_internal)

    model_output = Concatenate(axis=-1)(self.convs)

    '''Dropout Layer'''
    model_output = Dense(vanila_dimension, activation='tanh', name='fully_connect')(model_output)
    model_output = Dropout(dropout_rate, name='dropout')(model_output)

    '''Projection Layer & Output Layer'''
    model_output = Dense(output_dimesion, activation='tanh', name='output')(model_output)

    # Output Layer
    self.model = Model(inputs=model_input, outputs=model_output)
    self.model.compile(optimizer='rmsprop', loss='mse')

  def load_model(self, model_path):
    self.model.load_weights(model_path)

  def save_model(self, model_path, isoverwrite=True):
    self.model.save_weights(model_path, isoverwrite)

  def train(self, X_train, V, item_weight, seed):
    X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
    np.random.seed(seed)
    X_train = np.random.permutation(X_train)
    np.random.seed(seed)
    V = np.random.permutation(V)
    np.random.seed(seed)
    item_weight = np.random.permutation(item_weight)

    print("Train...CNN module")
    history = self.model.fit(x=X_train, y=V,
                             verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch,
                             sample_weight={'output': item_weight})

    # cnn_loss_his = history.history['loss']
    # cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
    # if cnn_loss_his != cmp_cnn_loss:
    #     self.nb_epoch = 1
    return history

  def get_projection_layer(self, X_train):
    X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
    Y = self.model.predict(X_train, batch_size=len(X_train))
    return Y
