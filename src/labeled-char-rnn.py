import numpy as np
import pickle

from keras.layers import Input, LSTM, TimeDistributed, Dense, merge, Dropout
from keras.models import Model


def read_batches(text, c_to_i, size):
    T = np.asarray([c_to_i[c] for c in text], dtype=np.int32)

    X = np.zeros((batch_size, seq_len, size))
    Y = np.zeros((batch_size, seq_len, size))

    for i in range(0, batch_size - seq_len - 1, seq_len):

        X[:] = 0
        Y[:] = 0
        for batch_idx in range(batch_size):
            start = batch_idx * batch_chars + i
            for j in range(seq_len):
                X[batch_idx, j, T[start + j]] = 1
                Y[batch_idx, j, T[start + j + 1]] = 1
        yield X, Y


print 'Reading data...'
with open('chars', 'rb') as f:
    minified_data = pickle.load(f)
with open('labels', 'rb') as f:
    label_data = pickle.load(f)
minified_data = ''.join(minified_data)
label_data = ''.join(label_data)

char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(minified_data))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)

lbl_to_idx = {lb: i for (i, lb) in enumerate(sorted(list(set(label_data))))}
idx_to_lbl = {i: lb for (lb, i) in lbl_to_idx.items()}
label_size = len(lbl_to_idx)

epochs = 50
seq_len = 50
batch_size = 100
batch_chars = len(minified_data) / batch_size
lstm_size = 256

char_input = Input(batch_shape=(batch_size, seq_len, vocab_size), name='char_input')
label_input = Input(batch_shape=(batch_size, seq_len, label_size), name='label_input')
x = merge([char_input, label_input], mode='concat', concat_axis=-1)  # checkif concat actually works as expected

lstm_layer = LSTM(lstm_size, return_sequences=True, stateful=True)(x)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(lstm_size, return_sequences=True, stateful=True)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(lstm_size, return_sequences=True, stateful=True)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)

char_output = TimeDistributed(Dense(vocab_size, activation='softmax'), name='char_output')(lstm_layer)
label_output = TimeDistributed(Dense(label_size, activation='softmax'), name='label_output')(lstm_layer)

model = Model([char_input, label_input], [char_output, label_output])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.summary()

total_loss = 0
for nepoch in range(epochs):
    prev_loss = total_loss
    total_loss = 0
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(read_batches(minified_data, char_to_idx, vocab_size),
                                                 read_batches(label_data, lbl_to_idx, label_size))):
        loss = model.train_on_batch([x1, x2], [y1, y2])
        print loss
