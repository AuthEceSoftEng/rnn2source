import numpy as np
import pickle
import logging

from keras.layers import Input, LSTM, TimeDistributed, Dense, merge, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from math import ceil

# Logger init
logging.basicConfig(filename='/home/vasilis/Dropbox/My stuff/Thesis/logs/rnn.log', level=logging.INFO)

# Hyperparameters
SEQ_LEN = 50
BATCH_SIZE = 200
LSTM_SIZE = 512
LAYERS = 3
NUM_EPOCHS = 40


def batch_generator(data, labels):
    T1 = np.asarray([char_to_idx[c] for c in data], dtype=np.int32)
    T2 = np.asarray([lbl_to_idx[l] for l in labels], dtype=np.int32)

    X1 = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))
    Y1 = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))
    X2 = np.zeros((BATCH_SIZE, SEQ_LEN, label_size))
    Y2 = np.zeros((BATCH_SIZE, SEQ_LEN, label_size))

    batch_chars = len(data) / BATCH_SIZE

    for i in range(0, batch_chars - SEQ_LEN - 1, SEQ_LEN):
        X1[:] = 0
        Y1[:] = 0
        X2[:] = 0
        Y2[:] = 0

        for batch_idx in range(BATCH_SIZE):
            start = batch_idx * batch_chars + i
            for j in range(SEQ_LEN):
                X1[batch_idx, j, T1[start + j]] = 1
                Y1[batch_idx, j, T1[start + j + 1]] = 1
                X2[batch_idx, j, T2[start + j]] = 1
                Y2[batch_idx, j, T2[start + j + 1]] = 1
        yield X1, Y1, X2, Y2

print 'Reading data...'
with open('../data/chars', 'rb') as f:
    minified_data = pickle.load(f)
with open('../data/labels', 'rb') as f:
    label_data = pickle.load(f)

splitPoint = int(ceil(len(minified_data) * 0.95))
train_minified_data = ''.join(minified_data[:splitPoint])
test_minified_data = ''.join(minified_data[splitPoint:])
train_label_data = ''.join(label_data[:splitPoint])
test_label_data = ''.join(label_data[splitPoint:])

char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(train_minified_data + test_minified_data))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)
lbl_to_idx = {lb: i for (i, lb) in enumerate(sorted(list(set(train_label_data + test_label_data))))}
idx_to_lbl = {i: lb for (lb, i) in lbl_to_idx.items()}
label_size = len(lbl_to_idx)

char_input = Input(batch_shape=(BATCH_SIZE, SEQ_LEN, vocab_size), name='char_input')
label_input = Input(batch_shape=(BATCH_SIZE, SEQ_LEN, label_size), name='label_input')
x = merge([char_input, label_input], mode='concat', concat_axis=-1)  # checkif concat actually works as expected

lstm_layer = LSTM(LSTM_SIZE, return_sequences=True, stateful=True)(x)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(LSTM_SIZE, return_sequences=True, stateful=True)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(LSTM_SIZE, return_sequences=True, stateful=True)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)

char_output = TimeDistributed(Dense(vocab_size, activation='softmax'), name='char_output')(lstm_layer)
label_output = TimeDistributed(Dense(label_size, activation='softmax'), name='label_output')(lstm_layer)

model = Model([char_input, label_input], [char_output, label_output])
rms = RMSprop(lr=0.002, clipvalue=5)
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'],  loss_weights=[1., 0.2])
model.summary()

starting_epoch = 0
avg_train_loss = 0
avg_train_acc = 0
avg_test_loss = 0
avg_test_acc = 0
prev_loss = 10

recovery = False
if recovery:
    pass

for epoch in range(NUM_EPOCHS):
    for i, (x1, y1, x2, y2) in enumerate(batch_generator(train_minified_data, train_label_data)):
        (loss, loss1, _, accuracy, _) = model.train_on_batch([x1, x2], [y1, y2])
        avg_train_loss += loss
        avg_train_acc += accuracy
    avg_train_loss /= (i + 1)
    avg_train_acc /= (i + 1)

    for i, (x1, y1, x2, y2) in enumerate(batch_generator(test_minified_data, test_label_data)):
        (loss, _, _, accuracy, _) = model.test_on_batch([x1, x2], [y1, y2])
        avg_test_loss += loss
        avg_test_acc += accuracy
    avg_test_loss /= (i + 1)
    avg_test_acc /= (i + 1)

    model.save_weights('../data/results/run13-%d-%f-%f.h5' % (epoch, avg_train_loss, avg_test_loss))
    print 'Epoch: %d.\tAverage train loss is: %f\tAverage test loss is: %f.' % (epoch, avg_train_loss, avg_test_loss)
    logging.info('Epoch: %d\nAvg train loss: %f\tAvg test loss: %f\tAvg train acc: %f \tAvg test acc: %f',
                 epoch, avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc)
    # sample(epoch, avg_train_loss, avg_test_loss)

    if (prev_loss - avg_train_loss) < 0.001:
        print 'Warning: Early stopping advised.'
        logging.warning('Warning: Early stopping advised.')
    prev_loss = avg_train_loss
