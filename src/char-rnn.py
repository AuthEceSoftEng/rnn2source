import numpy as np
import pickle
import logging

from keras.layers import Activation, Dropout, TimeDistributed, Dense, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from math import ceil

# Logger init
logging.basicConfig(filename='/home/vasilis/Dropbox/My stuff/Thesis/logs/rnn.log', level=logging.INFO)

# Hyperparameters
SEQ_LEN = 50
BATCH_SIZE = 200
LSTM_SIZE = 512
LAYERS = 3
NUM_EPOCHS = 40

# Data loading
with open('../data/chars', 'rb') as f:
    minified_data = pickle.load(f)

splitPoint = int(ceil(len(minified_data) * 0.95))
train_data = ''.join(minified_data[:splitPoint])
test_data = ''.join(minified_data[splitPoint:])
char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(train_data + test_data))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)
print 'Working on %d characters (%d unique).' % (len(train_data + test_data), vocab_size)


def batch_generator(text):
    t = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)
    x = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))
    y = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))
    batch_chars = len(text) / BATCH_SIZE

    for i in range(0, batch_chars - SEQ_LEN - 1, SEQ_LEN):

        x[:] = 0
        y[:] = 0
        for batch_idx in range(BATCH_SIZE):
            start = batch_idx * batch_chars + i
            for j in range(SEQ_LEN):
                x[batch_idx, j, t[start + j]] = 1
                y[batch_idx, j, t[start + j + 1]] = 1
        yield x, y


def build_model(infer):
    if infer:
        batch_size = seq_len = 1
    else:
        batch_size = BATCH_SIZE
        seq_len = SEQ_LEN
    model = Sequential()
    model.add(LSTM(LSTM_SIZE,
                   return_sequences=True,
                   batch_input_shape=(batch_size, seq_len, vocab_size),
                   stateful=True))

    model.add(Dropout(0.2))
    for l in range(LAYERS - 1):
        model.add(LSTM(LSTM_SIZE, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    rms = RMSprop(clipvalue=5, lr=0.002)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


print 'Building model.'
training_model = build_model(infer=False)
test_model = build_model(infer=True)
print '... done.'
# print training_model.summary()


def sample(epoch, train_loss, test_loss, sample_chars=256, primer_text='And the '):
    test_model.reset_states()
    test_model.load_weights('../data/results/run12-%d-%f-%f.h5' % (epoch, train_loss, test_loss))
    sampled = [char_to_idx[c] for c in primer_text]

    for c in primer_text:
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, char_to_idx[c]] = 1
        test_model.predict_on_batch(batch)

    for i in range(sample_chars):
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, sampled[-1]] = 1
        softmax = test_model.predict_on_batch(batch)[0].ravel()
        sample = np.random.choice(range(vocab_size), p=softmax)
        sampled.append(sample)

    print ''.join([idx_to_char[c] for c in sampled])

starting_epoch = 0
avg_train_loss = 0
avg_train_acc = 0
avg_test_loss = 0
avg_test_acc = 0
prev_loss = 5

recovery = False
if recovery:
    starting_epoch = 9
    avg_train_loss = 1.095104
    avg_test_loss = 0.891405
    training_model.load_weights('../data/results/run12-%d-%f-%f.h5' % (starting_epoch, avg_train_loss, avg_test_loss))


for epoch in range((starting_epoch + 1), NUM_EPOCHS):
    for i, (x, y) in enumerate(batch_generator(train_data)):
        if i % 200:
            training_model.reset_states()
        loss, accuracy = training_model.train_on_batch(x, y)
        avg_train_loss += loss
        avg_train_acc += accuracy
    avg_train_loss /= (i + 1)
    avg_train_acc /= (i + 1)

    for i, (x, y) in enumerate(batch_generator(test_data)):
        loss, accuracy = training_model.test_on_batch(x, y)
        avg_test_loss += loss
        avg_test_acc += accuracy
    avg_test_loss /= (i + 1)
    avg_test_acc /= (i + 1)

    training_model.save_weights('../data/results/run12-%d-%f-%f.h5' % (epoch, avg_train_loss, avg_test_loss))
    print 'Epoch: %d.\tAverage train loss is: %f\tAverage test loss is: %f.' % (epoch, avg_train_loss, avg_test_loss)
    logging.info('Epoch: %d\nAvg train loss: %f\tAvg test loss: %f\tAvg train acc: %f \tAvg test acc: %f',
                 epoch, avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc)
    sample(epoch, avg_train_loss, avg_test_loss)

    if (prev_loss - avg_train_loss) < 0.001:
        print 'Warning: Early stopping advised.'
        logging.warning('Warning: Early stopping advised.')
    prev_loss = avg_train_loss
