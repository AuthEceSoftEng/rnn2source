import numpy as np
import pickle
import logging
import argparse
import time

from keras.layers import Input, LSTM, TimeDistributed, Dense, merge, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from utils import labeled_batch_generator

parser = argparse.ArgumentParser(description='Train the labeled-char-rnn model')
parser.add_argument('-r', '--recovery', type=str, default='', help='filepath to model to recover training from')
args = parser.parse_args()
path_to_model = args.recovery

# Logger init
logging.basicConfig(filename='../data/logs/labeled-char-rnn.log', level=logging.INFO)

# Hyperparameters
SEQ_LEN = 100
BATCH_SIZE = 100
LSTM_SIZE = 1024
LAYERS = 3
NUM_EPOCHS = 80

# Data loading
with open('../data/chars', 'rb') as f:
    minified_data = pickle.load(f)
with open('../data/labels', 'rb') as f:
    label_data = pickle.load(f)

splitPoint = int(np.ceil(len(minified_data) * 0.95))
train_minified_data = ''.join(minified_data[:splitPoint])
test_minified_data = ''.join(minified_data[splitPoint:])
train_label_data = ''.join(label_data[:splitPoint])
test_label_data = ''.join(label_data[splitPoint:])

# TODO: Move these before the splitting
char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(train_minified_data + test_minified_data))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)
lbl_to_idx = {lb: i for (i, lb) in enumerate(sorted(list(set(train_label_data + test_label_data))))}
idx_to_lbl = {i: lb for (lb, i) in lbl_to_idx.items()}
label_size = len(lbl_to_idx)
print 'Working on %d characters (%d unique).' % (len(train_minified_data + test_minified_data), vocab_size)

char_input = Input(batch_shape=(BATCH_SIZE, SEQ_LEN, vocab_size), name='char_input')
label_input = Input(batch_shape=(BATCH_SIZE, SEQ_LEN, label_size), name='label_input')
x = merge([char_input, label_input], mode='concat', concat_axis=-1)

lstm_layer = LSTM(LSTM_SIZE, return_sequences=True, stateful=True)(x)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(LSTM_SIZE, return_sequences=True, stateful=True)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(LSTM_SIZE, return_sequences=True, stateful=True)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)

char_output = TimeDistributed(Dense(vocab_size, activation='softmax'), name='char_output')(lstm_layer)
label_output = TimeDistributed(Dense(label_size, activation='softmax'), name='label_output')(lstm_layer)

model = Model([char_input, label_input], [char_output, label_output])
rms = RMSprop(lr=0.001, clipvalue=5)
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'],  loss_weights=[1., 0.2])
print model.summary()

# Monitored parameters initialization
starting_epoch = 0
avg_train_loss = 0
avg_train_loss2 = 0
avg_train_loss1 = 0
avg_train_acc1 = 0
avg_train_acc2 = 0
avg_test_loss = 0
avg_test_loss1 = 0
avg_test_loss2 = 0
avg_test_acc1 = 0
avg_test_acc2 = 0
prev_loss = 100

if path_to_model:
    starting_epoch = int(path_to_model[-4]) # Conventionally take the number before the extension as an epoch to start
    model.load_weights('../data/results/labeled_char_rnn-%d.h5' % starting_epoch) # This is stupid

# for i, (x1, y1, x2, y2) in enumerate(labeled_batch_generator(train_minified_data, train_label_data)):
#     (loss, _, _, accuracy, _) = model.test_on_batch([x1, x2], [y1, y2])
#     avg_test_loss += loss
#     avg_test_acc += accuracy
# avg_test_loss /= (i + 1)
# avg_test_acc /= (i + 1)
# print avg_test_acc
# print avg_test_loss

print 'Built model, starting training.'
logging.info('Epochs\tTrain loss\tTrain loss1\tTrain loss2 \tTest loss\tTest loss1\tTest loss2\t'
             'Train acc1\ttest acc1\tTrain acc2\ttest acc2')
for epoch in range((starting_epoch + 1), NUM_EPOCHS):
    t1 = time.time()
    model.reset_states()
    for i, (x1, y1, x2, y2) in enumerate(labeled_batch_generator(train_minified_data, train_label_data, lbl_to_idx,
                                                                 char_to_idx, BATCH_SIZE, SEQ_LEN, vocab_size,
                                                                 label_size)):
        (loss, loss1, loss2, accuracy1, accuracy2) = model.train_on_batch([x1, x2], [y1, y2])
        avg_train_loss += loss
        avg_train_loss1 += loss1
        avg_train_loss2 += loss2
        avg_train_acc1 += accuracy1
        avg_train_acc2 += accuracy2
    avg_train_loss /= (i + 1)
    avg_train_loss1 /= (i + 1)
    avg_train_loss2 /= (i + 1)
    avg_train_acc1 /= (i + 1)
    avg_train_acc2 /= (i + 1)
    t2 = time.time()
    print "Epoch %i took %f minutes." % (epoch, ((t2 - t1)/60))

    model.reset_states()
    for i, (x1, y1, x2, y2) in enumerate(labeled_batch_generator(test_minified_data, test_label_data, lbl_to_idx,
                                                                 char_to_idx, BATCH_SIZE, SEQ_LEN, vocab_size,
                                                                 label_size)):
        (loss, loss1, loss2, accuracy1, accuracy2) = model.test_on_batch([x1, x2], [y1, y2])
        avg_test_loss += loss
        avg_test_loss1 += loss1
        avg_test_loss2 += loss2
        avg_test_acc1 += accuracy1
        avg_test_acc2 += accuracy2
    avg_test_loss /= (i + 1)
    avg_test_loss1 /= (i + 1)
    avg_test_loss2 /= (i + 1)
    avg_test_acc1 /= (i + 1)
    avg_test_acc2 /= (i + 1)

    model.save_weights('../data/results/labeled_char_rnn-%d.h5' % epoch)
    print 'Epoch: %d.\tAverage train loss is: %f\tAverage test loss is: %f.' % (epoch, avg_train_loss, avg_test_loss)
    logging.info('%d,\t%f,\t%f,\t%f,\t%f\t%f,\t%f,\t%f,\t%f,\t%f,\t%f', epoch, avg_train_loss, avg_train_loss1,
                 avg_train_loss2, avg_test_loss, avg_test_loss1, avg_test_loss2, avg_train_acc1, avg_test_acc1,
                 avg_train_acc2, avg_test_acc2)

    if (prev_loss - avg_test_loss) < 0.001:
        print 'Warning: Early stopping advised.'
        logging.warning('Early stopping advised.')
    prev_loss = avg_test_loss
