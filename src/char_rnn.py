import numpy as np
import pickle
import logging
import argparse
import time

from utils import batch_generator, build_model


def sample_during_training(ep, dict, sample_chars=256, primer_text='And the '):
    test_model.reset_states()
    test_model.load_weights('../data/results/char_rnn-%d.h5' % ep)
    sampled = [dict[c] for c in primer_text]

    for c in primer_text:
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, dict[c]] = 1
        test_model.predict_on_batch(batch)
    for i in range(sample_chars):
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, sampled[-1]] = 1
        softmax = test_model.predict_on_batch(batch)[0].ravel()
        sample = np.random.choice(range(vocab_size), p=softmax)
        sampled.append(sample)
    print ''.join([idx_to_char[c] for c in sampled])


parser = argparse.ArgumentParser(description='Train the char-rnn model')
parser.add_argument('-r', '--recovery', type=str, default='', help='filepath to model to recover training from')
args = parser.parse_args()
path_to_model = args.recovery

# Logger init
logging.basicConfig(filename='../data/logs/char-rnn.log', level=logging.INFO)

# Hyperparameters
SEQ_LEN = 100
BATCH_SIZE = 130
LSTM_SIZE = 1024
LAYERS = 3
NUM_EPOCHS = 80

# Data loading
with open('../data/chars', 'rb') as f:
    minified_data = pickle.load(f)

splitPoint = int(np.ceil(len(minified_data) * 0.95))
train_data = ''.join(minified_data[:splitPoint])
test_data = ''.join(minified_data[splitPoint:])
char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(train_data + test_data))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)
print 'Working on %d characters (%d unique).' % (len(train_data + test_data), vocab_size)

training_model = build_model(False, LSTM_SIZE, BATCH_SIZE, SEQ_LEN, LAYERS, vocab_size)
test_model = build_model(True, LSTM_SIZE, BATCH_SIZE, SEQ_LEN, LAYERS, vocab_size)
print training_model.summary()

starting_epoch = 0
avg_train_loss = 0
avg_train_acc = 0
avg_test_loss = 0
avg_test_acc = 0
prev_loss = 100

if path_to_model:
    starting_epoch = int(path_to_model[-4]) # Conventionally take the number before the extension as an epoch to start
    training_model.load_weights('../data/results/char_rnn-%d.h5' % starting_epoch)

print 'Built model, starting training.'
logging.info('Epochs \tAgv train loss \tAvg test loss \tAvg train acc \tAvg test acc')
for epoch in range((starting_epoch + 1), NUM_EPOCHS):
    t1 = time.time()
    training_model.reset_states()
    for i, (x, y) in enumerate(batch_generator(train_data, char_to_idx, BATCH_SIZE, SEQ_LEN, vocab_size)):
        loss, accuracy = training_model.train_on_batch(x, y)
        avg_train_loss += loss
        avg_train_acc += accuracy
    avg_train_loss /= (i + 1)
    avg_train_acc /= (i + 1)
    t2 = time.time()
    print "Epoch %i took %f minutes." % ((epoch), ((t2 - t1)/60))

    for i, (x, y) in enumerate(batch_generator(test_data, char_to_idx, BATCH_SIZE, SEQ_LEN, vocab_size)):
        loss, accuracy = training_model.test_on_batch(x, y)
        avg_test_loss += loss
        avg_test_acc += accuracy
    avg_test_loss /= (i + 1)
    avg_test_acc /= (i + 1)

    training_model.save_weights('../data/results/char_rnn-%d.h5' % epoch)
    print 'Epoch: %d.\tAverage train loss is: %f\tAverage test loss is: %f.' % (epoch, avg_train_loss, avg_test_loss)
    logging.info('%d,\t%f,\t%f,\t%f, \t%f', epoch, avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc)
    sample_during_training(epoch, char_to_idx)

    if (prev_loss - avg_train_loss) < 0.001:
        print 'Warning: Early stopping advised.'
        logging.warning('Early stopping advised.')
    prev_loss = avg_train_loss
