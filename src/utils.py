import numpy as np
import os
import time

from jsmin import jsmin
from keras.layers import Activation, Dropout, TimeDistributed, Dense, LSTM, Input, merge
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from linguist.libs.file_blob import FileBlob
from pygments.lexers.javascript import JavascriptLexer


def build_model(infer, lstm_size, batch_size, seq_len, layers, vocab):
    if infer:
        batch_size = seq_len = 1
    model = Sequential()
    model.add(LSTM(lstm_size,
                   return_sequences=True,
                   batch_input_shape=(batch_size, seq_len, vocab),
                   stateful=True))

    model.add(Dropout(0.4))  # TODO: Consider changing this to 0 if infer is true
    for l in range(layers - 1):
        model.add(LSTM(lstm_size, return_sequences=True, stateful=True))
        model.add(Dropout(0.4))

    model.add(TimeDistributed(Dense(vocab)))
    model.add(Activation('softmax'))
    rms = RMSprop(clipvalue=5, lr=0.002)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model


def build_labeled_model(lstm_size, batch_size, seq_len, char_vocab_size, lbl_vocab_size):
    char_input = Input(batch_shape=(batch_size, seq_len, char_vocab_size), name='char_input')
    label_input = Input(batch_shape=(batch_size, seq_len, lbl_vocab_size), name='label_input')
    x = merge([char_input, label_input], mode='concat', concat_axis=-1)

    lstm_layer = LSTM(lstm_size, return_sequences=True, stateful=True)(x)
    lstm_layer = Dropout(0.4)(lstm_layer)
    lstm_layer = LSTM(lstm_size, return_sequences=True, stateful=True)(lstm_layer)
    lstm_layer = Dropout(0.4)(lstm_layer)
    lstm_layer = LSTM(lstm_size, return_sequences=True, stateful=True)(lstm_layer)
    lstm_layer = Dropout(0.4)(lstm_layer)

    char_output = TimeDistributed(Dense(char_vocab_size, activation='softmax'), name='char_output')(lstm_layer)
    label_output = TimeDistributed(Dense(lbl_vocab_size, activation='softmax'), name='label_output')(lstm_layer)

    model = Model([char_input, label_input], [char_output, label_output])
    rms = RMSprop(lr=0.001, clipvalue=5)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'], loss_weights=[1., 0.2])
    return model


def sample(preds, temperature=0.35):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def batch_generator(text, char_to_idx, batch_size, seq_len, vocab_size):
    t = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)
    x = np.zeros((batch_size, seq_len, vocab_size))
    y = np.zeros((batch_size, seq_len, vocab_size))
    batch_chars = len(text) / batch_size

    for i in range(0, batch_chars - seq_len - 1, seq_len):

        x[:] = 0
        y[:] = 0
        for batch_idx in range(batch_size):
            start = batch_idx * batch_chars + i
            for j in range(seq_len):
                x[batch_idx, j, t[start + j]] = 1
                y[batch_idx, j, t[start + j + 1]] = 1
        yield x, y


def labeled_batch_generator(data, labels, lbl_to_idx, char_to_idx, batch_size, seq_len, vocab_size, label_size):
    t1 = np.asarray([char_to_idx[c] for c in data], dtype=np.int32)
    t2 = np.asarray([lbl_to_idx[l] for l in labels], dtype=np.int32)

    x1 = np.zeros((batch_size, seq_len, vocab_size))
    y1 = np.zeros((batch_size, seq_len, vocab_size))
    x2 = np.zeros((batch_size, seq_len, label_size))
    y2 = np.zeros((batch_size, seq_len, label_size))

    batch_chars = len(data) / batch_size

    for i in range(0, batch_chars - seq_len - 1, seq_len):
        x1[:] = 0
        y1[:] = 0
        x2[:] = 0
        y2[:] = 0

        for batch_idx in range(batch_size):
            start = batch_idx * batch_chars + i
            for j in range(seq_len):
                x1[batch_idx, j, t1[start + j]] = 1
                y1[batch_idx, j, t1[start + j + 1]] = 1
                x2[batch_idx, j, t2[start + j]] = 1
                y2[batch_idx, j, t2[start + j + 1]] = 1
        yield x1, y1, x2, y2


def jsparser(path):
    print "Reading data..."
    minified_data = ['']
    label_data = []
    t1 = time.time()
    filecounter = 0
    excluded = {'test', 'tests', '__tests__' 'locale', 'locales', 'ngLocale'}
    point = JavascriptLexer()

    for root, dirs, files in os.walk(path, topdown=True):
        dirs[:] = [d for d in dirs if d not in excluded]  # exclude test directories
        for name in files:
            if name.endswith(".js"):
                blob = FileBlob(os.path.join(root, name))  # Linguist file checking
                if not (blob.is_binary or blob.is_generated):
                    filecounter += 1
                    with open(os.path.join(root, name)) as js_file:
                        data = js_file.read()
                        minidata = '\xff' + jsmin(data) + '\xfe'
                        labels = []
                        for token in point.get_tokens_unprocessed(minidata):
                            (index, label, seq) = token
                            for i in range(len(seq)):
                                labels.append(label)
                        minified_data.append(minidata)
                        label_data.append(labels)
    minified_data = ''.join(minified_data)

    t2 = time.time()
    print "Created the dataset in: %f milliseconds from %d files" % ((t2 - t1) * 1000., filecounter)

    chars = list(set(minified_data))
    data_size, vocab_size = len(minified_data), len(chars)
    print 'data has %d characters, %d unique.' % (data_size, vocab_size)

    return minified_data


def temp(softmax, temp):
    softmax = np.log(softmax) / temp
    softmaxT = np.exp(softmax) / np.sum(np.exp(softmax))
    return softmaxT
