import numpy as np
import os
import time
from pygments.lexers.javascript import JavascriptLexer

from jsmin import jsmin
from linguist.libs.file_blob import FileBlob

from keras.layers import Activation, Dropout, TimeDistributed, Dense, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop


def build_model(infer, lstm_size, batch_size, seq_len, layers, vocab):
    if infer:
        batch_size = seq_len = 1
    model = Sequential()
    model.add(LSTM(lstm_size,
                   return_sequences=True,
                   batch_input_shape=(batch_size, seq_len, vocab),
                   stateful=True))

    model.add(Dropout(0.2)) # TODO: Consider changing this to 0 if infer is true
    for l in range(layers - 1):
        model.add(LSTM(lstm_size, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(vocab)))
    model.add(Activation('softmax'))
    rms = RMSprop(clipvalue=5, lr=0.0015)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
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
