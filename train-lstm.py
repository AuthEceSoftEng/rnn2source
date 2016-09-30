from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import sys
import h5py

# data I/O
data = open('data/input.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
seq_length = 50
step = 3    # step used to create semi redundant arrays of information

inputs = []
target = []
for i in range(0, data_size - seq_length, step):
    inputs.append(data[i: i + seq_length])
    target.append(data[i + seq_length])
print('Number of samples:', len(inputs))

print('Vectorization...')
X = np.zeros((len(inputs), seq_length, len(chars)), dtype=np.bool)
y = np.zeros((len(inputs), len(chars)), dtype=np.bool)
for i, sentence in enumerate(inputs):
    for t, char in enumerate(sentence):
        X[i, t, char_to_ix[char]] = 1
    y[i, char_to_ix[target[i]]] = 1

# define the checkpoint
filepath = "data/results/weights-improvement-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1)

model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape=(seq_length, vocab_size), return_sequences=True, forget_bias_init='one'))
model.add(LSTM(vocab_size))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def sample(preds, temperature=0.35):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1, callbacks=[checkpoint])

    start_index = random.randint(0, data_size - seq_length - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = data[start_index: start_index + seq_length]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, seq_length, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_to_ix[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = ix_to_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
print()
