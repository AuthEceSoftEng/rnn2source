import argparse
import numpy as np
import random
import sys

from keras.models import load_model

from utils import jsparser, sample

parser = argparse.ArgumentParser(description='Sample a trained model')
parser.add_argument('filepath', help='filepath to model')
parser.add_argument('-s', '--seed', help='seed input', type=str, default=[])
parser.add_argument('-t', '--temperature', help='set sampling temperature', type=float, default=0.2)
parser.add_argument('-l', '--length', help='set output length', type=int, default=100)
args = parser.parse_args()

path = args.filepath
seed = args.seed
temperature = args.temperature
length = args.length

model = load_model(path)
model.summary()
print model.get_weights()[0]
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# data I/O
data = jsparser('/home/vasilis/Documents/projects')
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
seq_length = len(seed)

if not seed:
    start_index = random.randint(0, data_size - seq_length - 1)
    generated = ''
    sentence = data[start_index: start_index + seq_length]
    generated += sentence
    print('no seed given')
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)
else:
    print('----- Generating with seed: "' + seed + '"')
    sentence = seed
    generated = ''
    generated += seed

for i in range(length):
    x = np.zeros((1, seq_length, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_to_ix[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds)
    next_char = ix_to_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()
