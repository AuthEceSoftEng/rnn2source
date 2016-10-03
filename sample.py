from keras.models import load_model
import argparse
import sys
import numpy as np



parser = argparse.ArgumentParser(description='Sample a trained model')
parser.add_argument('filepath', help='filepath to model')
parser.add_argument('seed', help='seed input')
parser.add_argument('-t', '--temperature', help='set samping temperature', type=float, default=0.2)
parser.add_argument('-l', '--length', help='set sample output length', type=int, default=10)
args = parser.parse_args()

path = args.filepath
seed = args.seed
temperature = args.temperature
length = args.length

model = load_model(path)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# data I/O
data = open('/home/vasilis/PycharmProjects/lstm-keras/data/input.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
seq_length = 50


generated = ''
generated += seed
print('----- Generating with seed: "' + seed + '"')
sys.stdout.write(generated)

for i in range(length):
    x = np.zeros((1, seq_length, len(chars)))
    for t, char in enumerate(seed):
        x[0, t, char_to_ix[char]] = 1.

    preds = model.predict(x, verbose=1)[0]
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    next_index = np.argmax(np.random.multinomial(1, preds, 1))
    next_char = ix_to_char[next_index]

    generated += next_char
    seed = seed[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()

