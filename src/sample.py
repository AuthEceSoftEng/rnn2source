import argparse
import numpy as np
import pickle
import time

import jsbeautifier

from utils import build_model, temp

parser = argparse.ArgumentParser(description='Sample a trained model')
parser.add_argument('filepath', help='filepath to model')
parser.add_argument('-s', '--seed', help='seed input', type=str, default="")
parser.add_argument('-t', '--temperature', help='set sampling temperature', type=float, default=0.85)
parser.add_argument('-l', '--length', help='set output length', type=int, default=10000)
parser.add_argument('-p', '--project', help='load the test project', default='../data/github_test_chars')
args = parser.parse_args()

path = args.filepath
seed = args.seed
temperature = args.temperature
length = args.length
project_seed_path = args.project
numFilesToCreate = 100

opts = jsbeautifier.default_options()
opts.jslint_happy = True

# Data loading
with open('../data/chars', 'rb') as f:
    minified_data = pickle.load(f)

splitPoint = int(np.ceil(len(minified_data) * 0.95))
train_data = ''.join(minified_data[:splitPoint])
test_data = ''.join(minified_data[splitPoint:])
char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(train_data + test_data))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)

with open('../data/github_test_chars', 'r') as f:
    project_seed = pickle.load(f)

initial_seed = ''.join(project_seed)
initial_seed = initial_seed.replace('\x1b', '\x0a')
missingKeys = set(initial_seed) - set(char_to_idx)

print 'Working on %d characters (%d unique).' % (len(train_data + test_data), vocab_size)
model = build_model(True, 1024, 1, 1, 3, vocab_size)
model.load_weights(path)
model.reset_states()

start_time = time.time()
for c in [char_to_idx[c] for c in initial_seed]:
    batch = np.zeros((1, 1, vocab_size))
    batch[0, 0, c] = 1
    model.predict_on_batch(batch)

print("--- %s seconds ---" % (time.time() - start_time))
sampled = [char_to_idx[c] for c in seed]

for c in seed:
    batch = np.zeros((1, 1, vocab_size))
    batch[0, 0, char_to_idx[c]] = 1
    model.predict_on_batch(batch)
for i in range(numFilesToCreate):
    sampled.append(0)
    while True:
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, sampled[-1]] = 1
        softmax = model.predict_on_batch(batch)[0].ravel()
        sample = np.random.choice(range(vocab_size), p=temp(softmax, temperature))
        sampled.append(sample)
        if sample == 1:
            text = ''.join([idx_to_char[c] for c in sampled[1:-1]])
            with open("../data/sampledCode/temptest%i.js" % i, "w") as produced_file:
                produced_file.write(jsbeautifier.beautify(text, opts))
                print 'printed file'
            sampled[:] = []
            break
        if len(sampled) > 15000:
            batch = np.zeros((1, 1, vocab_size))
            batch[0, 0, 1] = 1
            model.predict_on_batch(batch)
            text = ''.join([idx_to_char[c] for c in sampled[1:]])
            with open("../data/sampledCode/temptest%i.js" % i, "w") as produced_file:
                produced_file.write(jsbeautifier.beautify(text, opts))
            print 'that\'s too much - printed file nonetheless'
            sampled[:] = []
            break
