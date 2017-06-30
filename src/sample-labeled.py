import argparse
import numpy as np
import pickle
import time

import jsbeautifier
from pygments.lexers.javascript import JavascriptLexer
from pygments.token import Token

from utils import build_labeled_model, temp

typoes = {Token.Literal.String.Regex: 'r', Token.Keyword: 'k', Token.Literal.String: 's',
          Token.Punctuation: 'p', Token.Literal.Number: 'n', Token.Operator: 'o', Token.Text: 'p',
          Token.Name: 'i'}
point = JavascriptLexer()

# TODO: Add temperature
parser = argparse.ArgumentParser(description='Sample a trained model with labels')
parser.add_argument('filepath', help='filepath to model')
parser.add_argument('-s', '--seed', help='seed input', type=str, default='')
parser.add_argument('-t', '--temperature', help='set sampling temperature', type=float, default=0.85)
parser.add_argument('-l', '--length', help='set output length', type=int, default=1000)
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
print path
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

with open('../data/github_test_chars', 'r') as f:
    project_seed = pickle.load(f)
with open('../data/github_test_labels', 'r') as f:
    project_seed_labels = pickle.load(f)

initial_seed = ''.join(project_seed)
initial_seed = initial_seed.replace('\x1b', '\x0a')
initial_seed_labels = ''.join(project_seed_labels)
missingKeys = set(initial_seed) - set(char_to_idx)

print 'Working on %d characters (%d unique).' % (len(train_minified_data + test_minified_data), vocab_size)
model = build_labeled_model(lstm_size=1024, batch_size=1, seq_len=1, char_vocab_size=vocab_size,
                            lbl_vocab_size=label_size)
model.load_weights(path)
model.reset_states()

# TODO: Make this a function because it's repeated
labels = []
for (_, typo, seq) in point.get_tokens_unprocessed(seed):
    tag = typoes.get(typo)
    if not tag:
        tag = typoes.get(typo.split()[-2], 'e')
    labels.extend(tag for i in range(len(seq)))

start_time = time.time()
for c, l in zip([char_to_idx[c] for c in initial_seed], [lbl_to_idx[l] for l in initial_seed_labels]):
    char_in = np.zeros((1, 1, vocab_size))
    char_in[0, 0, c] = 1
    label_in = np.zeros((1, 1, label_size))
    label_in[0, 0, l] = 1
    model.predict_on_batch([char_in, label_in])
print("--- %s seconds ---" % (time.time() - start_time))

sampled = [char_to_idx[c] for c in seed]
sampled_labels = [lbl_to_idx[l] for l in labels]

for c, l in zip(seed, labels):
    char_in = np.zeros((1, 1, vocab_size))
    char_in[0, 0, char_to_idx[c]] = 1
    label_in = np.zeros((1, 1, label_size))
    label_in[0, 0, lbl_to_idx[l]] = 1
    model.predict_on_batch([char_in, label_in])

for i in range(numFilesToCreate):
    sampled.append(0)
    sampled_labels.append(5)
    while True:
        char_in = np.zeros((1, 1, vocab_size))
        char_in[0, 0, sampled[-1]] = 1
        label_in = np.zeros((1, 1, label_size))
        label_in[0, 0, sampled_labels[-1]] = 1
        softmax_char, softmax_label = model.predict_on_batch([char_in, label_in])
        sample = np.random.choice(range(vocab_size), p=temp(softmax_char.ravel(), temperature))
        sample_label = np.random.choice(range(label_size), p=temp(softmax_label.ravel(), temperature))
        sampled.append(sample)
        sampled_labels.append(sample_label)
        if sample == 1:
            text = ''.join([idx_to_char[c] for c in sampled[1:-1]])
            with open("../data/sampledCode/githubsampled%i.js" % i, "w") as produced_file:
                produced_file.write(jsbeautifier.beautify(text, opts))
                print 'printed file'
            sampled[:] = []
            sampled_labels[:] = []
            break
        if len(sampled) > 15000:
            char_in = np.zeros((1, 1, vocab_size))
            char_in[0, 0, 1] = 1
            label_in = np.zeros((1, 1, label_size))
            label_in[0, 0, 5] = 1
            model.predict_on_batch([char_in, label_in])
            text = ''.join([idx_to_char[c] for c in sampled[1:]])
            with open("../data/sampledCode/githubsampled%i.js" % i, "w") as produced_file:
                produced_file.write(jsbeautifier.beautify(text, opts))
            print 'that\'s too much - printed file nonetheless'
            sampled[:] = []
            sampled_labels[:] = []
            break
