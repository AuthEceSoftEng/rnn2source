import argparse
import numpy as np
import pickle

from utils import jsparser, sample, build_model

# TODO: Add temperature and file control
parser = argparse.ArgumentParser(description='Sample a trained model')
parser.add_argument('filepath', help='filepath to model')
parser.add_argument('-s', '--seed', help='seed input', type=str, default="export function getAttachContext(spec:document,update:'mount: data:yield:any'){destructuring=function(rawDocs){buffer._onAssignment();return")
parser.add_argument('-t', '--temperature', help='set sampling temperature', type=float, default=0.2)
parser.add_argument('-l', '--length', help='set output length', type=int, default=1000)
args = parser.parse_args()

path = args.filepath
seed = args.seed
temperature = args.temperature
length = args.length

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
# print seed
# print char_to_idx
# print idx_to_char
# print model.get_weights()[0]
model = build_model(True, 1024, 1, 1, 3, vocab_size)
model.load_weights(path)
model.reset_states()

sampled = [char_to_idx[c] for c in seed]

for c in seed:
    batch = np.zeros((1, 1, vocab_size))
    batch[0, 0, char_to_idx[c]] = 1
    model.predict_on_batch(batch)
for i in range(5):
    batch = np.zeros((1, 1, vocab_size))
    batch[0, 0, sampled[-1]] = 1
    softmax = model.predict_on_batch(batch)[0].ravel() # TODO: Check what ravel is
    sample = np.random.choice(range(vocab_size), p=softmax)
    sampled.append(sample)
print ''.join([idx_to_char[c] for c in sampled])
# for c in sampled:
#     print c