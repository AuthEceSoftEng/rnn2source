import numpy as np

from keras.models import load_model

from utils import jsparser

# data I/O
data = jsparser('/home/vasilis/Documents/projects')
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
model = load_model('data/results/stateful-95-0.718849)')
model.reset_states()

primer_text = 'The meaning of life is:'
sample_chars = 128

sampled = [char_to_ix[c] for c in primer_text]

for c in primer_text:
    batch = np.zeros((1, 1, vocab_size))
    batch[0, 0, char_to_ix[c]] = 1
    model.predict_on_batch(batch)

for i in range(sample_chars):
    batch = np.zeros((1, 1, vocab_size))
    batch[0, 0, sampled[-1]] = 1
    softmax = model.predict_on_batch(batch)[0].ravel()
    sample = np.random.choice(range(vocab_size), p=softmax)
    sampled.append(sample)

print ''.join([ix_to_char[c] for c in sampled])
