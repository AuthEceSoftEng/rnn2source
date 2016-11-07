from keras.layers import Dense, TimeDistributed, LSTM, Dropout
from keras.models import Sequential

from utils import *


def read_batches(text):
    T = np.asarray([char_to_ix[c] for c in text], dtype=np.int32)

    X = np.zeros((batch_size, seq_length, vocab_size))
    Y = np.zeros((batch_size, seq_length, vocab_size))

    for i in range(0, batch_chars - seq_length - 1, seq_length):

        X[:] = 0
        Y[:] = 0
        for batch_idx in range(batch_size):
            start = batch_idx * batch_chars + i
            for j in range(seq_length):
                X[batch_idx, j, T[start + j]] = 1
                Y[batch_idx, j, T[start + j + 1]] = 1
        yield X, Y


# data I/O
data = jsparser('/home/vasilis/Documents/projects')
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
batch_size = 64
seq_length = 64
batch_chars = data_size / batch_size

model = Sequential()
model.add(LSTM(128, activation='tanh', batch_input_shape=(batch_size, seq_length, vocab_size), return_sequences=True,
               forget_bias_init='one', stateful=True))
model.add(Dropout(0.2))
model.add(LSTM(128, stateful=True, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# train the model, output generated text after each iteration
for iteration in range(100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    for i, (x_train, y_train) in enumerate(read_batches(data)):
        loss = model.train_on_batch(x_train, y_train)

        print iteration, i, loss

        if i % 1000 == 0:
            model.save('data/results/stateful-%d-%f)' % (iteration, loss))
