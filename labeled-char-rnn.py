from keras.layers import Input, LSTM, TimeDistributed, Dense, merge, Dropout
from keras.models import Model

vocab_size = 200
seq_len = 25
label_size = 10
batch_size = 50
lstm_size = 512

char_input = Input(batch_shape=(batch_size, seq_len, vocab_size), name='char_input')
label_input = Input(batch_shape=(batch_size, seq_len, label_size), name='label_input')
x = merge([char_input, label_input], mode='concat', concat_axis=-1)  # checkif concat actually works as expected

lstm_layer = LSTM(lstm_size, return_sequences=True, stateful=True)(x)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(lstm_size, return_sequences=True, stateful=True)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(lstm_size, return_sequences=True, stateful=True)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)

char_output = TimeDistributed(Dense(vocab_size, activation='softmax'), name='char_output')(lstm_layer)
label_output = TimeDistributed(Dense(label_size, activation='softmax'), name='label_output')(lstm_layer)

model = Model([char_input, label_input], [char_output, label_output])
model.summary()
