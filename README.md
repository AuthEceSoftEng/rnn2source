# rnn2source
This code implements javascript source code generation using deep recurrent neural networks(RNN) with Long Short-Term Memory(LSTM) cells. In a nutshell, our model takes text files of source code, minifies and "reads" them; then after being trained, generates sequences of source code. Currently 2 approaches are available:

- Character Level Learning:
This approach is based on Andrej Karpathy's [char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). It is modified to work with javascript repositories.

- Labeled Character Learning:
The first proposed approach consists of categorizing each character as one of eight simple classes(regex, keyword, string, number, operator, punctuator, identifier, other) . The model in this case takes both the character and its label as input and makes predictions for the next set.

# Requirements
This code is written in Python 2.7 using Keras with Theano as a backend.

[Installing Theano](http://deeplearning.net/software/theano/install.html)

[Installing Keras](http://deeplearning.net/software/theano/install.html)

As deep Recurrent Neural Networks are very expensive computationally you are advised to use a GPU to train them.

[Using Theano with the GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html)

Various libraries are used to manipulate & evaluate the source code. If you have [pip](https://pip.pypa.io/en/stable/installing/) installed you can try installing them with a single command:

```
pip install -r requirements.txt
```

# Usage

# Licence
MIT
