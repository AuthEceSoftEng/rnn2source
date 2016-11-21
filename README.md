# rnn2source
This code implements javascript source code generation using deep recurrent neural networks with LSTM cells. Currently 2 approaches are available:

- Character level learning
This approach is based on Andrej Karpathy's [char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). It is modified to work with javascript repositories.

- Labeled Character learning
The first proposed approach consists of tagging the characters based on a javascript parser to categorize them in 8 simple classes. The model in this case takes both the character and its label as input and makes predictions for the next ones.

# Requirements

# Usage

# Licence
MIT
