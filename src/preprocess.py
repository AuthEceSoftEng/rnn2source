import os
import pickle
import time
from random import shuffle

from jsmin import jsmin
from linguist.libs.file_blob import FileBlob
from pygments.lexers.javascript import JavascriptLexer
from pygments.token import Token

path = '/home/vasilis/Desktop/lodash-master'  # TODO: Use argparse to get that
# path = '/home/vasilis/Documents/projects/mbostock-d3-b516d77/src/geo'
print "Reading data..."

t1 = time.time()
minified_data = []
label_data = []
excluded = {'test', 'tests', '__tests__' 'locale', 'locales', 'ngLocale'}
typoes = {Token.Literal.String.Regex: 'r', Token.Keyword: 'k', Token.Literal.String: 's',
          Token.Punctuation: 'p', Token.Literal.Number: 'n', Token.Operator: 'o', Token.Text: 'p',
          Token.Name: 'i'}
point = JavascriptLexer()

for root, dirs, files in os.walk(path, topdown=True):
    dirs[:] = [d for d in dirs if d not in excluded]  # exclude test directories
    for name in files:
        if name.endswith(".js"):
            blob = FileBlob(os.path.join(root, name))  # Linguist file checking
            if not (blob.is_binary or blob.is_generated):
                with open(os.path.join(root, name)) as js_file:
                    minidata = jsmin(js_file.read())

                labels = []
                chars = []
                for (_, typo, seq) in point.get_tokens_unprocessed(minidata):
                    # print typo, seq
                    chars.append(seq)
                    tag = typoes.get(typo)
                    if not tag:
                        tag = typoes.get(typo.split()[-2], 'e')
                    labels.extend(tag for i in range(len(seq)))

                labels = 'p' + ''.join(labels) + 'p'  # Add start/end special characters
                chars = '\x01' + ''.join(chars) + '\x02'

                if len(chars) != len(labels):
                    print 'wtf', len(chars), len(labels)  # TODO: Clean up
                    print os.path.join(root, name)
                if chars not in minified_data:
                    minified_data.append(chars)
                    label_data.append(labels)

# Create a shuffled dataset
minified_data_shuf = []
label_data_shuf = []
index_shuf = range(len(minified_data))
shuffle(index_shuf)
for index in index_shuf:
    minified_data_shuf.append(minified_data[index])
    label_data_shuf.append(label_data[index])

# Save files
with open('../data/npm_chars_shuf', 'wb') as f:
    pickle.dump(minified_data_shuf, f)

with open('../data/npm_labels_shuf', 'wb') as f:
    pickle.dump(label_data_shuf, f)

with open('../data/npm_test_chars', 'wb') as f:
    pickle.dump(minified_data, f)

with open('../data/npm_test_labels', 'wb') as f:
    pickle.dump(label_data, f)

minified_data = ''.join(minified_data)
label_data = ''.join(label_data)
minified_data_shuf = ''.join(minified_data_shuf)
label_data_shuf = ''.join(label_data_shuf)

t2 = time.time()
chars = sorted(list(set(minified_data)))
data_size, vocab_size = len(minified_data), len(chars)
print "Created the dataset in: %f milliseconds." % ((t2 - t1) * 1000.)
print 'Data has %d characters, %d unique.' % (data_size, vocab_size)
