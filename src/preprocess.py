import os
import pickle
import time

from jsmin import jsmin
from linguist.libs.file_blob import FileBlob
from pygments.lexers.javascript import JavascriptLexer
from pygments.token import Token


# path = '/home/vasilis/Documents/projects'   # TODO: Use argparse to get that
path = '/home/vasilis/Documents/projects/mbostock-d3-b516d77/src/geo'
print "Reading data..."
t1 = time.time()
minified_data = []
label_data = []
filecounter = 0
excluded = {'test', 'tests', '__tests__' 'locale', 'locales', 'ngLocale'}
point = JavascriptLexer()

for root, dirs, files in os.walk(path, topdown=True):
    dirs[:] = [d for d in dirs if d not in excluded]  # exclude test directories
    for name in files:
        if name.endswith(".js"):
            blob = FileBlob(os.path.join(root, name))  # Linguist file checking
            if not (blob.is_binary or blob.is_generated):
                with open(os.path.join(root, name)) as js_file:
                    minidata = jsmin(js_file.read())
                    # if minidata not in minified_data:
                filecounter += 1
                labels = ''
                chars = ''
                for type, seq in point.get_tokens(minidata):
                    chars += seq
                    if type in Token.Literal.String.Regex:
                        tag = 'r'  # regex
                    elif type in Token.Keyword:
                        tag = 'k'  # keyword
                    elif type in Token.Literal.String:
                        tag = 's'  # string
                    elif type in Token.Literal.Number:
                        tag = 'n'  # number
                    elif type in Token.Operator:
                        tag = 'o'  # operator
                    elif (type in Token.Text) or (type in Token.Punctuation):
                        tag = 'p'  # punctuation
                    elif type is Token.Name:
                        tag = 'i'  # identifier
                    else:
                        tag = 'e'  # others

                    for i in range(len(seq)):
                        # if not len(labels) == len(minidata):    # other, not for,  solution
                        labels += tag   # join instead



                if len(labels) == len(chars):
                    minified_data.append(chars)
                    label_data.append(labels)


with open('chars', 'wb') as f:
    pickle.dump(minified_data, f)

with open('labels', 'wb') as f:
    pickle.dump(label_data, f)

minified_data = ''.join(minified_data)
label_data = ''.join(label_data)

t2 = time.time()
chars = sorted(list(set(minified_data)))
data_size, vocab_size = len(minified_data), len(chars)
print "Created the dataset in: %f milliseconds from %d files." % ((t2 - t1) * 1000., filecounter)
print 'Data has %d characters, %d unique.' % (data_size, vocab_size)
print len(label_data)
# for c, l in zip(minified_data, label_data):
#     print c, l