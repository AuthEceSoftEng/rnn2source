import os
import pickle
import time
from pygments.lexers.javascript import JavascriptLexer
from pygments.token import Token

from jsmin import jsmin
from linguist.libs.file_blob import FileBlob

path = '/home/vasilis/Documents/projects'
print "Reading data..."
minified_data = []
label_data = []
t1 = time.time()
filecounter = 0
excluded = {'test', 'tests', '__tests__' 'locale', 'locales', 'ngLocale'}
point = JavascriptLexer()

print type(label_data)
for root, dirs, files in os.walk(path, topdown=True):
    dirs[:] = [d for d in dirs if d not in excluded]  # exclude test directories
    for name in files:
        if name.endswith(".js"):
            blob = FileBlob(os.path.join(root, name))  # Linguist file checking
            if not (blob.is_binary or blob.is_generated):
                with open(os.path.join(root, name)) as js_file:
                    data = js_file.read()
                    minidata = '\xff' + jsmin(data) + '\xfe'
                    if minidata not in minified_data:
                        filecounter += 1
                        labels = ''
                        for token in point.get_tokens(minidata):
                            (type, seq) = token
                            if type == Token.Literal.String.Regex:
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
                            elif type in Token.Name:
                                tag = 'i'  # identifier
                            else:
                                tag = 'e'  # others

                            for i in range(len(seq)):
                                if not len(labels) == len(minidata):
                                    labels += tag
                        if len(labels) == len(minidata):
                            minified_data.append(minidata)
                            label_data.append(labels)


with open('chars', 'wb') as f:
    pickle.dump(minified_data, f)

with open('labels', 'wb') as f:
    pickle.dump(label_data, f)

minified_data = ''.join(minified_data)

t2 = time.time()
print "Created the dataset in: %f milliseconds from %d files" % ((t2 - t1) * 1000., filecounter)

chars = list(set(minified_data))
data_size, vocab_size = len(minified_data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
