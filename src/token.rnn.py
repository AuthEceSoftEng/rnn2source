import os
import time
from pygments.lexers.javascript import JavascriptLexer

from jsmin import jsmin
from linguist.libs.file_blob import FileBlob

path = '/home/vasilis/Documents/projects'
print "Reading data..."
minified_data = ['']
print type(minified_data)
t1 = time.time()
filecounter = 0
excluded = {'test', 'tests', '__tests__' 'locale', 'locales', 'ngLocale'}
point = JavascriptLexer()
label_data = []
print type(label_data)
for root, dirs, files in os.walk(path, topdown=True):
    dirs[:] = [d for d in dirs if d not in excluded]  # exclude test directories
    for name in files:
        if name.endswith(".js"):
            blob = FileBlob(os.path.join(root, name))  # Linguist file checking
            if not (blob.is_binary or blob.is_generated):
                filecounter += 1
                with open(os.path.join(root, name)) as js_file:
                    data = js_file.read()
                    minidata = '\xff' + jsmin(data) + '\xfe'
                    labels = []
                    for token in point.get_tokens_unprocessed(minidata):
                        (index, label, seq) = token
                        for i in range(len(seq)):
                            labels.append(label)
                    minified_data.append(minidata)
                    label_data.append(labels)
        if filecounter == 250: break
minified_data = ''.join(minified_data)

t2 = time.time()
print "Created the dataset in: %f milliseconds from %d files" % ((t2 - t1) * 1000., filecounter)

chars = list(set(minified_data))
data_size, vocab_size = len(minified_data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
