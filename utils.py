import numpy as np
import os
import time
from pygments.lexers.javascript import JavascriptLexer

from jsmin import jsmin
from linguist.libs.file_blob import FileBlob

def sample(preds, temperature=0.35):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def jsparser(path):
    from jsmin import jsmin
    import os
    import time

    print "Reading data..."
    minified_data = ['']
    t1 = time.time()
    for root, dirs, files in os.walk(path):
        for name in files:
            if FileBlob(os.path.join(root, name)).is_generated:
                if name.endswith(".js"):
                    with open(os.path.join(root, name)) as js_file:
                        data = js_file.read()
                        print data
                    if len(data) > 11:
                        minidata = jsmin(data) + '\xfe'
                        minified_data.append(minidata)
                        minified_data = '\xff'.join(minified_data)
    minified_data = '\xff'.join(minified_data)

    t2 = time.time()
    print "Created the dataset in: %f milliseconds" % ((t2 - t1) * 1000.)

    return minified_data


def jsparserling(path):

    print "Reading data..."
    minified_data = ['']
    label_data = []
    t1 = time.time()
    filecounter = 0
    excluded = set(['test', 'tests', '__tests__' 'locale', 'locales', 'ngLocale'])
    point = JavascriptLexer()

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
    minified_data = ''.join(minified_data)

    t2 = time.time()
    print "Created the dataset in: %f milliseconds from %d files" % ((t2 - t1) * 1000., filecounter)

    chars = list(set(minified_data))
    data_size, vocab_size = len(minified_data), len(chars)
    print 'data has %d characters, %d unique.' % (data_size, vocab_size)
