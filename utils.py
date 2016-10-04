import numpy as np


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
            if name.endswith(".js"):
                with open(os.path.join(root, name)) as js_file:
                    data = js_file.read()
                if len(data) > 11:
                    minidata = jsmin(data) + '\xfe'
                    minified_data.append(minidata)

    minified_data = '\xff'.join(minified_data)

    t2 = time.time()
    print "Created the dataset in: %f milliseconds" % ((t2 - t1) * 1000.)

    return minified_data
