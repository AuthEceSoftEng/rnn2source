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
