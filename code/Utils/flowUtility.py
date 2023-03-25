import numpy as np

TAG_FLOAT = 202021.25


def readFlow(path):

    with open(path, 'rb') as f:
        tag = float(np.fromfile(f, np.float32, count=1)[0])
        assert(tag == TAG_FLOAT)
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]

        flow = np.fromfile(f, np.float32, count=h * w * 2)
        flow.resize((h, w, 2))

    return flow
