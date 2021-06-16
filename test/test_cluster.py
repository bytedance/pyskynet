
import sys
sys.path.append("../")

import pyskynet
import pyskynet.cluster as cluster
import pyskynet.foreign as foreign
import numpy as np
import time

@foreign.dispatch("dosth")
def dosth(a):
    print(a)
    return a

pyskynet.start()


cluster.open(8080)



a1 = np.arange(100)
t1 = time.time()
a2, = cluster.call("127.0.0.1:8080", ".python", "dosth", a1)
t2 = time.time()
print(t2-t1)

t1 = time.time()
a2, = cluster.call("127.0.0.1:8080", ".python", "dosth", a1)
t2 = time.time()
print(t2-t1)

pyskynet.join()
