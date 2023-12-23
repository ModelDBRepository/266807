import multiprocessing as mp

def test():
    from matplotlib import pyplot as plt 
    plt.plot([1,2,3])
    plt.show()

pool1 = mp.Pool()
pool2 = mp.Pool()

pool1.apply_async(test)

pool2.apply_async(test)

import time
time.sleep(60)