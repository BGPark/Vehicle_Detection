from multiprocessing import Pool, Manager


def work(item):
    k = 0
    for i in range(10000):
        k += i
    print(item)

    # handle critical section
    lock.acquire()
    accuracy.value += 0.0001
    total_count.value += 1
    lock.release()
    return


def init(l, acc, count):
    global lock, accuracy, total_count
    lock = l
    accuracy = acc
    total_count = count
    return


if __name__ == '__main__':
    m = Manager()
    lock = m.Lock()
    accuracy = m.Value('acc', 0)
    total_count = m.Value('count', 0)

    pool = Pool(initializer=init, initargs=(lock, accuracy, total_count, ))
    result = pool.map(work, range(1, 10000))
    print(accuracy, total_count)


