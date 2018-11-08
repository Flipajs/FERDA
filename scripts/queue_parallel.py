#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
from builtins import range
from multiprocessing import Lock, Process, Queue, current_process

def worker(work_queue, done_queue):
    try:
        for url in iter(work_queue.get, 'STOP'):
            status_code = print_site_status(url)
            done_queue.put("%s - %s got %s." % (current_process().name, url, status_code))
    except Exception as e:
        done_queue.put("%s failed on %s with: %s" % (current_process().name, url, e.message))
    return True


def print_site_status(url):
    x = url[10] + '___'

    return x


def main():
    sites = (
        'http://penny-arcade.com/',
        'http://reallifecomics.com/',
        'http://sinfest.net/',
        'http://userfriendly.org/',
        'http://savagechickens.com/',
        'http://xkcd.com/',
        'http://duelinganalogs.com/',
        'http://cad-comic.com/',
        'http://samandfuzzy.com/',
    )
    workers = 2
    work_queue = Queue()
    done_queue = Queue()
    processes = []

    for url in sites:
        work_queue.put(url)

    for w in range(workers):
        p = Process(target=worker, args=(work_queue, done_queue))
        p.start()
        processes.append(p)
        work_queue.put('STOP')

    for p in processes:
        p.join()

    done_queue.put('STOP')

    for status in iter(done_queue.get, 'STOP'):
        print(status)


if __name__ == '__main__':
    main()