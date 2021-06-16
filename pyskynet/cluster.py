

import pyskynet

import pyskynet.boot
import pyskynet.proto as pyskynet_proto
from pyskynet.skynet_py_foreign_seri import remotepack, remoteunpack, luapack
import gevent
import gevent.event

clusterd = None


def __load_clusterd():
    global clusterd
    clusterd = pyskynet.uniqueservice("foreign_clusterd")


pyskynet.boot.init(__load_clusterd)

node_to_sender = {}
node_to_task_queue = {}


def __request_sender(task_queue, node):
    sender = pyskynet_proto.call(clusterd, "lua", "sender", node)[0]
    i = 0
    while i < len(task_queue):
        task = task_queue[i]
        # tuple for send task, AsyncResult for call task
        if type(task) == tuple:
            pyskynet_proto.send(sender, "lua", "push", task[0], remotepack(*task[1]))
        else:
            confirm = gevent.event.Event()
            task.set((confirm, sender))
            confirm.wait()
        i += 1
    node_to_task_queue.pop(node)
    node_to_sender[node] = sender


def __get_sender(node):
    sender = node_to_sender.get(node)
    if sender is None:
        task = gevent.event.AsyncResult()
        task_queue = node_to_task_queue.get(node)
        if task_queue is None:
            task_queue = [task]
            node_to_task_queue[node] = task_queue
            gevent.spawn(__request_sender, task_queue, node)
        else:
            task_queue.append(task)
        confirm, sender = task.get()
        confirm.set()
        return sender
    else:
        return sender


def call(node, address, *args):
    return remoteunpack(*pyskynet_proto.rawcall(
        __get_sender(node), "lua", *luapack("req", address, *remotepack(*args))))


def send(node, address, *args):
    sender = node_to_sender.get(node)
    if sender is None:
        node_to_task_queue[node].append((address, args))
    else:
        pyskynet_proto.send(sender, "lua", "push", address, remotepack(*args))


def open(port):
    assert 0 <= port <= 65535, "cluster.open's port must be in [0, 65535]"
    pyskynet_proto.call(clusterd, "lua", "listen", "0.0.0.0", port)
