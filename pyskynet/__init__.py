
###############################
# some api different from lua #
###############################
import pyskynet.boot
import pyskynet.skynet_py_mq
import pyskynet.foreign
import pyskynet.skynet_py_main as skynet_py_main
import pyskynet.skynet_py_foreign_seri as foreign_seri
import pyskynet.proto as pyskynet_proto

start = pyskynet.boot.start
join = pyskynet.boot.join
boot_config = pyskynet.boot.boot_config

#############
# proto api #
#############

PTYPE_TEXT = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_TEXT
PTYPE_CLIENT = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_CLIENT
PTYPE_SOCKET = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_SOCKET
PTYPE_LUA = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_LUA
PTYPE_FOREIGN_REMOTE = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_FOREIGN_REMOTE
PTYPE_FOREIGN = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_FOREIGN

pyskynet.rawcall = pyskynet_proto.rawcall
pyskynet.rawsend = pyskynet_proto.rawsend
pyskynet.ret = pyskynet_proto.ret

#################
# env set & get #
#################


def getenv(key):
    data = skynet_py_main.py_getenv(key)
    if data is None:
        return None
    else:
        return foreign_seri.remoteunpack(data)[0]


def setenv(key, value):
    if skynet_py_main.self() != 0:
        assert getenv(key) is None, "Can't setenv exist key : %s " % key
    msg_ptr, msg_size = foreign_seri.remotepack(value)
    skynet_py_main.py_setenv(key, msg_ptr, msg_size)
    foreign_seri.trash(msg_ptr, msg_size)


def envs():
    key = None
    re = {}
    while True:
        key = skynet_py_main.py_nextenv(key)
        if(key is None):
            break
        else:
            re[key] = getenv(key)
    return re


###############
# service api #
###############
def newservice(service_name, *args):
    assert type(service_name) == str or type(service_name) == bytes, "newservice's name must be str or bytes"
    for arg in args:
        assert type(arg) == str or type(service_name) == bytes, "newservice's arg must be str or bytes"
    return pyskynet_proto.call(".launcher", PTYPE_LUA, "LAUNCH", "snlua", service_name, *args)[0]


def uniqueservice(service_name, *args):
    assert type(service_name) == str or type(service_name) == bytes, "uniqueservice's name must be str or bytes"
    for arg in args:
        assert type(arg) == str or type(service_name) == bytes, "uniqueservice's arg must be str or bytes"
    return pyskynet_proto.call(".service", PTYPE_LUA, "LAUNCH", service_name, *args)[0]

def fileservice(filename, *args):
    return newservice("fast_service", "file", filename, *args)

def scriptservice(script, *args):
    info = skynet_py_main.ptr_wrap(*foreign_seri.remotepack(script))
    return newservice("fast_service", "script", info, *args)


class __CanvasService(object):
    def __init__(self, service):
        self.service = service

    def reset(self, *args):
        return pyskynet_proto.call(self.service, PTYPE_FOREIGN, "reset", *args)

    def render(self, *args):
        return pyskynet_proto.call(self.service, PTYPE_FOREIGN, "render", *args)

    def __del__(self):
        return pyskynet_proto.send(self.service, PTYPE_FOREIGN, "exit")


def canvas(script, name="unknowxml"):
    info = skynet_py_main.ptr_wrap(*foreign_seri.remotepack(script))
    return __CanvasService(newservice("canvas_service", info, name))


def self():
    address = skynet_py_main.self()
    assert address > 0, "service pyholder not start "
    return address


def test(script, *args):
    import pyskynet.foreign
    return pyskynet.foreign.call(boot.boot_service, "run", script, *args)
