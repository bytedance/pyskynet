
###############################
# some api different from lua #
###############################
import pyskynet.boot
import pyskynet.skynet_py_mq
import pyskynet.foreign as foreign
import pyskynet.skynet_py_main as skynet_py_main
import pyskynet.skynet as skynet

__version__ = '0.1.5'
start = pyskynet.boot.start
join = pyskynet.boot.join
boot_config = pyskynet.boot.boot_config

proto = skynet # for compatiable with old code
#############
# skynet api #
#############

PTYPE_TEXT = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_TEXT
PTYPE_CLIENT = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_CLIENT
PTYPE_SOCKET = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_SOCKET
PTYPE_LUA = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_LUA
PTYPE_FOREIGN_REMOTE = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_FOREIGN_REMOTE
PTYPE_FOREIGN = pyskynet.skynet_py_mq.SKYNET_PTYPE.PTYPE_FOREIGN

pyskynet.rawcall = skynet.rawcall
pyskynet.rawsend = skynet.rawsend
pyskynet.ret = skynet.ret

#################
# env set & get #
#################


def getenv(key):
    data = skynet_py_main.getlenv(key)
    if data is None:
        return None
    else:
        return foreign.remoteunpack(data)[0]


def setenv(key, value):
    if skynet_py_main.self() != 0:
        assert (key is None) or (getenv(key) is None), "Can't setenv exist key : %s " % key
    msg_ptr, msg_size = foreign.remotepack(value)
    newkey = skynet_py_main.setlenv(key, msg_ptr, msg_size)
    foreign.trash(msg_ptr)
    return newkey


def envs():
    key = None
    re = {}
    while True:
        key = skynet_py_main.nextenv(key)
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
        assert type(arg) == str or type(arg) == bytes, "newservice's arg must be str or bytes"
    return skynet.call(".launcher", PTYPE_LUA, "LAUNCH", "snlua", service_name, *args)[0]


def uniqueservice(service_name, *args):
    assert type(service_name) == str or type(service_name) == bytes, "uniqueservice's name must be str or bytes"
    for arg in args:
        assert type(arg) == str or type(arg) == bytes, "uniqueservice's arg must be str or bytes"
    return skynet.call(".service", PTYPE_LUA, "LAUNCH", service_name, *args)[0]


def scriptservice(scriptaddr_or_loadargs, *args):
    t1 = type(scriptaddr_or_loadargs)
    if t1 == str and scriptaddr_or_loadargs.startswith("0x"):
        scriptaddr = scriptaddr_or_loadargs
    elif t1 == bytes and scriptaddr_or_loadargs.startswith(b"0x"):
        scriptaddr = scriptaddr_or_loadargs
    elif t1 == str or t1 == bytes:
        scriptaddr = setenv(None, [scriptaddr_or_loadargs])
    elif t1 == list:
        scriptaddr = setenv(None, scriptaddr_or_loadargs)
    else:
        raise Exception("loadservice's first args must be str or bytes or list")
    return newservice("script_service", scriptaddr, *args)


class __CanvasService(object):
    def __init__(self, service):
        self.service = service

    def reset(self, *args):
        return foreign.call(self.service, "reset", *args)

    def render(self, *args):
        return foreign.call(self.service, "render", *args)

    def __del__(self):
        return foreign.send(self.service, "exit")


def canvas(script, name="unknowxml"):
    canvas_service = newservice("canvas_service")
    foreign.call(canvas_service, "init", script, name)
    return __CanvasService(canvas_service)


def self():
    address = skynet_py_main.self()
    assert address > 0, "service pyholder not start "
    return address


def test(script, *args):
    return foreign.call(boot.boot_service, "run", script, *args)
