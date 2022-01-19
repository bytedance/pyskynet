
import pyskynet.skynet_py_foreign_seri as foreign_seri
import pyskynet.proto as pyskynet_proto
import pyskynet.skynet_py_mq


PTYPE_FOREIGN = pyskynet.skynet_py_mq.PTYPE_FOREIGN
PTYPE_FOREIGN_REMOTE = pyskynet.skynet_py_mq.PTYPE_FOREIGN_REMOTE


class CMDDispatcher(object):
    def __init__(self):
        self.cmd2func = {}

    def __call__(self, first, *args):
        return self.cmd2func[first](*args)

    def __setitem__(self, k, v):
        if isinstance(k, str):
            self.cmd2func[k] = v
            self.cmd2func[k.encode("ascii")] = v
        elif isinstance(k, bytes):
            self.cmd2func[k] = v
            self.cmd2func[k.decode("ascii")] = v
        else:
            self.cmd2func[k] = v

    def __getitem__(self, k):
        return self.cmd2func[k]


CMD = CMDDispatcher()

remotepack = foreign_seri.remotepack
remoteunpack = foreign_seri.remoteunpack

trash = foreign_seri.trash


def __foreign_dispatch(session, source, argtuple):
    ret = CMD(*argtuple)
    if session != 0:
        if type(ret) == tuple:
            pyskynet_proto.ret(*foreign_seri.refpack(*ret))
        else:
            pyskynet_proto.ret(*foreign_seri.refpack(ret))


def __foreign_remote_dispatch(session, source, argtuple):
    ret = CMD(*argtuple)
    if session != 0:
        if type(ret) == tuple:
            pyskynet_proto.ret(*foreign_seri.remotepack(*ret))
        else:
            pyskynet_proto.ret(*foreign_seri.remotepack(ret))

def __dontpackhere():
    raise pyskynet_proto.PySkynetCallException("don't use pack here")

# dispatch foreign message
pyskynet_proto.register_protocol(
        id=PTYPE_FOREIGN,
        name="foreign",
        pack=__dontpackhere,
        unpack=foreign_seri.refunpack,
        dispatch=__foreign_dispatch,
        )

# dispatch foreign message
pyskynet_proto.register_protocol(
        id=PTYPE_FOREIGN_REMOTE,
        name="foreign_remote",
        pack=__dontpackhere,
        unpack=foreign_seri.remoteunpack,
        dispatch=__foreign_remote_dispatch,
        )


def dispatch(cmd=None, func=None):
    global CMD

    def wrapper(func):
        CMD[cmd] = func
        return func
    if not (func is None):
        assert callable(func), "dispatch's second arg must be callable"
        CMD[cmd] = func
    elif callable(cmd):
        CMD = func
    elif isinstance(cmd, dict):
        for k, v in cmd.items():
            CMD[k] = v
    elif isinstance(cmd, str) or isinstance(cmd, bytes):
        return wrapper
    else:
        raise Exception("dispatch failed for unexception args")


def call(addr, *args):
    msg_ptr, msg_size = foreign_seri.refpack(*args)
    return foreign_seri.refunpack(*pyskynet_proto.rawcall(
        addr, PTYPE_FOREIGN, msg_ptr, msg_size))


def send(addr, *args):
    msg_ptr, msg_size = foreign_seri.refpack(*args)
    pyskynet_proto.rawsend(addr, PTYPE_FOREIGN, msg_ptr, msg_size)
