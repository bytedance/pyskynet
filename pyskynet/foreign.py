
import pyskynet.skynet_py_foreign_seri as foreign_seri
import pyskynet.skynet as skynet
import pyskynet.skynet_py_mq as skynet_py_mq


PTYPE_FOREIGN = skynet_py_mq.PTYPE_FOREIGN
PTYPE_FOREIGN_REMOTE = skynet_py_mq.PTYPE_FOREIGN_REMOTE


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
            msg_ptr, msg_size = foreign_seri.__refpack(*ret)
        else:
            msg_ptr, msg_size = foreign_seri.__refpack(ret)
        hook_ptr = foreign_seri.__packhook(msg_ptr)
        if hook_ptr:
            if not skynet.ret(hook_ptr, msg_size):
                foreign_seri.__unref(msg_ptr)
                foreign_seri.trash(msg_ptr)
        else:
            skynet.ret(msg_ptr, msg_size)


def __foreign_remote_dispatch(session, source, argtuple):
    ret = CMD(*argtuple)
    if session != 0:
        if type(ret) == tuple:
            skynet.ret(*foreign_seri.remotepack(*ret))
        else:
            skynet.ret(*foreign_seri.remotepack(ret))

def __dontpackhere():
    raise skynet.PySkynetCallException("don't use pack here")

# dispatch foreign message
skynet.register_protocol(
        id=PTYPE_FOREIGN,
        name="foreign",
        pack=__dontpackhere,
        unpack=foreign_seri.__refunpack,
        dispatch=__foreign_dispatch,
        )

# dispatch foreign message
skynet.register_protocol(
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

def __safe_rawsend(addr, zero_none, msg_ptr, msg_size):
    hook_ptr = foreign_seri.__packhook(msg_ptr)
    if hook_ptr:
        session = skynet_py_mq.csend(addr, PTYPE_FOREIGN, zero_none, hook_ptr, msg_size)
        if session is None:
            foreign_seri.__unref(msg_ptr)
            foreign_seri.trash(msg_ptr)
    else:
        session = skynet_py_mq.csend(addr, PTYPE_FOREIGN, zero_none, msg_ptr, msg_size)
    return session


def call(addr, *args):
    msg_ptr, msg_size = foreign_seri.__refpack(*args)
    session = __safe_rawsend(addr, None, msg_ptr, msg_size)
    if session is None:
        raise psproto.PySkynetCallException("send to invalid address %08x" % dst)
    re = skynet.__yield_call(addr, session)
    return foreign_seri.__refunpack(*re)


def send(addr, *args):
    msg_ptr, msg_size = foreign_seri.__refpack(*args)
    session = __safe_rawsend(addr, 0, msg_ptr, msg_size)
    return session is not None
