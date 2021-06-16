
import pyskynet.skynet_py_mq as skynet_py_mq
import pyskynet.skynet_py_foreign_seri as foreign_seri
import gevent
from gevent.event import AsyncResult

SKYNET_PTYPE = skynet_py_mq.SKYNET_PTYPE

#####################
# py<->skynet proto #
#####################

local_session_to_ar = {}

pyskynet_proto_dict = {}


class PySkynetProto(object):
    def __init__(self, id, name, pack=None, unpack=None, dispatch=None):
        self.id = id
        self.name = name
        self.pack = pack
        self.unpack = unpack
        self.dispatch = dispatch


# skynet.lua
def register_protocol(id, name, pack=None, unpack=None, dispatch=None):
    if type(name) == bytes:
        name = name.decode()
    assert id not in pyskynet_proto_dict, "pyskynet proto id=%s existed" % id
    assert name not in pyskynet_proto_dict, "pyskynet proto name=%s existed" % name
    assert type(name) == str and type(id) == int and \
    id >= 0 and id <= 255, "pyskynet proto register failed id=%s, name=%s" % (id, name)
    psproto = PySkynetProto(id, name, pack, unpack, dispatch)
    pyskynet_proto_dict[id] = psproto
    pyskynet_proto_dict[name] = psproto


# skynet.lua
def _error_dispatch(error_session, error_source, arg):
    co = gevent.getcurrent()
    co_to_remote_session.pop(co)
    co_to_remote_address.pop(co)
    local_session_to_ar[error_session].set(None)


# skynet.lua
register_protocol(
        id=SKYNET_PTYPE.PTYPE_LUA,
        name="lua",
        pack=foreign_seri.luapack,
        unpack=foreign_seri.luaunpack,
        )

# skynet.lua
register_protocol(
        id=SKYNET_PTYPE.PTYPE_ERROR,
        name="error",
        pack=foreign_seri.luapack,
        unpack=lambda a, b: None,
        dispatch=_error_dispatch,
        )


# skynet.lua
def dispatch(name, func):
    """
        dispatch in skynet.lua,
        set dispatch
    """
    pyskynet_proto_dict[name].dispatch = func


####################
# code for session #
####################

co_to_remote_session = {}
co_to_remote_address = {}


class PySkynetCallException(Exception):
    pass


# skynet.lua
def rawcall(dst, type_name_or_id, msg_ptr, msg_size):
    """
        rawcall in skynet.lua, rpc call
    """
    psproto = pyskynet_proto_dict[type_name_or_id]
    session = skynet_py_mq.csend(dst, psproto.id, None, msg_ptr, msg_size)
    ar = AsyncResult()
    local_session_to_ar[session] = ar
    re = ar.get()
    if re:
        return re
    else:
        raise PySkynetCallException("call failed from %s" % dst)


# skynet.lua
def rawsend(dst, type_name_or_id, msg_ptr, msg_size):
    """
        rawsend in skynet.lua, send don't need ret
    """
    psproto = pyskynet_proto_dict[type_name_or_id]
    skynet_py_mq.csend(dst, psproto.id, 0, msg_ptr, msg_size)


# skynet.lua
def call(addr, type_name_or_id, *args):
    return foreign_seri.unpack(*rawcall(addr, type_name_or_id, *foreign_seri.pack(*args)))


# skynet.lua
def send(addr, type_name_or_id, *args):
    rawsend(addr, type_name_or_id, *foreign_seri.pack(*args))


# skynet.lua
def ret(ret_msg_ptr, ret_size):
    """
        ret in skynet.lua, return for other's call
    """
    co = gevent.getcurrent()
    session = co_to_remote_session.pop(co)
    source = co_to_remote_address.pop(co)
    if session == 0:
        pass  # send don't need ret
    else:
        skynet_py_mq.csend(source, SKYNET_PTYPE.PTYPE_RESPONSE, session, ret_msg_ptr, ret_size)
    # TODO if package is to large, c++ will trigger some exception...


################
# raw dispatch #
################
def async_handle():
    """
        python actor's main loop, recv and deal message
    """
    source, type_id, session, ptr, length = skynet_py_mq.crecv()
    if source is None:
        return
    else:
        gevent.spawn(async_handle)
    if type_id == SKYNET_PTYPE.PTYPE_RESPONSE:
        # TODO exception
        ar = local_session_to_ar.pop(session)
        ar.set((ptr, length))
    else:
        # TODO exception
        psproto = pyskynet_proto_dict[type_id]
        co = gevent.getcurrent()
        co_to_remote_session[co] = session
        co_to_remote_address[co] = source
        psproto.dispatch(session, source, psproto.unpack(ptr, length))
        if co in co_to_remote_session:
            if session != 0:
                print("Maybe forgot response session %s from %s " % (session, source))
            co_to_remote_session.pop(co)
            co_to_remote_address.pop(co)
