# cython: language_level=3

from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer, PyCapsule_SetPointer, PyCapsule_GetName
from cpython.pycapsule cimport PyCapsule_CheckExact
from cpython.unicode cimport PyUnicode_CheckExact, PyUnicode_AsUTF8String
from cpython.bytes cimport PyBytes_CheckExact, PyBytes_AS_STRING
from libc.string cimport memcpy, strcmp
from cpython.ref cimport Py_XDECREF, PyObject

from skynet_py cimport *


cdef void free_pyptr(object capsule):
    cdef void *ptr = PyCapsule_GetPointer(capsule, "pyptr")
    skynet_free(ptr)

def crecv():
    cdef SkynetPyMessage msg
    cdef int ret = skynet_py_queue_pop(&msg)
    while ret == 0 and msg.type == PTYPE_DECREF_PYTHON:
        Py_XDECREF(<PyObject*>msg.data)
        ret = skynet_py_queue_pop(&msg)
    if ret != 0:
        return None, None, None, None, None
    else:
        # TODO when msg.type is error , msg.data will be nil ?
        if msg.data != NULL:
            return msg.source, msg.type, msg.session, PyCapsule_New(msg.data, "pyptr", free_pyptr), msg.size
        else:
            return msg.source, msg.type, msg.session, b"", 0

# see lsend in lua-skynet.c
def csend(py_dst, int type_id, py_session, py_msg, py_size=None):
    assert skynet_py_address() > 0, "skynet threads has not been started yet, call 'pyskynet.start()' first."
    # 1. check dst
    cdef char * dstname = NULL
    cdef int dst = 0
    cdef bytes py_dst_bytes
    if PyBytes_CheckExact(py_dst):
        dstname = py_dst
    elif PyUnicode_CheckExact(py_dst):
        py_dst_bytes = PyUnicode_AsUTF8String(py_dst)
        dstname = py_dst_bytes
    else:
        dst = py_dst
    # 2. check session
    cdef int session = 0
    if py_session is None:
        type_id |= PTYPE_TAG_ALLOCSESSION
        session = 0
    else:
        session = py_session
    # 3. check ptr, size
    cdef char * ptr = NULL
    cdef int size = 0
    cdef bytes py_msg_bytes
    if PyCapsule_CheckExact(py_msg):
        type_id |= PTYPE_TAG_DONTCOPY
        ptr = <char *>PyCapsule_GetPointer(py_msg, "cptr")
        size = py_size
    elif PyBytes_CheckExact(py_msg):
        ptr = py_msg
        size = len(py_msg)
    elif PyUnicode_CheckExact(py_msg):
        py_msg_bytes = PyUnicode_AsUTF8String(py_msg)
    else:
        raise Exception("type:%s unexcept when skynet csend"%type(py_msg))

    if dstname == NULL:
        session = skynet_py_send(dst, type_id, session, ptr, size)
    else:
        session = skynet_py_sendname(dstname, type_id, session, ptr, size)
    if session < 0:
        if session == -2:
            raise Exception("package is too large:%s"%session)
        else:
            return None
    else:
        return session

# extern
def tostring(capsule, size_t size):
    b = tobytes(capsule, size)
    return b.decode()

# extern
def tobytes(capsule, size_t size):
    cdef char *name = PyCapsule_GetName(capsule)
    cdef char *ptr = <char *>PyCapsule_GetPointer(capsule, name)
    cdef bytes s
    if strcmp(name, "cptr") == 0 or strcmp(name, "pyptr") == 0:
        s = ptr[:size]
        return s
    else:
        raise Exception("capsule unpack failed for name = %s")

class PTYPEEnum(object):
    def __init__(self):
        self.PTYPE_TEXT=PTYPE_TEXT
        self.PTYPE_RESPONSE=PTYPE_RESPONSE
        self.PTYPE_MULTICAST=PTYPE_MULTICAST
        self.PTYPE_CLIENT=PTYPE_CLIENT
        self.PTYPE_SYSTEM=PTYPE_SYSTEM
        self.PTYPE_HARBOR=PTYPE_HARBOR
        self.PTYPE_SOCKET=PTYPE_SOCKET
        self.PTYPE_ERROR=PTYPE_ERROR
        self.PTYPE_QUEUE=PTYPE_RESERVED_QUEUE
        self.PTYPE_DEBUG=PTYPE_RESERVED_DEBUG
        self.PTYPE_LUA=PTYPE_RESERVED_LUA
        self.PTYPE_SNAX=PTYPE_RESERVED_SNAX
        self.PTYPE_TRACE=12 # TRACE defined in skynet.lua
        self.PTYPE_FOREIGN=PTYPE_FOREIGN
        self.PTYPE_FOREIGN_REMOTE=PTYPE_FOREIGN_REMOTE
        self.PTYPE_TAG_ALLOCSESSION=PTYPE_TAG_ALLOCSESSION
        self.PTYPE_TAG_DONTCOPY=PTYPE_TAG_DONTCOPY


SKYNET_PTYPE_id_to_name = {}
SKYNET_PTYPE_name_to_id = {}
SKYNET_PTYPE = PTYPEEnum()

for key, id in SKYNET_PTYPE.__dict__.items():
    name = key[6:].lower()
    SKYNET_PTYPE_id_to_name[id] = name
    SKYNET_PTYPE_name_to_id[name] = id
    locals()[key] = id


SKYNET_PTYPE_user_builtin_ids = [SKYNET_PTYPE.PTYPE_LUA, SKYNET_PTYPE.PTYPE_CLIENT, SKYNET_PTYPE.PTYPE_SOCKET, SKYNET_PTYPE.PTYPE_TEXT, SKYNET_PTYPE.PTYPE_FOREIGN, SKYNET_PTYPE.PTYPE_FOREIGN_REMOTE]
def user_assert_id_name(id, name):
    is_builtin = False
    if name in SKYNET_PTYPE_name_to_id:
        assert id == SKYNET_PTYPE_name_to_id[name], "skynet proto type name=%s must bind id=%s" % (name, id)
        is_builtin = True
    if id in SKYNET_PTYPE_id_to_name:
        assert name == SKYNET_PTYPE_id_to_name[id], "skynet proto type id=%s must bind name=%s" % (id, name)
        is_builtin = True
    if is_builtin:
        assert id in SKYNET_PTYPE_user_builtin_ids, "skynet proto can only register builtin proto 'lua' or 'client' or 'socket' or 'text' or 'foreign' or 'foreign_remote', but get name=%s" % name

    assert 0 <= id and id <= 255, "skynet proto type id must be less than 256 but get id=%s" % id
