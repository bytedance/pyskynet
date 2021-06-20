# cython: language_level=3

import _cffi_backend
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New, PyCapsule_CheckExact
from cpython.bytes cimport PyBytes_FromString, PyBytes_FromStringAndSize, PyBytes_CheckExact
from libc.stdio cimport sscanf, sprintf

from skynet_py cimport *

cdef extern from "skynet_env.h":
    const char * skynet_getenv(const char *key);

cdef extern from "skynet_modify/skynet_py.h":
    void *skynet_py_setlenv(const char *key, const char *value_str, size_t sz)
    const char *skynet_py_getlenv(const char *key, size_t *sz);
    const char *skynet_py_nextenv(const char *key)

ctypedef (char *)(* f_type)(object, object)

def init(ffi, async_send, async_handle):
    # because gevent's libuv bind with cffi, so we can get pointer by this way
    cdef void ** _cffi_exports = <void **>PyCapsule_GetPointer(_cffi_backend._C_API, "cffi")
    # get the '_cffi_to_c_pointer' function in _cffi_include.h of cffi
    cdef f_type _cffi_to_c_pointer = <f_type>_cffi_exports[11]
    skynet_py_init(
            <int (*)(void*)>_cffi_to_c_pointer(async_send, ffi.typeof(async_send)),
            <void *>_cffi_to_c_pointer(async_handle, ffi.typeof(async_handle))
            )

cdef __check_bytes(s):
    t = type(s)
    if t == str:
        return s.encode("utf-8")
    elif t == bytes:
        return s
    else:
        raise Exception("type %s can't convert to bytes" % str(t))

def setlenv(key, capsule_or_bytes, py_sz=None):
    cdef size_t sz
    cdef const char *ptr
    if PyCapsule_CheckExact(capsule_or_bytes):
        ptr = <char *>PyCapsule_GetPointer(capsule_or_bytes, "cptr")
        sz = py_sz
    elif PyBytes_CheckExact(capsule_or_bytes):
        ptr = capsule_or_bytes
        sz = len(capsule_or_bytes)
    else:
        raise Exception("skynet_py env value must be bytes or pointer")
    if not (key is None):
        key = __check_bytes(key)
        skynet_py_setlenv(key, ptr, sz)
        return None
    cdef void *newkey = skynet_py_setlenv(NULL, ptr, sz)
    cdef char addr[32];
    cdef int k = sprintf(addr, "%p", newkey)
    return addr[:k]


def getlenv(key):
    key = __check_bytes(key)
    cdef size_t sz
    cdef const char * value = skynet_py_getlenv(key, &sz);
    if value != NULL:
        return PyBytes_FromStringAndSize(value, sz);
    else:
        return None

def nextenv(key):
    cdef const char * ptr
    if key is None:
        ptr = skynet_py_nextenv(NULL)
    else:
        key = __check_bytes(key)
        ptr = skynet_py_nextenv(key)
    if ptr == NULL:
        return None
    else:
        return PyBytes_FromString(ptr)

def start(int thread, int profile):
    cdef skynet_config config;
    config.thread = thread
    config.profile = profile
    # use getenv for a stable ptr
    config.module_path = skynet_getenv("cservice")
    config.bootstrap = skynet_getenv("bootstrap")
    config.logservice = skynet_getenv("logservice")
    config.logger = skynet_getenv("logger")
    # ignore
    config.harbor = 0
    config.daemon = NULL # just ignore daemon
    skynet_py_start(&config)

def exit():
    skynet_py_exit()

def self():
    return skynet_py_address()
