# cython: language_level=3

cdef extern from "lua.h":
    ctypedef int lua_Integer


cdef extern from "skynet.h":
    ctypedef int int64_t
    ctypedef int int32_t
    ctypedef int uint32_t
    ctypedef int uint16_t
    ctypedef int uint8_t
    void skynet_free(void *)
    void* skynet_malloc(size_t)
    cdef enum:
        PTYPE_TEXT
        PTYPE_RESPONSE
        PTYPE_MULTICAST
        PTYPE_CLIENT
        PTYPE_SYSTEM
        PTYPE_HARBOR
        PTYPE_SOCKET
        PTYPE_ERROR
        PTYPE_RESERVED_QUEUE
        PTYPE_RESERVED_DEBUG
        PTYPE_RESERVED_LUA
        PTYPE_RESERVED_SNAX
        PTYPE_TAG_ALLOCSESSION
        PTYPE_TAG_DONTCOPY

cdef extern from "skynet_foreign/skynet_foreign.h":
    cdef struct skynet_foreign:
        pass
    void skynet_foreign_incref(skynet_foreign *obj);
    void skynet_foreign_decref(skynet_foreign *obj);

cdef extern from "skynet_modify/skynet_py.h":
    cdef struct skynet_config:
        int thread
        int harbor
        int profile
        const char * daemon
        const char * module_path
        const char * bootstrap
        const char * logger
        const char * logservice
    cdef struct SkynetPyMessage:
        int type
        int session
        uint32_t source
        void * data
        size_t size
    cdef enum:
        PTYPE_FOREIGN_REMOTE
        PTYPE_FOREIGN
        PTYPE_DECREF_PYTHON
    int skynet_py_queue_pop(SkynetPyMessage * )
    int skynet_py_send(uint32_t dst, int type, int session, void* msg, size_t sz);
    int skynet_py_sendname(const char *dst, int type, int session, void* msg, size_t sz);
    void skynet_py_init(int (*p_uv_async_send)(void *), void * p_uv_async_t);
    void skynet_py_start(skynet_config * config)
    void skynet_py_exit();

