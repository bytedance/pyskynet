# cython: language_level=3

from libc.string cimport memcpy, strcmp
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer, PyCapsule_GetName
# import some type check
from cpython.pycapsule cimport PyCapsule_CheckExact
from cpython.unicode cimport PyUnicode_CheckExact, PyUnicode_AsUTF8String, PyUnicode_DecodeUTF8
from cpython.bytes cimport PyBytes_CheckExact, PyBytes_AS_STRING, PyBytes_FromStringAndSize, PyBytes_FromString, PyBytes_GET_SIZE
from cpython.object cimport PyObject, PyTypeObject
from cpython.memoryview cimport PyMemoryView_Check
from cpython.bytearray cimport PyByteArray_AS_STRING
from numpy cimport PyArray_CheckExact
import numpy as np
cimport numpy as cnp
cnp.import_array()

from skynet_py cimport *

cdef extern from "skynet_py_foreign_seri_ext.c": #from "lua-foreign_seri.c":
    cdef enum:
        MODE_LUA
        MODE_FOREIGN_REF
        MODE_FOREIGN_REMOTE

    # for read
    cdef struct read_block:
        int mode
    void rb_init(read_block* rb, char* buffer, int64_t size, int mode)
    void* rb_read(read_block* rb, int64_t sz)
    uint8_t COMBINE_TYPE(uint8_t, uint8_t)

    # for write
    cdef struct write_block:
        char *buffer
        int64_t len
        int mode
    void wb_init(write_block* wb, int mode)
    void wb_free(write_block *wb)
    void wb_nil(write_block* wb)
    void wb_integer(write_block* wb, lua_Integer v)
    void wb_real(write_block* wb, double v)
    void wb_boolean(write_block* wb, int v)
    void wb_pointer(write_block *wb, void *v)
    void wb_string(write_block *wb, const char *ptr, int sz)
    void wb_write(write_block *wb, const void *buf, int64_t sz)
    cdef enum:
        TYPE_NIL
        TYPE_BOOLEAN
        TYPE_NUMBER
        TYPE_NUMBER_ZERO
        TYPE_NUMBER_BYTE
        TYPE_NUMBER_WORD
        TYPE_NUMBER_DWORD
        TYPE_NUMBER_QWORD
        TYPE_NUMBER_REAL

        TYPE_USERDATA
        TYPE_SHORT_STRING
        TYPE_LONG_STRING
        TYPE_TABLE
        TYPE_FOREIGN_USERDATA
    cdef enum:
        MAX_COOKIE
        MAX_DEPTH

    # deal for PyArray
    bint PyArray_foreign_check_typechar(object py_obj)
    void wb_foreign_PyArray(write_block *wb, object arr, object arr_iter)
    object unpack_PyArray(read_block *wb, int cookie)

########################
# functions for unpack #
########################

cdef double py_get_real(read_block* rb) except *:
    cdef double n_double
    cdef void * ptr
    ptr = rb_read(rb, sizeof(n_double))
    if(ptr==NULL):
        raise Exception("invalid stream")
    memcpy(&n_double, ptr, sizeof(n_double));
    return n_double

cdef lua_Integer py_get_integer(read_block* rb, int cookie) except *:
    cdef uint8_t n_uint8
    cdef uint16_t n_uint16
    cdef int32_t n_int32
    cdef int64_t n_int64
    cdef void * ptr
    if(cookie==TYPE_NUMBER_ZERO):
        return 0
    elif(cookie==TYPE_NUMBER_BYTE):
        ptr = rb_read(rb, sizeof(n_uint8));
        if (ptr == NULL):
            raise Exception("invalid stream")
        memcpy(&n_uint8, ptr, sizeof(n_uint8));
        return n_uint8
    elif(cookie==TYPE_NUMBER_WORD):
        ptr = rb_read(rb, sizeof(n_uint16));
        if(ptr == NULL):
            raise Exception("invalid stream")
        memcpy(&n_uint16, ptr, sizeof(n_uint16));
        return n_uint16
    elif(cookie==TYPE_NUMBER_DWORD):
        ptr = rb_read(rb, sizeof(n_int32));
        if(ptr == NULL):
            raise Exception("invalid stream")
        memcpy(&n_int32, ptr, sizeof(n_int32));
        return n_int32
    elif(cookie==TYPE_NUMBER_QWORD):
        ptr = rb_read(rb, sizeof(n_int64));
        if(ptr == NULL):
            raise Exception("invalid stream")
        memcpy(&n_int64, ptr, sizeof(n_int64));
        return n_int64
    else:
        raise Exception("invalid stream")

#cdef skynet_foreign* py_get_foreign(read_block *rb) except NULL:
#    cdef skynet_foreign* userdata = NULL
#    cdef void * buffer = NULL
#    cdef void ** v
#    cdef int32_t * psize
#    if rb.mode == MODE_FOREIGN:
#        v = <void **>rb_read(rb,sizeof(userdata));
#        if v == NULL:
#            raise Exception("invalid stream")
#        memcpy(&userdata, v, sizeof(userdata));
#    elif rb.mode == MODE_FOREIGN_REMOTE:
#        userdata = skynet_foreign_deserialize(rb, <void *(*)(void*, int)>rb_read);
#        if userdata == NULL:
#            raise Exception("invalid stream")
#    else:
#        raise Exception("unexcept mode for TYPE_FOREIGN_USERDATA")
#    return userdata

cdef void* py_get_pointer(read_block *rb) except NULL:
    cdef void * userdata
    cdef void ** v = <void **>rb_read(rb,sizeof(userdata));
    if v == NULL:
        raise Exception("invalid stream")
    memcpy(&userdata, v, sizeof(userdata));
    return userdata

cdef char * py_get_string(read_block *rb, int value_type, int cookie, size_t *out) except NULL:
    cdef char * p = NULL
    cdef uint16_t *plen2
    cdef uint32_t *plen4
    if value_type==TYPE_SHORT_STRING:
        p = <char *>rb_read(rb, cookie)
        out[0] = cookie
    elif value_type==TYPE_LONG_STRING:
        if(cookie == 2):
            plen2 = <uint16_t *>rb_read(rb, 2)
            if (plen2 == NULL):
                raise Exception("invalid stream")
            p = <char *>rb_read(rb, plen2[0])
            out[0] = plen2[0]
        elif(cookie == 4):
            plen4 = <uint32_t *>rb_read(rb, 4)
            if(plen4 == NULL):
                raise Exception("invalid stream")
            p = <char *>rb_read(rb, plen4[0])
            out[0] = plen4[0]
    return p

cdef void py_push_value(l, read_block *rb, int value_type, int cookie, bint iskey) except *:
    cdef char * ptr = NULL
    cdef size_t length = 0
    cdef void * lightuserdata = NULL
    cdef skynet_foreign * foreign_obj = NULL
    if value_type == TYPE_NIL:
        l.append(None)
    elif value_type == TYPE_BOOLEAN:
        l.append(cookie>0)
    elif value_type == TYPE_NUMBER:
        if (cookie == TYPE_NUMBER_REAL):
            l.append(py_get_real(rb))
        else:
            l.append(py_get_integer(rb, cookie))
    elif value_type==TYPE_SHORT_STRING:
        ptr = py_get_string(rb, value_type, cookie, &length)
        #if iskey:
            #l.append(PyUnicode_DecodeUTF8(ptr, length, NULL))
        #else:
        l.append(PyBytes_FromStringAndSize(ptr, length))
    elif value_type==TYPE_LONG_STRING:
        ptr = py_get_string(rb, value_type, cookie, &length)
        #if iskey:
            #l.append(PyUnicode_DecodeUTF8(ptr, length, NULL))
        #else:
        l.append(PyBytes_FromStringAndSize(ptr, length))
    elif value_type==TYPE_TABLE:
        py_unpack_table(l, rb, cookie)
    elif value_type==TYPE_USERDATA:
        lightuserdata = py_get_pointer(rb)
        l.append(PyCapsule_New(lightuserdata, "cptr", NULL))
    elif value_type==TYPE_FOREIGN_USERDATA:
        arr = unpack_PyArray(rb, cookie)
        if arr is None:
            raise Exception("invalid stream when unpacking arr")
        l.append(arr)
    else:
        raise Exception("invalid stream for value type exception")

cdef void py_unpack_one(l, read_block *rb, bint iskey) except *:
    cdef uint8_t value_type
    cdef uint8_t *value_ptr = <uint8_t*>rb_read(rb, sizeof(value_type));
    if value_ptr == NULL:
        raise Exception("invalid stream")
    value_type = value_ptr[0]
    py_push_value(l, rb, value_type & 0x7, value_type>>3, iskey);

cdef void py_unpack_table(l, read_block *rb, int array_size) except *:
    cdef uint8_t value_type
    cdef uint8_t *value_ptr
    cdef int cookie
    if array_size == MAX_COOKIE-1:
        value_ptr = <uint8_t *>rb_read(rb, sizeof(value_type))
        if value_ptr == NULL:
            raise Exception("invalid stream")
        value_type=value_ptr[0]
        cookie = value_type>>3
        if (value_type & 7) != TYPE_NUMBER or cookie == TYPE_NUMBER_REAL:
            raise Exception("invalid stream")
        array_size = py_get_integer(rb, cookie)
    #l.append(t)
    next_l = []
    for i in range(1, array_size+1):
        py_unpack_one(next_l, rb, 0)
    next_t = {}
    while True:
        py_unpack_one(next_l, rb, 1)
        if next_l[-1] is None:
            next_l.pop()
            break
        py_unpack_one(next_l, rb, 0)
        next_t[next_l[-2]] = next_l[-1]
        next_l.pop()
        next_l.pop()
    if not next_t:
        l.append(next_l)
    else:
        for i, v in enumerate(next_l):
            next_t[i + 1] = v
        l.append(next_t)

cdef void cunpack(l, char *msg, size_t size, int mode) except *:
    cdef read_block rb
    rb_init(&rb, msg, size, mode);
    cdef int i = 0
    cdef uint8_t value_type = 0
    cdef uint8_t *value_ptr = NULL
    while True:
        value_ptr = <uint8_t *>rb_read(&rb, sizeof(value_type))
        if value_ptr == NULL:
            break
        value_type = value_ptr[0]
        py_push_value(l, &rb, value_type & 0x7, value_type>>3, 0);

# extern
cdef py_foreign_unpack(int mode, capsule_or_bytes, py_sz):
    cdef const char *name
    cdef char *ptr
    cdef size_t sz
    l = []
    if PyCapsule_CheckExact(capsule_or_bytes):
        name = PyCapsule_GetName(capsule_or_bytes)
        ptr = <char *>PyCapsule_GetPointer(capsule_or_bytes, name)
        sz = py_sz
        if strcmp(name, "cptr") == 0 or strcmp(name, "pyptr") == 0:
            cunpack(l, ptr, sz, mode)
            return tuple(l)
        else:
            raise Exception("capsule unpack failed for name = %s " % PyBytes_FromString(name))
    elif PyBytes_CheckExact(capsule_or_bytes):
        ptr = <char *>PyBytes_AS_STRING(capsule_or_bytes)
        sz = PyBytes_GET_SIZE(capsule_or_bytes)
        cunpack(l, ptr, sz, mode)
        return tuple(l)
    else:
        raise Exception("Unexcept type %s " % str(type(capsule_or_bytes)))

######################
# functions for pack #
######################
cdef void py_pack_list(write_block* wb, list_obj, int depth) except *:
    cdef int array_size = len(list_obj)
    cdef uint8_t n
    if array_size >= MAX_COOKIE - 1:
        n = COMBINE_TYPE(TYPE_TABLE, MAX_COOKIE - 1);
        wb_write(wb, &n, 1);
        wb_integer(wb, array_size)
    else:
        n = COMBINE_TYPE(TYPE_TABLE, array_size);
        wb_write(wb, &n, 1);
    for v in list_obj:
        py_pack_one(wb,v,depth)
    wb_nil(wb)

cdef void py_pack_dict(write_block* wb, dict_obj, int depth) except *:
    cdef uint8_t n = COMBINE_TYPE(TYPE_TABLE, 0);
    wb_write(wb, &n, 1);
    for k, v in dict_obj.items():
        py_pack_one(wb,k,depth)
        py_pack_one(wb,v,depth)
    wb_nil(wb)

cdef void py_pack_one(write_block* wb, py_arg, int depth) except *:
    cdef int64_t bytes_sz = 0
    cdef char *bytes_ptr = NULL
    cdef const char *name = NULL
    if depth > MAX_DEPTH:
        wb_free(wb)
        raise Exception("serialize can't pack too depth table")
    if py_arg is None:
        wb_nil(wb)
    elif PyCapsule_CheckExact(py_arg):
        name = PyCapsule_GetName(py_arg)
        if strcmp(name, "cptr") == 0:
            wb_pointer(wb, PyCapsule_GetPointer(py_arg, "cptr"))
        else:
            wb_free(wb)
            raise Exception("unexception capsule")
    elif PyArray_CheckExact(py_arg):
        if not PyArray_foreign_check_typechar(py_arg):
            wb_free(wb)
            raise Exception("unexception typechar %s"%py_arg.dtype.char)
        if wb.mode == MODE_FOREIGN_REF:
            wb_foreign_PyArray(wb, py_arg, None)
        elif wb.mode == MODE_FOREIGN_REMOTE:
            arr_iter = py_arg.flat
            wb_foreign_PyArray(wb, py_arg, arr_iter)
        else:
            wb_free(wb)
            raise Exception("seri wb foreign unexception mode")
    elif isinstance(py_arg, dict):
        py_pack_dict(wb, py_arg, depth+1)
    elif isinstance(py_arg, list):
        py_pack_list(wb, py_arg, depth+1)
    else:
        py_arg_type = type(py_arg)
        if py_arg_type == int or np.issubdtype(py_arg_type, np.integer):
            wb_integer(wb, py_arg)
        elif py_arg_type == float or np.issubdtype(py_arg_type, np.floating):
            wb_real(wb, py_arg)
        elif py_arg_type == bool:
            wb_boolean(wb, py_arg)
        elif py_arg_type == bytes:
            bytes_sz = len(py_arg)
            bytes_ptr = py_arg
            wb_string(wb, bytes_ptr, bytes_sz)
        elif py_arg_type == bytearray:
            bytes_sz = len(py_arg)
            bytes_ptr = PyByteArray_AS_STRING(py_arg)
            wb_string(wb, bytes_ptr, bytes_sz)
        elif py_arg_type == str:
            py_arg_bytes = PyUnicode_AsUTF8String(py_arg)
            bytes_sz = len(py_arg_bytes)
            bytes_ptr = py_arg_bytes
            wb_string(wb, bytes_ptr, bytes_sz)
        else:
            wb_free(wb)
            raise Exception("Unsupport type %s to serialize"%str(py_arg_type))

cdef py_foreign_pack(int mode, argtuple):
    cdef write_block wb
    wb_init(&wb, mode)
    for one in argtuple:
        py_pack_one(&wb, one, 0)
    return PyCapsule_New(wb.buffer, "cptr", NULL), wb.len

#########################
# outside pack & unpack #
#########################

def tobytes(capsule, size_t size):
    cdef char *ptr = <char *>PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule))
    return ptr[:size]

def luapack(*args):
    return py_foreign_pack(MODE_LUA, args)

def luaunpack(capsule, size=None):
    return py_foreign_unpack(MODE_LUA, capsule, size)

def pack(*args):
    return py_foreign_pack(MODE_FOREIGN_REF, args)

def unpack(capsule, size=None):
    return py_foreign_unpack(MODE_FOREIGN_REF, capsule, size)

def remotepack(*args):
    return py_foreign_pack(MODE_FOREIGN_REMOTE, args)

def remoteunpack(capsule, size=None):
    return py_foreign_unpack(MODE_FOREIGN_REMOTE, capsule, size)

def trash(capsule, size_t sz):
    cdef void *ptr = PyCapsule_GetPointer(capsule, "cptr")
    skynet_free(ptr)
