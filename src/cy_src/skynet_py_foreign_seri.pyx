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

cdef extern from "skynet_py_foreign_seri_ext.c":
    cdef enum:
        MODE_LUA
        MODE_FOREIGN_REF
        MODE_FOREIGN_REMOTE
    ctypedef int intptr_t

    # for read
    uint8_t COMBINE_TYPE(uint8_t, uint8_t)
    cdef struct read_block:
        int mode
    void rb_init(read_block* rb, char* buffer, int64_t size, int mode)
    void* rb_read(read_block* rb, int64_t sz)
    bint rb_get_integer(read_block *rb, int cookie, lua_Integer *pout) except 0
    bint rb_get_real(read_block *rb, double *pout) except 0
    bint rb_get_pointer(read_block *rb, void ** pout) except 0
    char* rb_get_string(read_block *rb, uint8_t ahead, size_t *psize) except NULL
    char *foreign_unhook(char* buf)

    # for write
    cdef struct write_block:
        char *buffer
        intptr_t nextbase
        int64_t len
        int mode
    void wb_init(write_block* wb, int mode)
    void wb_free(write_block *wb)
    void wb_nil(write_block* wb)
    void wb_boolean(write_block* wb, int v)
    void wb_put_integer(write_block* wb, lua_Integer v)
    void wb_put_real(write_block* wb, double v)
    void wb_put_string(write_block *wb, const char *ptr, int sz)
    void wb_put_pointer(write_block *wb, void *v)
    void wb_write(write_block *wb, const void *buf, int64_t sz)
    char** foreign_hook(char *buf)

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
    void pywb_put_PyArray(write_block *wb, object arr, object arr_iter)
    object pyrb_get_PyArray(read_block *wb, int cookie)

########################
# functions for unpack #
########################

cdef uint8_t* pyrb_unpack_one(l, read_block *rb, bint in_table) except *:
    cdef uint8_t *aheadptr = <uint8_t*>rb_read(rb, 1);
    cdef uint8_t ahead
    cdef int value_type
    cdef int cookie
    cdef char * strptr = NULL
    cdef size_t strlength = 0
    cdef void * lightuserdata = NULL
    cdef lua_Integer intvalue;
    cdef double doublevalue;
    if aheadptr == NULL:
        if in_table:
            raise Exception("invalid stream")
        return NULL
    ahead = aheadptr[0]
    value_type = ahead & 0x7
    cookie = ahead >> 3
    if value_type == TYPE_NIL:
        l.append(None)
    elif value_type == TYPE_BOOLEAN:
        l.append(cookie>0)
    elif value_type == TYPE_NUMBER:
        if (cookie == TYPE_NUMBER_REAL):
            rb_get_real(rb, &doublevalue)
            l.append(doublevalue)
        else:
            rb_get_integer(rb, cookie, &intvalue)
            l.append(intvalue)
    elif value_type == TYPE_SHORT_STRING or value_type == TYPE_LONG_STRING:
        strptr = rb_get_string(rb, ahead, &strlength)
        l.append(PyBytes_FromStringAndSize(strptr, strlength))
    elif value_type==TYPE_TABLE:
        pyrb_unpack_table(l, rb, cookie)
    elif value_type==TYPE_USERDATA:
        rb_get_pointer(rb, &lightuserdata)
        l.append(PyCapsule_New(lightuserdata, "cptr", NULL))
    elif value_type==TYPE_FOREIGN_USERDATA:
        arr = pyrb_get_PyArray(rb, cookie)
        if arr is None:
            raise Exception("invalid stream when unpacking arr")
        l.append(arr)
    else:
        raise Exception("invalid stream for value type exception")
    return aheadptr

cdef void pyrb_unpack_table(l, read_block *rb, lua_Integer array_size) except *:
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
        rb_get_integer(rb, cookie, &array_size)
    #l.append(t)
    next_l = []
    for i in range(1, array_size+1):
        pyrb_unpack_one(next_l, rb, 1)
    next_t = {}
    while True:
        pyrb_unpack_one(next_l, rb, 1)
        if next_l[-1] is None:
            next_l.pop()
            break
        pyrb_unpack_one(next_l, rb, 1)
        next_t[next_l[-2]] = next_l[-1]
        next_l.pop()
        next_l.pop()
    if not next_t:
        l.append(next_l)
    else:
        for i, v in enumerate(next_l):
            next_t[i + 1] = v
        l.append(next_t)

# extern
cdef pymode_unpack(int mode, capsule_or_bytes, py_sz):
    cdef const char *name
    cdef char *ptr
    cdef char *realbuffer
    cdef size_t sz
    cdef read_block rb
    if PyCapsule_CheckExact(capsule_or_bytes):
        name = PyCapsule_GetName(capsule_or_bytes)
        ptr = <char *>PyCapsule_GetPointer(capsule_or_bytes, name)
        sz = py_sz
        if strcmp(name, "cptr") != 0 and strcmp(name, "pyptr") != 0:
            raise Exception("capsule unpack failed for name = %s " % PyBytes_FromString(name))
    elif PyBytes_CheckExact(capsule_or_bytes):
        ptr = <char *>PyBytes_AS_STRING(capsule_or_bytes)
        sz = PyBytes_GET_SIZE(capsule_or_bytes)
    else:
        raise Exception("Unexcept type %s " % str(type(capsule_or_bytes)))
    realbuffer = foreign_unhook(ptr)
    rb_init(&rb, realbuffer, sz, mode);
    l = []
    while True:
        if pyrb_unpack_one(l, &rb, 0) == NULL:
            break
    if realbuffer != ptr:
        skynet_free(realbuffer)
    return tuple(l)

######################
# functions for pack #
######################
cdef void pywb_list(write_block* wb, list_obj, int depth) except *:
    cdef int array_size = len(list_obj)
    cdef uint8_t n
    if array_size >= MAX_COOKIE - 1:
        n = COMBINE_TYPE(TYPE_TABLE, MAX_COOKIE - 1);
        wb_write(wb, &n, 1);
        wb_put_integer(wb, array_size)
    else:
        n = COMBINE_TYPE(TYPE_TABLE, array_size);
        wb_write(wb, &n, 1);
    for v in list_obj:
        pywb_pack_one(wb,v,depth)
    wb_nil(wb)

cdef void pywb_dict(write_block* wb, dict_obj, int depth) except *:
    cdef uint8_t n = COMBINE_TYPE(TYPE_TABLE, 0);
    wb_write(wb, &n, 1);
    for k, v in dict_obj.items():
        pywb_pack_one(wb,k,depth)
        pywb_pack_one(wb,v,depth)
    wb_nil(wb)

cdef void pywb_pack_one(write_block* wb, py_arg, int depth) except *:
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
            wb_put_pointer(wb, PyCapsule_GetPointer(py_arg, "cptr"))
        else:
            wb_free(wb)
            raise Exception("unexception capsule")
    elif PyArray_CheckExact(py_arg):
        if not PyArray_foreign_check_typechar(py_arg):
            wb_free(wb)
            raise Exception("unexception typechar %s"%py_arg.dtype.char)
        if wb.mode == MODE_FOREIGN_REF:
            pywb_put_PyArray(wb, py_arg, None)
        elif wb.mode == MODE_FOREIGN_REMOTE:
            arr_iter = py_arg.flat
            pywb_put_PyArray(wb, py_arg, arr_iter)
        else:
            wb_free(wb)
            raise Exception("seri wb foreign unexception mode")
    elif isinstance(py_arg, dict):
        pywb_dict(wb, py_arg, depth+1)
    elif isinstance(py_arg, list):
        pywb_list(wb, py_arg, depth+1)
    else:
        py_arg_type = type(py_arg)
        if py_arg_type == int or np.issubdtype(py_arg_type, np.integer):
            wb_put_integer(wb, py_arg)
        elif py_arg_type == float or np.issubdtype(py_arg_type, np.floating):
            wb_put_real(wb, py_arg)
        elif py_arg_type == bool:
            wb_boolean(wb, py_arg)
        elif py_arg_type == bytes:
            bytes_sz = len(py_arg)
            bytes_ptr = py_arg
            wb_put_string(wb, bytes_ptr, bytes_sz)
        elif py_arg_type == bytearray:
            bytes_sz = len(py_arg)
            bytes_ptr = PyByteArray_AS_STRING(py_arg)
            wb_put_string(wb, bytes_ptr, bytes_sz)
        elif py_arg_type == str:
            py_arg_bytes = PyUnicode_AsUTF8String(py_arg)
            bytes_sz = len(py_arg_bytes)
            bytes_ptr = py_arg_bytes
            wb_put_string(wb, bytes_ptr, bytes_sz)
        else:
            wb_free(wb)
            raise Exception("Unsupport type %s to serialize"%str(py_arg_type))

cdef pymode_pack(int mode, argtuple):
    cdef write_block wb
    cdef char ** hookptr
    wb_init(&wb, mode)
    for one in argtuple:
        pywb_pack_one(&wb, one, 0)
    return PyCapsule_New(wb.buffer, "cptr", NULL), wb.len

#########################
# outside pack & unpack #
#########################

def tobytes(capsule, size_t size):
    cdef char *ptr = <char *>PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule))
    return ptr[:size]

def luapack(*args):
    return pymode_pack(MODE_LUA, args)

def luaunpack(capsule, size=None):
    return pymode_unpack(MODE_LUA, capsule, size)

def pack(*args):
    return pymode_pack(MODE_FOREIGN_REF, args)

def unpack(capsule, size=None):
    return pymode_unpack(MODE_FOREIGN_REF, capsule, size)

def remotepack(*args):
    return pymode_pack(MODE_FOREIGN_REMOTE, args)

def remoteunpack(capsule, size=None):
    return pymode_unpack(MODE_FOREIGN_REMOTE, capsule, size)

def trash(capsule, size_t sz):
    cdef void *ptr = PyCapsule_GetPointer(capsule, "cptr")
    skynet_free(ptr)
