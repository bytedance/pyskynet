
#include "skynet_foreign/skynet_foreign.h"
#include "skynet_foreign/numsky.h"
#include "Python.h"
#include "numpy/arrayobject.h"

#include "foreign_seri/seri.h"
#include "foreign_seri/read_block.h"
#include "foreign_seri/write_block.h"

static bool PyArray_foreign_check_typechar(PyObject *py_obj){
    PyArrayObject *arr = (PyArrayObject*)(py_obj);
	char type = PyArray_DESCR(arr)->type;
    for(int i=0;i<sizeof(NS_DTYPE_CHARS);i++) {
        if(NS_DTYPE_CHARS[i] == type) {
			return true;
		}
	}
	return false;
}

static void foreign_capsule_destructor(PyObject* capsule) {
    struct skynet_foreign *ptr = (struct skynet_foreign*)PyCapsule_GetPointer(capsule, SF_METANAME);
    skynet_foreign_decref(ptr);
}


/*
// if base is foreign, create a capsule as base,
// if base is pyobj, return pyobj as base
static PyObject *__foreign_check_base(struct skynet_foreign* high_obj, const char *capsule_name) {
	PyObject *base;
	struct skynet_foreign* low_obj;
	if(high_obj->ref_type == SF_REF_FOREIGN) {
		low_obj = (struct skynet_foreign*)high_obj->foreign_base;
	} else {
		low_obj = high_obj;
	}
	if(low_obj->ref_type == SF_REF_PYTHON) {
		base = low_obj->foreign_base;
		Py_INCREF(base);
		skynet_foreign_decref(high_obj);
	} else if(low_obj->ref_type == SF_REF_ALLOC) {
		base = PyCapsule_New(low_obj, capsule_name, foreign_capsule_destructor);
		if(low_obj != high_obj) {
			skynet_foreign_incref(low_obj);
			skynet_foreign_decref(high_obj);
		}
	} else {
		printf("[ERROR]sf_obj has wrong flags!!!!\n");
		base = NULL;
	}
	return base;
}
*/

// array, skynet to python
static PyObject *unpack_PyArray(struct read_block *rb, int cookie) {
	struct numsky_ndarray *ns_arr = unpack_ns_arr(rb, cookie);
	if(ns_arr == NULL) {
		Py_INCREF(Py_None);
		return ((PyObject *)Py_None);
	}
	PyObject *base = PyCapsule_New(ns_arr->foreign_base, SF_METANAME, foreign_capsule_destructor);
	PyArrayObject *arr = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(ns_arr->dtype->type_num),
			ns_arr->nd, ns_arr->dimensions, ns_arr->strides, ns_arr->dataptr, 0 | NPY_ARRAY_WRITEABLE, NULL);
	PyArray_SetBaseObject(arr, base);
	numsky_ndarray_refdata(ns_arr, NULL, NULL);
	numsky_ndarray_destroy(ns_arr);
	return (PyObject*)arr;
}

static void wb_foreign_PyArray(struct write_block *wb, PyObject *py_obj, PyObject *py_arr_iter) {
    PyArrayObject *arr = (PyArrayObject*)(py_obj);
	int nd = PyArray_NDIM(arr);
	if(nd >= MAX_COOKIE) {
		printf("nd > 31 in python, error TODO\n");
	}
	// 1. nd & type
	uint8_t n = COMBINE_TYPE(TYPE_FOREIGN_USERDATA, nd);
	wb_write(wb, &n, 1);
	// 2. typechar
	char typechar = PyArray_DESCR(arr)->type;
	wb_write(wb, &typechar, 1);
	// 3. dimension
    for(int i=0;i<nd;i++) {
		wb_uint(wb, PyArray_DIMS(arr)[i]);
	}
	if(wb->mode == MODE_FOREIGN_REF) {
		// 4. strides
		wb_write(wb, PyArray_STRIDES(arr), sizeof(npy_intp)*nd);
		// 5. foreign_base & dataptr
		Py_INCREF(py_obj);
		struct skynet_foreign* foreign_base = skynet_foreign_newrefpy(py_obj, PyArray_DATA(arr), SF_FLAGS_WRITEABLE);
		wb_write(wb, &foreign_base, sizeof(foreign_base));
		wb_write(wb, &(foreign_base->data), sizeof(foreign_base->data));
	} else if(wb->mode == MODE_FOREIGN_REMOTE) {
		//  value seri
		PyArrayIterObject *arr_iter = (PyArrayIterObject*)(py_arr_iter);
		// 4. data
		int itemsize = PyArray_ITEMSIZE(arr);
		while(PyArray_ITER_NOTDONE(arr_iter)) {
			void *ptr=PyArray_ITER_DATA(arr_iter);
			wb_write(wb, ptr, itemsize);
			PyArray_ITER_NEXT(arr_iter);
		}
	}
}
