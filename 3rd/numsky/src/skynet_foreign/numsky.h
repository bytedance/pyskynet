

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "skynet_foreign/skynet_foreign.h"

#include "lua.h"
#include "lauxlib.h"
#include <stdlib.h>
#include <string.h>

#define NS_ARR_METANAME "numsky.ndarray"

/****************
 * numsky.dtype *
 ****************/
struct numsky_dtype {
	int type_num;                // NPY_TYPES
	char typechar;              // NPY_TYPECHAR
	char kind;
	int elsize;
	char *name;
	void (*dataptr_push)(lua_State*L, char* dataptr);
	void (*dataptr_check)(lua_State*L, char* dataptr, int stacki);
};

extern char NS_DTYPE_CHARS[10];
struct numsky_dtype *numsky_get_dtype_by_char(char typechar);

/******************
 * numsky.ndarray *
 ******************/
struct numsky_ndarray {
	struct skynet_foreign *foreign_base;
	char * dataptr;

    // some fields like PyArrayObject_fields in "ndarraytypes.h"
    struct numsky_dtype *dtype;        // pointer to a static obj in numsky_dtype.c
	int count;					// element count

    int nd;
    npy_intp *strides;
    npy_intp dimensions[0]; // dimensions's length may not equal with nd for indexing
};

static inline struct numsky_ndarray* numsky_ndarray_precreate(int nd, char typechar) {
	struct numsky_ndarray *arr = (struct numsky_ndarray*)malloc(sizeof(struct numsky_ndarray) + sizeof(npy_intp) * 2*nd);
	// 0. set base as NULL
	arr->foreign_base = NULL;
    // 1. get type
    arr->dtype = numsky_get_dtype_by_char(typechar);
    // 2. set nd, strides pointer
    arr->nd = nd;
	arr->strides = arr->dimensions + nd;
	return arr;
}

static inline void numsky_ndarray_destroy(struct numsky_ndarray* arr) {
	if(arr->foreign_base != NULL) {
		skynet_foreign_decref(arr->foreign_base);
	}
	free(arr);
}

static inline void numsky_ndarray_refdata(struct numsky_ndarray *arr, struct skynet_foreign *base, char* dataptr) {
	arr->foreign_base = base;
	arr->dataptr = dataptr;
}

// set count, strides, alloc data, need arr->dimensions setted
static inline void numsky_ndarray_autostridecountalloc(struct numsky_ndarray *arr) {
	int count = 1;
	for(int i=arr->nd-1;i>=0;i--){
		arr->strides[i] = arr->dtype->elsize*count;
		count*=arr->dimensions[i];
	}
	arr->count = count;
	arr->strides = arr->dimensions + arr->nd;
	struct skynet_foreign *foreign_base = skynet_foreign_newbytes(arr->dtype->elsize*count);
	numsky_ndarray_refdata(arr, foreign_base, foreign_base->data);
}

// set count, strides, need arr->dimensions setted
static inline void numsky_ndarray_autostridecount(struct numsky_ndarray *arr) {
    int count = 1;
    for(int i=arr->nd-1;i>=0;i--){
        arr->strides[i] = arr->dtype->elsize*count;
        count*=arr->dimensions[i];
    }
	arr->count = count;
}

// set count, need arr->dimensions setted
static inline void numsky_ndarray_autocount(struct numsky_ndarray *arr) {
    int count = 1;
    for(int i=arr->nd-1;i>=0;i--){
        count*=arr->dimensions[i];
    }
	arr->count = count;
}


void numsky_ndarray_copyfrom(struct numsky_ndarray *arr, char* buf);

void numsky_ndarray_copyto(struct numsky_ndarray *arr, char* buf);

/***************
 * numsky.nditer *
 ***************/
struct numsky_nditer {
	int nd;
	char * dataptr;
	struct numsky_ndarray *ao; /* ndarray object */
	npy_intp coordinates[0]; /* N-dimensional loop */
};

inline struct numsky_nditer* numsky_nditer_create(struct numsky_ndarray *arr_obj) {
	struct numsky_nditer *iter = (struct numsky_nditer*)malloc(sizeof(struct numsky_nditer) + sizeof(npy_intp) * arr_obj->nd);
    memset(iter->coordinates, 0, sizeof(npy_intp) * arr_obj->nd);
	iter->nd = arr_obj->nd;
	iter->ao = arr_obj;
	iter->dataptr = arr_obj->dataptr;
	return iter;
}

/* next, when next is never called, iter->dataptr == iter->ao->dataptr*/
inline void numsky_nditer_next(struct numsky_nditer *iter) {
	struct numsky_ndarray *ao = iter->ao;
	int i=iter->nd-1;
	for(;i>=0;i--){
		int dimi_m1 = ao->dimensions[i] - 1;
		if(iter->coordinates[i] < dimi_m1) {
			iter->coordinates[i] ++;
			iter->dataptr += ao->strides[i];
			return ;
		} else {
			iter->coordinates[i] = 0;
			iter->dataptr -= ao->strides[i] * dimi_m1;
		}
	}
}

inline void numsky_nditer_destroy(struct numsky_nditer *iter) {
	free(iter);
}

/* regard iter->dataptr as the start ptr of a sub array, get the ndim of sub array*/
int numsky_nditer_sub_ndim(struct numsky_nditer *iter);

/****************
 * numsky.slice *
 ****************/
struct numsky_slice {
	int start;		// 0 means None, negative for reverse index
	int stop;		// 0 means None, negative for reverse index
	int step;		// step == 0 for integer index, step !=0 for interval index
};

#ifdef __cplusplus
}
#endif
