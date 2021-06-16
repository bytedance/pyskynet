#pragma once


// 1. malloc
#include <stdlib.h>
#define foreign_malloc malloc
#define foreign_free free



#ifdef BUILD_FOR_PYSKYNET

// 2. spinlock
#include "spinlock.h"

// 3. npy
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#else

// spinlock do nothing
#define SPIN_INIT(q)
#define SPIN_LOCK(q)
#define SPIN_UNLOCK(q)
#define SPIN_DESTROY(q)

#define npy_intp long

// TODO not right...

#define NPY_BOOL 111

#define NPY_INT8 112
#define NPY_UINT8 113

#define NPY_INT16 114
#define NPY_UINT16 115

#define NPY_INT32 116
#define NPY_UINT32 117

#define NPY_INT64 118
#define NPY_UINT64 119

#define NPY_FLOAT32 120
#define NPY_FLOAT64 121

#endif

