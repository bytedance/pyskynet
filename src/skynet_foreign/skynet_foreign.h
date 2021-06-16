
#pragma once

#include "skynet_foreign/hook_skynet_py.h"

#include <stdbool.h>
#include <stdint.h>

#define SF_METANAME "skynet.foreign"
#define SF_FLAGS_WRITEABLE 1

enum skynet_foreign_reftype {
	SF_REF_SELF=0, // just self
	//SF_REF_UNSAFE=1, //  ref a unsafe ptr
	//SF_REF_MANAGED=2, //  ref a managed ptr
#ifdef BUILD_FOR_PYSKYNET // for python
	SF_REF_PYTHON=3, // ref a python obj
#endif
};

struct skynet_foreign {
#ifdef BUILD_FOR_PYSKYNET // for python
	struct spinlock lock;
#endif
    uint8_t flags;
	enum skynet_foreign_reftype ref_type;
    int ref_count;
	void *ref_obj; // PyObject* or other managed item
	char *data;
	int64_t __data[0];
};

/**********
 * flags *
**********/
static inline void skynet_foreign_ENABLEFLAGS(struct skynet_foreign* obj, uint8_t flags) {
    obj->flags |= flags;
}

static inline void skynet_foreign_CLEARFLAGS(struct skynet_foreign* obj, uint8_t flags) {
    obj->flags &= ~flags;
}

static inline bool skynet_foreign_CHKFLAGS(struct skynet_foreign* obj, uint8_t flags) {
    return (obj->flags & flags) == flags;
}

/**********
 * init *
**********/
static inline struct skynet_foreign* skynet_foreign_newbytes(size_t data_size) {
	struct skynet_foreign *obj = (struct skynet_foreign*)foreign_malloc(sizeof(struct skynet_foreign) + data_size);
    SPIN_INIT(obj);
	obj->flags = SF_FLAGS_WRITEABLE;
    obj->ref_count = 1;
	obj->ref_type = SF_REF_SELF;
	obj->ref_obj = NULL;
	obj->data = (char*)obj->__data;
	return obj;
}

#ifdef BUILD_FOR_PYSKYNET // for python
// steal pyobj's ref
static inline struct skynet_foreign* skynet_foreign_newrefpy(void *pyobj, char *data, uint8_t flags) {
	struct skynet_foreign *obj = (struct skynet_foreign*)foreign_malloc(sizeof(struct skynet_foreign));
    SPIN_INIT(obj);
    obj->flags = flags;
	obj->ref_count = 1;
	obj->ref_type = SF_REF_PYTHON;
	obj->ref_obj = pyobj;
	obj->data = data;
	return obj;
}
#endif

void skynet_foreign_incref(struct skynet_foreign *obj);

void skynet_foreign_decref(struct skynet_foreign *obj);

