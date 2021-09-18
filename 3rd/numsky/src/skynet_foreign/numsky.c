#include <string.h>
#include "skynet_foreign/skynet_foreign.h"
#include "skynet_foreign/numsky.h"

void numsky_ndarray_copyfrom(struct numsky_ndarray *arr, char* buf){
	struct numsky_nditer * iter = numsky_nditer_create(arr);
	for(long i=0;i<arr->count;numsky_nditer_next(iter), i++) {
		memcpy(iter->dataptr, buf, arr->dtype->elsize);
		buf += arr->dtype->elsize;
	}
	numsky_nditer_destroy(iter);
}

void numsky_ndarray_copyto(struct numsky_ndarray *arr, char* buf){
	struct numsky_nditer * iter = numsky_nditer_create(arr);
	for(long i=0;i<arr->count;numsky_nditer_next(iter), i++) {
		memcpy(buf, iter->dataptr, arr->dtype->elsize);
		buf += arr->dtype->elsize;
	}
	numsky_nditer_destroy(iter);
}
