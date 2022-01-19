
#pragma once

#include "skynet_foreign/numsky.h"

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>

#define TYPE_NIL 0
#define TYPE_BOOLEAN 1
// hibits 0 false 1 true
#define TYPE_NUMBER 2
// hibits 0 : 0 , 1: byte, 2:word, 4: dword, 6: qword, 8 : double
#define TYPE_NUMBER_ZERO 0
#define TYPE_NUMBER_BYTE 1
#define TYPE_NUMBER_WORD 2
#define TYPE_NUMBER_DWORD 4
#define TYPE_NUMBER_QWORD 6
#define TYPE_NUMBER_REAL 8

#define TYPE_USERDATA 3
#define TYPE_SHORT_STRING 4
// hibits 0~31 : len
#define TYPE_LONG_STRING 5
#define TYPE_TABLE 6
#define TYPE_FOREIGN_USERDATA 7

#define MAX_COOKIE 32
#define COMBINE_TYPE(t,v) ((t) | (v) << 3)

#define MAX_DEPTH 32

#define MODE_LUA 0
#define MODE_FOREIGN_REF 1
#define MODE_FOREIGN_REMOTE 2

union fbuf_i64 {
    int64_t i_val;
    char *p_val;
};

inline bool fbuf_isbuffer(union fbuf_i64 *fbuf_header) {
    return (fbuf_header->i_val == 0) || ((fbuf_header->i_val & 1) != 0);
}

inline bool fbuf_needhook(union fbuf_i64 *fbuf_header) {
    return fbuf_header->i_val & 1;
}

inline int64_t fbuf_nextbase_get(union fbuf_i64 *ptr) {
    return ptr->i_val >> 1;
}

inline void fbuf_lastbase_put(union fbuf_i64 *ptr, int64_t lastbase) {
    ptr->i_val = (lastbase << 1) | 1;
}

inline void foreign_unref(char *buffer) {
    union fbuf_i64* p_header = (union fbuf_i64*)buffer;
	if(fbuf_isbuffer(p_header)) {
        int64_t nextbase = fbuf_nextbase_get(p_header);
        while(nextbase > 0) {
            struct skynet_foreign *foreign_base;
            memcpy(&foreign_base, buffer+nextbase, sizeof(void*));
            skynet_foreign_decref(foreign_base);
            nextbase = fbuf_nextbase_get((union fbuf_i64*)(buffer+nextbase+2*sizeof(void*)));
        }
	}
}

inline char** foreign_hook(char *buffer) {
    union fbuf_i64 *p_header = (union fbuf_i64*)buffer;
	if(fbuf_needhook(p_header)) {
		union fbuf_i64 *p_hook = skynet_malloc(sizeof(union fbuf_i64));
        p_hook->p_val = buffer;
		return (char**)p_hook;
	} else {
		return NULL;
	}
}

inline char *mode_unhook(int mode, char* buffer) {
	if(mode == MODE_LUA) {
		return buffer;
	} else {
        union fbuf_i64 *p_header = (union fbuf_i64*)buffer;
		if(fbuf_isbuffer(p_header)) {
			return buffer;
		} else {
			return p_header->p_val;
		}
	}
}
