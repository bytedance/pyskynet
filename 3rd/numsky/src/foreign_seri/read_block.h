
#pragma once

#include "foreign_seri/seri.h"

struct read_block {
	char * buffer;
	intptr_t nextbase;
	int64_t len;
	int64_t ptr;
	int mode;
};

void rb_init(struct read_block * rb, char * buffer, int64_t size, int mode);
void *rb_read(struct read_block *rb, int64_t sz);
struct numsky_ndarray* unpack_ns_arr(struct read_block *rb, int nd);

int mode_unpack(lua_State *L, int mode);
