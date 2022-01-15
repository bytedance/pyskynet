
#pragma once

#include "foreign_seri/seri.h"

struct read_block {
	char * buffer;
	int len;
	int ptr;
	bool refarr;
};

void rball_init(struct read_block * rb, char * buffer, int size, bool refarr);
void *rb_read(struct read_block *rb, int sz);

int lua_unpack(lua_State *L, bool refarr);
