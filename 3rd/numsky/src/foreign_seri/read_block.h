
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
bool rb_get_integer(struct read_block *rb, int cookie, lua_Integer *pout);
bool rb_get_real(struct read_block *rb, double * pout);
bool rb_get_pointer(struct read_block *rb, void ** pout);
char* rb_get_string(struct read_block *rb, uint8_t ahead, size_t *psize);
struct numsky_ndarray* rb_get_nsarr(struct read_block *rb, int nd);

char *mode_unhook(int mode, char* buffer);
int lmode_unpack(int mode, lua_State *L);
