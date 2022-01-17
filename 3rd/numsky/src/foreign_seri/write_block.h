/*
	modify from https://github.com/cloudwu/lua-serialize
 */

#define LUA_LIB

#include "foreign_seri/seri.h"

struct write_block {
	char *buffer;
	intptr_t nextbase;
	int64_t capacity;
	int64_t len;
	int mode;
};

void wb_init(struct write_block *wb, int mode);
void wb_free(struct write_block *wb);
void wb_ref_base(struct write_block *wb, struct skynet_foreign* foreign_base, char *dataptr);
void wb_write(struct write_block *b, const void *buf, int64_t sz);
void wb_nil(struct write_block *wb);
void wb_boolean(struct write_block *wb, int boolean);
void wb_put_integer(struct write_block *wb, lua_Integer v);
void wb_put_real(struct write_block *wb, double v);
void wb_put_pointer(struct write_block *wb, void *v);
void wb_put_string(struct write_block *wb, const char *str, int len);
void wb_uint(struct write_block* wb, npy_intp v);

char** mode_hook(int mode, char *buffer, intptr_t nextbase);
int lmode_pack(int mode, lua_State *L);
