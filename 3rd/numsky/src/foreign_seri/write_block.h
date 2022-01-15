/*
	modify from https://github.com/cloudwu/lua-serialize
 */

#define LUA_LIB

#include "foreign_seri/seri.h"

struct block {
	struct block * next;
	char buffer[BLOCK_SIZE];
};

struct write_block {
	struct block * head;
	struct block * current;
	int len;
	int ptr;
	bool refarr;
};

void wb_write(struct write_block *b, const void *buf, int sz);
void wb_init(struct write_block *wb , struct block *b, bool refarr);
void wb_free(struct write_block *wb);
void wb_nil(struct write_block *wb);
void wb_boolean(struct write_block *wb, int boolean);
void wb_integer(struct write_block *wb, lua_Integer v);
void wb_real(struct write_block *wb, double v);
void wb_pointer(struct write_block *wb, void *v);
void wb_string(struct write_block *wb, const char *str, int len);

int lua_pack(lua_State *L, bool refarr);
