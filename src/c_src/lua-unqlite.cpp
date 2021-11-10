#define LUA_LIB

#include "lua-binding.h"
#include "unqlite.h"
#include <string>

#define BUFFER_LENGTH 1024*1024

struct unqlite_ext {
	unqlite * db;
	bool closed;
};

struct unqlite_cursor_ext {
    unqlite *db;
	unqlite_kv_cursor *cursor;
	char *buffer;
};

namespace luabinding {
	template <> struct ClassTypeVariable<unqlite_ext> {
	   using ispointer = std::false_type;
	};
	template <> struct ClassTypeVariable<unqlite_cursor_ext> {
	   using ispointer = std::false_type;
	};
	template <> const char* Class_<unqlite_ext>::metaname= "unqlite.db";
	template <> const char* Class_<unqlite_cursor_ext>::metaname= "unqlite.cursor";
}

// unqlite.db
namespace luabinding {

    template <> void Class_<unqlite_ext>::clazz(Class_<unqlite_ext> & c) {
		c.setMetaDefaultIndex()
		.setFieldFunction("commit", [](lua_State*L) -> int {
				unqlite_ext *pDbExt = luabinding::ClassUtil<unqlite_ext>::check(L, 1);
				int rc = unqlite_commit(pDbExt->db);
				if(rc != UNQLITE_OK) {
					return luaL_error(L, "unqlite commit failed %d", rc);
				}
				return 0;
			})
		.setFieldFunction("store", [](lua_State*L) -> int {
				unqlite_ext *pDbExt = luabinding::ClassUtil<unqlite_ext>::check(L, 1);
				size_t keyLen = 0;
				const char * key = luaL_checklstring(L, 2, &keyLen);
				size_t valueLen = 0;
				const char * value = luaL_checklstring(L, 3, &valueLen);
				int rc = unqlite_kv_store(pDbExt->db,key,keyLen,value,valueLen);
				if(rc != UNQLITE_OK) {
					return luaL_error(L, "unqlite store failed %d", rc);
				}
				return 0;
			})
		.setFieldFunction("append", [](lua_State*L) -> int {
				unqlite_ext *pDbExt = luabinding::ClassUtil<unqlite_ext>::check(L, 1);
				size_t keyLen = 0;
				const char * key = luaL_checklstring(L, 2, &keyLen);
				size_t valueLen = 0;
				const char * value = luaL_checklstring(L, 3, &valueLen);
				int rc = unqlite_kv_append(pDbExt->db,key,keyLen,value,valueLen);
				if(rc != UNQLITE_OK) {
					return luaL_error(L, "unqlite append failed %d", rc);
				}
				return 0;
			})
		.setFieldFunction("fetch", [](lua_State*L) -> int {
				unqlite_ext *pDbExt = luabinding::ClassUtil<unqlite_ext>::check(L, 1);
				size_t keyLen = 0;
				const char * key = luaL_checklstring(L, 2, &keyLen);
				unqlite_int64 valueLen = BUFFER_LENGTH;
				std::unique_ptr<char[]> value(new char[valueLen]);
				int rc = unqlite_kv_fetch(pDbExt->db,key,keyLen,value.get(),&valueLen);
                if(rc == UNQLITE_NOTFOUND) {
                    lua_pushnil(L);
                    return 1;
                } else if(rc == UNQLITE_OK) {
                    lua_pushlstring(L, value.get(), valueLen);
                    return 1;
				} else {
					return luaL_error(L, "unqlite fetch failed %d", rc);
                }
			})
		.setFieldFunction("delete", [](lua_State*L) -> int {
				unqlite_ext *pDbExt = luabinding::ClassUtil<unqlite_ext>::check(L, 1);
				size_t keyLen = 0;
				const char * key = luaL_checklstring(L, 2, &keyLen);
				int rc = unqlite_kv_delete(pDbExt->db,key,keyLen);
				if(rc != UNQLITE_OK) {
					return luaL_error(L, "unqlite delete failed %d", rc);
				}
				return 0;
			})
		.setFieldFunction("cursor", [](lua_State*L) -> int {
				unqlite_ext *pDbExt = luabinding::ClassUtil<unqlite_ext>::check(L, 1);
				size_t keyLen = 0;
				unqlite_kv_cursor *pCur;
				int rc = unqlite_kv_cursor_init(pDbExt->db,&pCur);
				if(rc != UNQLITE_OK) {
					return luaL_error(L, "unqlite create cursor failed %d", rc);
				}
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::newalloc(L);
                pCurExt->db = pDbExt->db;
                pCurExt->cursor = pCur;
                pCurExt->buffer = new char[BUFFER_LENGTH];
				lua_pushvalue(L, 1);
				lua_setuservalue(L, -2);
				return 1;
			})
		.setFieldFunction("close", [](lua_State*L) -> int {
				unqlite_ext *pDbExt = luabinding::ClassUtil<unqlite_ext>::check(L, 1);
				int rc = unqlite_close(pDbExt->db);
				if(rc != UNQLITE_OK) {
					return luaL_error(L, "unqlite close failed %d", rc);
				}
				pDbExt->closed = true;
				return 0;
			})
		.setMetaFunction("__gc", [](lua_State*L) -> int {
				unqlite_ext *pDbExt = luabinding::ClassUtil<unqlite_ext>::check(L, 1);
				if(!pDbExt->closed) {
					unqlite_close(pDbExt->db);
					pDbExt->closed = true;
				}
				return 0;
			});
	}

    template <> int Class_<unqlite_ext>::ctor(lua_State*L) {
		size_t len = 0;
		const char * name = luaL_checklstring(L, 1, &len);
		unqlite *pDb;
		int rc = unqlite_open(&pDb, name, UNQLITE_OPEN_CREATE | UNQLITE_OPEN_OMIT_JOURNALING);
		if( rc != UNQLITE_OK ) {
			return luaL_error(L, "unqlite open failed %d", rc);
		}
		unqlite_ext *pDbExt = luabinding::ClassUtil<unqlite_ext>::newalloc(L);
		pDbExt->db = pDb;
		pDbExt->closed = false;
		return 1;
	}

}

int unqlite_readonly(lua_State*L) {
	size_t len = 0;
	const char * name = luaL_checklstring(L, 1, &len);
	unqlite *pDb;
	int rc = unqlite_open(&pDb, name, UNQLITE_OPEN_READONLY | UNQLITE_OPEN_OMIT_JOURNALING);
	if( rc != UNQLITE_OK ) {
		return luaL_error(L, "unqlite open failed %d", rc);
	}
	unqlite_ext *pDbExt = luabinding::ClassUtil<unqlite_ext>::newalloc(L);
	pDbExt->db = pDb;
	pDbExt->closed = false;
	return 1;
}

// unqlite.cursor
namespace luabinding {
    template <> void Class_<unqlite_cursor_ext>::clazz(Class_<unqlite_cursor_ext> & c) {
		c.setMetaDefaultIndex()
		.setFieldFunction("seek", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
				size_t keyLen = 0;
				const char * key = luaL_checklstring(L, 2, &keyLen);
				size_t matchLen = 0;
				const char * matchStr = luaL_checklstring(L, 2, &matchLen);
				int rc;
				int seek_match;
				if(matchLen != 2) {
					return luaL_error(L, "seek match must be ==, >=, <=.");
				} else if(matchStr[1] != '='){
					return luaL_error(L, "seek match must be ==, >=, <=..");
				} else {
					switch(matchStr[0]) {
						case '=': seek_match = UNQLITE_CURSOR_MATCH_EXACT; break;
						case '>': seek_match = UNQLITE_CURSOR_MATCH_GE; break;
						case '<': seek_match = UNQLITE_CURSOR_MATCH_LE; break;
						default: return luaL_error(L, "seek match must be ==, >=, <=...");
					}
				}
                rc = unqlite_kv_cursor_seek(pCurExt->cursor, key, keyLen, seek_match);
                if(rc != UNQLITE_OK) {
                    return luaL_error(L, "cursor seek error %d", rc);
                }
                return 0;
            })
		.setFieldFunction("first", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
                int rc = unqlite_kv_cursor_first_entry(pCurExt->cursor);
                if(rc != UNQLITE_OK) {
                    return luaL_error(L, "cursor first error %d", rc);
                }
                return 0;
            })
		.setFieldFunction("last", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
                int rc = unqlite_kv_cursor_last_entry(pCurExt->cursor);
                if(rc != UNQLITE_OK) {
                    return luaL_error(L, "cursor last error %d", rc);
                }
                return 0;
            })
		.setFieldFunction("valid", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
                int result = unqlite_kv_cursor_valid_entry(pCurExt->cursor);
				lua_pushboolean(L, result);
                return 1;
            })
		.setFieldFunction("next", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
                int rc = unqlite_kv_cursor_next_entry(pCurExt->cursor);
                if(rc != UNQLITE_OK) {
                    return luaL_error(L, "cursor next error %d", rc);
                }
                return 0;
            })
		.setFieldFunction("prev", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
                int rc = unqlite_kv_cursor_prev_entry(pCurExt->cursor);
                if(rc != UNQLITE_OK) {
                    return luaL_error(L, "cursor prev error %d", rc);
                }
                return 0;
            })
		.setFieldFunction("delete", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
                int rc = unqlite_kv_cursor_delete_entry(pCurExt->cursor);
                if(rc != UNQLITE_OK) {
                    return luaL_error(L, "cursor delete failed %d", rc);
                }
                return 0;
            })
		.setFieldFunction("key", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
				int keyLen = BUFFER_LENGTH;
                int rc = unqlite_kv_cursor_key(pCurExt->cursor, pCurExt->buffer, &keyLen);
                if(rc != UNQLITE_OK) {
                    return luaL_error(L, "cursor get key failed %d", rc);
                }
				lua_pushlstring(L, pCurExt->buffer, keyLen);
                return 1;
            })
		.setFieldFunction("value", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
				unqlite_int64 valueLen = BUFFER_LENGTH;
                int rc = unqlite_kv_cursor_data(pCurExt->cursor, pCurExt->buffer, &valueLen);
                if(rc != UNQLITE_OK) {
                    return luaL_error(L, "cursor get value failed %d", rc);
                }
				lua_pushlstring(L, pCurExt->buffer, valueLen);
                return 1;
            })
		.setFieldFunction("release", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
                int rc = unqlite_kv_cursor_release(pCurExt->db, pCurExt->cursor);
                if(rc != UNQLITE_OK) {
                    return luaL_error(L, "cursor release failed %d", rc);
                }
				delete [] pCurExt->buffer;
                pCurExt->buffer = NULL;
                return 0;
            })
		.setMetaFunction("__gc", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
                if(pCurExt->buffer!=NULL) {
                    unqlite_kv_cursor_release(pCurExt->db, pCurExt->cursor);
					delete [] pCurExt->buffer;
					pCurExt->buffer = NULL;
                }
                return 0;
            })
		.setMetaFunction("__call", [](lua_State*L) -> int {
				unqlite_cursor_ext *pCurExt = luabinding::ClassUtil<unqlite_cursor_ext>::check(L, 1);
				int rc;
				if(lua_isnoneornil(L, 3)) {
					rc = unqlite_kv_cursor_first_entry(pCurExt->cursor);
					if(rc == UNQLITE_DONE) {
						unqlite_kv_cursor_release(pCurExt->db, pCurExt->cursor);
						delete [] pCurExt->buffer;
						pCurExt->buffer = NULL;
						return 0;
					} else if(rc != UNQLITE_OK) {
						return luaL_error(L, "cursor first error %d", rc);
					}
				} else {
					rc = unqlite_kv_cursor_next_entry(pCurExt->cursor);
					if(rc == UNQLITE_DONE) {
						unqlite_kv_cursor_release(pCurExt->db, pCurExt->cursor);
						delete [] pCurExt->buffer;
						pCurExt->buffer = NULL;
						return 0;
					} else if(rc != UNQLITE_OK) {
						return luaL_error(L, "cursor next error %d", rc);
					}
				}
				int valid = unqlite_kv_cursor_valid_entry(pCurExt->cursor);
				if(!valid) {
					unqlite_kv_cursor_release(pCurExt->db, pCurExt->cursor);
					delete [] pCurExt->buffer;
					pCurExt->buffer = NULL;
					return 0;
				}
				int keyLen = BUFFER_LENGTH;
				rc = unqlite_kv_cursor_key(pCurExt->cursor, pCurExt->buffer, &keyLen);
				if(rc != UNQLITE_OK) {
					return luaL_error(L, "cursor get key failed %d", rc);
				}
				lua_pushlstring(L, pCurExt->buffer, keyLen);
				unqlite_int64 valueLen = BUFFER_LENGTH;
				rc = unqlite_kv_cursor_data(pCurExt->cursor, pCurExt->buffer, &valueLen);
				if(rc != UNQLITE_OK) {
					return luaL_error(L, "cursor get key failed %d", rc);
				}
				lua_pushlstring(L, pCurExt->buffer, valueLen);
				return 2;
            });
    }

    template <> int Class_<unqlite_cursor_ext>::ctor(lua_State*L) {
        return luaL_error(L, "never create unqlite_cursor_ext by ctor");
    }
}

extern "C" {
	LUAMOD_API int luaopen_unqlite(lua_State* L) {
		luabinding::Module_ m(L);
		m.start();

        // 1. unqlite database
		luabinding::ClassUtil<unqlite_ext>::bind(L);
		m.setFunction("open", luabinding::Class_<unqlite_ext>::ctor);
		m.setFunction("readonly", unqlite_readonly);

        // 2. unqlite cursor
		luabinding::ClassUtil<unqlite_cursor_ext>::bind(L);

		m.finish();
		return 1;
	}

}
