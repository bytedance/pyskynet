#include "numsky/lua-numsky.h"
#include "numsky/lua-numsky_module.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"

/**************
 * dtype meta *
 **************/

static int
lnumsky_dtype__tostring(lua_State *L){
	struct numsky_dtype* dtype = luabinding::ClassUtil<numsky_dtype>::check(L, 1);
	char buffer[40];
	snprintf(buffer, sizeof(buffer), "dtype('%s')", dtype->name);
	lua_pushstring(L, buffer);
	return 1;
}

static int
lnumsky_dtype__call(lua_State *L){
	struct numsky_dtype* dtype = luabinding::ClassUtil<numsky_dtype>::check(L, 1);
	char dataptr[16];
	lnumsky_template_fp(L, dtype->typechar, numsky::dataptr_check)(L, dataptr, 2);
	lnumsky_template_fp(L, dtype->typechar, numsky::dataptr_push)(L, dataptr);
	return 1;
}

/********************
 * dtype properties *
 ********************/

static void lnumsky_dtype_num_getter(lua_State *L, struct numsky_dtype* dtype){
	lua_pushinteger(L, dtype->type_num);
}

static void lnumsky_dtype_char_getter(lua_State *L, struct numsky_dtype* dtype){
	lua_pushlstring(L, &dtype->typechar, 1);
}

static void lnumsky_dtype_name_getter(lua_State *L, struct numsky_dtype* dtype){
	lua_pushstring(L, dtype->name);
}

static void lnumsky_dtype_itemsize_getter(lua_State *L, struct numsky_dtype* dtype){
	lua_pushinteger(L, dtype->elsize);
}

namespace luabinding {
    template <> void Class_<numsky_dtype>::clazz(Class_<numsky_dtype>& c) {
		c.setFieldProperty("num", lnumsky_dtype_num_getter, NULL)
			.setFieldProperty("char", lnumsky_dtype_char_getter, NULL)
			.setFieldProperty("itemsize", lnumsky_dtype_itemsize_getter, NULL)
			.setFieldProperty("name", lnumsky_dtype_name_getter, NULL)
			.setMetaDefaultIndex()
			.setMetaFunction("__tostring", lnumsky_dtype__tostring)
			.setMetaFunction("__call", lnumsky_dtype__call);

		lua_State*L = c.L;
		luaL_getmetatable(L, luabinding::Class_<numsky_dtype>::metaname);
		for(size_t i=0;i<sizeof(NS_DTYPE_CHARS);i++){
			numsky_dtype *dtype = numsky_get_dtype_by_char(NS_DTYPE_CHARS[i]);
			luabinding::ClassUtil<numsky_dtype>::newwrap(L, dtype);
			lua_seti(L, -2, NS_DTYPE_CHARS[i]);
		}
		lua_pop(L, 1);
	}

    template <> int Class_<numsky_dtype>::ctor(lua_State*L) {
		return luaL_error(L, "ctor dtype not allow");
	}
} // namespace luabinding

