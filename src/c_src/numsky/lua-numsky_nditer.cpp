#include "numsky/lua-numsky.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/lua-numsky_module.h"

static int lnumsky_nditer__call(lua_State *L) {
	auto iter = luabinding::ClassUtil<numsky_nditer>::check(L, 1);
	if(lua_isnoneornil(L, 3) && iter->ao->count>0) {
		lnumsky_template_fp(L, iter->ao->dtype->typechar, numsky::dataptr_push)(L, iter->dataptr);
		return 1;
	} else {
		numsky_nditer_next(iter);
		if(iter->dataptr == iter->ao->dataptr) { // means another loop
			lua_pushnil(L);
			return 1;
		} else {
			lnumsky_template_fp(L, iter->ao->dtype->typechar, numsky::dataptr_push)(L, iter->dataptr);
			return 1;
		}
	}
}

namespace luabinding {
    template <> void Class_<numsky_nditer>::clazz(Class_<numsky_nditer>&c) {
		c.setMetaFunction("__call", lnumsky_nditer__call)
			.setMetaDefaultGC(numsky_nditer_destroy);
	}

    template <> int Class_<numsky_nditer>::ctor(lua_State*L) {
		auto arr_obj = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
		auto iter_obj = numsky_nditer_create(arr_obj);
		luabinding::ClassUtil<numsky_nditer>::newwrap(L, iter_obj);
		lua_pushvalue(L, 1);
		lua_setuservalue(L, 2);
		return 1;
	}
} // namespace luabinding
