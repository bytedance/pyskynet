#include "numsky/lua-numsky.h"

#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/ufunc/lua-numsky_ufunc.h"

/********************
 * ufunc properties *
 ********************/

static void lnumsky_ufunc_nin_getter(lua_State *L, numsky_ufunc* ufunc) {
	lua_pushinteger(L, ufunc->nin);
}

static void lnumsky_ufunc_nout_getter(lua_State *L, numsky_ufunc* ufunc) {
	lua_pushinteger(L, ufunc->nout);
}

static void lnumsky_ufunc_nargs_getter(lua_State *L, numsky_ufunc* ufunc) {
	lua_pushinteger(L, ufunc->nin + ufunc->nout);
}

/*******************
 * ufunc functions *
 *******************/

static int lnumsky_ufunc__tostring(lua_State *L) {
	numsky_ufunc* ufunc = luabinding::ClassUtil<numsky_ufunc>::check(L, 1);
	char buffer[40];
	sprintf(buffer, "ufunc('%s')", ufunc->name);
	lua_pushstring(L, buffer);
	return 1;
}

static int lnumsky_ufunc__call(lua_State *L) {
	numsky_ufunc* ufunc = luabinding::ClassUtil<numsky_ufunc>::check(L, 1);
	if(ufunc->nout==1) {
		if(ufunc->nin==2) {
			return numsky::ufunc__call_21(L, ufunc, 2, 3);
		} else if (ufunc->nout==1) {
			return numsky::ufunc__call_11(L, ufunc, 2);
		} else {
			luaL_error(L, "ufunc.nout must be 1 or 2");
		}
	}
	return ufunc->nout;
}

/**************************
 * ufunc instance binding *
 **************************/

template <numsky::UFUNC_ENUM N> void binding_ufunc(lua_State *L, int ufunc_meta_stacki) {
	numsky_ufunc *ptr = const_cast<numsky_ufunc*>(&(numsky::ufunc_instance<N>::ufunc));
	luabinding::ClassUtil<numsky_ufunc>::newwrap(L, ptr);
	lua_seti(L, ufunc_meta_stacki, N);
	binding_ufunc<(numsky::UFUNC_ENUM)(N-1)>(L, ufunc_meta_stacki);
}

// enum UFUNC_add -> UFUNC_shr
template <> void binding_ufunc<numsky::UFUNC_add>(lua_State*L, int ufunc_meta_stacki) {
	numsky_ufunc *ptr = const_cast<numsky_ufunc*>(&(numsky::ufunc_instance<numsky::UFUNC_add>::ufunc));
	luabinding::ClassUtil<numsky_ufunc>::newwrap(L, ptr);
	lua_seti(L, ufunc_meta_stacki, numsky::UFUNC_add);
}

// enum UFUNC_unm -> UFUNC_sqrt
template <> void binding_ufunc<numsky::UFUNC_unm>(lua_State*L, int ufunc_meta_stacki) {
	numsky_ufunc *ptr = const_cast<numsky_ufunc*>(&(numsky::ufunc_instance<numsky::UFUNC_unm>::ufunc));
	luabinding::ClassUtil<numsky_ufunc>::newwrap(L, ptr);
	lua_seti(L, ufunc_meta_stacki, numsky::UFUNC_unm);
}

namespace luabinding {
    template <> void Class_<numsky_ufunc>::clazz(Class_<numsky_ufunc> & c) {

		c.setFieldProperty("nin", lnumsky_ufunc_nin_getter, NULL)
		   .setFieldProperty("nout", lnumsky_ufunc_nout_getter, NULL)
		   .setFieldProperty("nargs", lnumsky_ufunc_nargs_getter, NULL)
		   .setFieldFunction("reduce", numsky::ufunc_reduce)
		   .setMetaFunction("__tostring", lnumsky_ufunc__tostring)
		   .setMetaFunction("__call", lnumsky_ufunc__call)
		   .setMetaDefaultIndex();

		lua_State*L = c.L;
		luaL_getmetatable(L, luabinding::Class_<numsky_ufunc>::metaname);
		int ufunc_meta_stacki = lua_gettop(L);
		binding_ufunc<numsky::UFUNC_atan2>(L, ufunc_meta_stacki);
		binding_ufunc<numsky::UFUNC_sqrt>(L, ufunc_meta_stacki);
		lua_pop(L, 1);
	}

    template <> int Class_<numsky_ufunc>::ctor(lua_State*L) {
		return luaL_error(L, "ctor ufunc not allow");
	}
}

