#include <string>

#include "numsky/lua-numsky.h"
#include "numsky/lua-numsky_module.h"

#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/ufunc/lua-numsky_ufunc.h"

/**********************
 * ndarray properties *
 **********************/
namespace numsky {
	void ThrowableContext::throw_func(const std::string & data) {
		luaL_error(L, "%s", data.c_str());
	}
}

static void
lnumsky_ndarray_ndim_getter(lua_State *L, struct numsky_ndarray* arr_obj){
	lua_pushinteger(L, arr_obj->nd);
}

static void
lnumsky_ndarray_shape_getter(lua_State *L, struct numsky_ndarray* arr_obj){
	numsky::new_tuple<npy_intp>(L, arr_obj->nd, arr_obj->dimensions);
}

static void
lnumsky_ndarray_strides_getter(lua_State *L, struct numsky_ndarray* arr_obj){
	numsky::new_tuple<npy_intp>(L, arr_obj->nd, arr_obj->strides);
}

static void
lnumsky_ndarray_dtype_getter(lua_State *L, struct numsky_ndarray* arr_obj){
	luaL_getmetatable(L, luabinding::Class_<numsky_dtype>::metaname);
	lua_geti(L, -1, arr_obj->dtype->typechar);
}

static int lnumsky_ndarray__len(lua_State* L) {
	auto arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	if(arr->nd <= 0) {
		return luaL_error(L, "can't get length for arr with nd == %d", arr->nd);
	} else {
		lua_pushinteger(L, arr->dimensions[0]);
		return 1;
	}
}

/**************
 * ndarray gc *
 **************/

static int lnumsky_ndarray__gc(lua_State* L) {
	auto arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	numsky_ndarray_destroy(arr);
	return 0;
}

template <numsky::UFUNC_ENUM ufunc_num> int __bop(lua_State* L) {
	return numsky::ufunc__call_21(L, const_cast<numsky_ufunc*>(&numsky::ufunc_instance<ufunc_num>::ufunc), 1, 2);
}

template <numsky::UFUNC_ENUM ufunc_num> int __uop(lua_State* L) {
	return numsky::ufunc__call_11(L, const_cast<numsky_ufunc*>(&numsky::ufunc_instance<ufunc_num>::ufunc), 1);
}

namespace luabinding {
    template <> void Class_<numsky_ndarray>::clazz(Class_<numsky_ndarray>&c) {
		c.setFieldProperty("ndim", lnumsky_ndarray_ndim_getter, NULL)
		   .setFieldProperty("shape", lnumsky_ndarray_shape_getter, NULL)
		   .setFieldProperty("shape", lnumsky_ndarray_shape_getter, NULL)
		   .setFieldProperty("strides", lnumsky_ndarray_strides_getter, NULL)
		   .setFieldProperty("dtype", lnumsky_ndarray_dtype_getter, NULL)

		   .setFieldFunction("flatten", numsky::ndarray_methods_flatten)
		   .setFieldFunction("reshape", numsky::ndarray_methods_reshape)
		   .setFieldFunction("copy", numsky::ndarray_methods_copy)
		   .setFieldFunction("astype", numsky::ndarray_methods_astype)
		   .setFieldFunction("roll", numsky::ndarray_methods_roll)

		   .setMetaFunction("__len", lnumsky_ndarray__len)
		   .setMetaFunction("__gc", lnumsky_ndarray__gc)
		   .setMetaFunction("__index", numsky::ndarray__index)
		   .setMetaFunction("__newindex", numsky::ndarray__newindex)
		   .setMetaFunction("__tostring", numsky::ndarray__tostring)

		   .setMetaFunction("__add", __bop<numsky::UFUNC_add>)
		   .setMetaFunction("__sub", __bop<numsky::UFUNC_sub>)
		   .setMetaFunction("__mul", __bop<numsky::UFUNC_mul>)
		   .setMetaFunction("__div", __bop<numsky::UFUNC_div>)
		   .setMetaFunction("__idiv", __bop<numsky::UFUNC_idiv>)
		   .setMetaFunction("__mod", __bop<numsky::UFUNC_mod>)
		   .setMetaFunction("__pow", __bop<numsky::UFUNC_pow>)

		   .setMetaFunction("__band", __bop<numsky::UFUNC_band>)
		   .setMetaFunction("__bor", __bop<numsky::UFUNC_bor>)
		   .setMetaFunction("__bxor", __bop<numsky::UFUNC_bxor>)
		   .setMetaFunction("__shl", __bop<numsky::UFUNC_shl>)
		   .setMetaFunction("__shr", __bop<numsky::UFUNC_shr>)

		   .setMetaFunction("__unm", __uop<numsky::UFUNC_unm>)
		   .setMetaFunction("__bnot", __uop<numsky::UFUNC_bnot>);

	}

    template <> int Class_<numsky_ndarray>::ctor(lua_State*L) {
		int type1 = lua_type(L, 1);
		if(type1 == LUA_TNONE) {
			return luaL_error(L, "array() required at least one argument");
		}
		char typechar = '\0';
		if(!lua_isnone(L, 2)) {
			struct numsky_dtype *dtype = luabinding::ClassUtil<numsky_dtype>::check(L, 2);
			typechar = dtype->typechar;
		}
		if(!lua_isnone(L, 3)) {
			luaL_error(L, "array({...}, dtype) required 1 or 2 argument but get 3");
		}
		numsky::ThrowableContext ctx(L);
		numsky::table_to_array<true>(&ctx, 1, typechar);
		return 1;
	}


} // namespace luabinding

