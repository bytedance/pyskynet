#define LUA_LIB

#include "numsky/lua-numsky.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/ufunc/lua-numsky_ufunc.h"
#include "numsky/tinygl/lua-numsky_tinygl.h"
#include "numsky/canvas/AstNode.h"

extern "C" {
	LUA_API void lnsarr_copyto(lua_State*L, int stackPos, char typechar, char *dataptr) {
		auto arr = luabinding::ClassUtil<numsky_ndarray>::check(L, stackPos);
		if(arr->dtype->typechar==typechar) {
			numsky_ndarray_copyto(arr, dataptr);
		} else {
			lnumsky_template_fp2(L, arr->dtype->typechar, typechar, numsky::ndarray_t_copyto)(arr, dataptr);
		}
	}

	LUA_API void lnsarr_copyfrom(lua_State*L, int stackPos, char typechar, char *dataptr) {
		auto arr = luabinding::ClassUtil<numsky_ndarray>::check(L, stackPos);
		if(arr->dtype->typechar==typechar) {
			numsky_ndarray_copyfrom(arr, dataptr);
		} else {
			lnumsky_template_fp2(L, arr->dtype->typechar, typechar, numsky::ndarray_t_copyfrom)(arr, dataptr);
		}
	}

	LUA_API bool lnsarr_isnsarr(lua_State*L, int stackPos) {
		auto arr = luabinding::ClassUtil<numsky_ndarray>::test(L, stackPos);
		if(arr == NULL) {
			return false;
		} else {
			return true;
		}
	}

	LUA_API npy_intp* lnsarr_shape(lua_State*L, int stackPos, size_t *nd) {
		auto arr = luabinding::ClassUtil<numsky_ndarray>::check(L, stackPos);
		nd[0] = arr->nd;
		return arr->dimensions;
	}

	LUA_API char lnsarr_typechar(lua_State*L, int stackPos) {
		auto arr = luabinding::ClassUtil<numsky_ndarray>::check(L, stackPos);
		return arr->dtype->typechar;
	}

}
